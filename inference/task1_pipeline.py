"""PENGWIN 2026 Task 1 — STU-Net 2-stage 추론 진입점 (V0.4, femur 활성화).

V0.4 는 V0.3.x 의 ResEnc 2-stage(Ds532 anatomy + Ds537 V291 fracture, femur zero-stub)
를 새 STU-Net 2-stage(Ds539 anatomy 5-class + Ds538 fracture, femur 활성화)로 교체한다.
ABBC 4-channel 헤드는 동일하므로 inline decode 는 그대로 재사용한다.

파이프라인:

    CT -> LPS 정규화 -> bone HU clip [-1000, 2000] -> bone-LUT 정규화
      -> Dataset539 anatomy classifier 추론 (full CT, 5-channel softmax:
         0=bg,1=Sacrum,2=LeftHip,3=RightHip,4=Femur)
      -> [Sacrum, LeftHip, RightHip, Femur] 각각에 대해:
           bbox = pad(anatomy_prob >= 0.5, pad_vox=24)
           3-channel 입력 구성 = [CT_LUT_crop, prob_crop, SDF(prob_crop)]
           Dataset538 STU-Net 추론 (ABBC 4-channel)
           디코딩 (core-seed watershed)
           PENGWIN 라벨 범위 [lo..hi] 로 재매핑 (Femur -> 151-200)
           전체 라벨 볼륨에 paste
      -> uint8 라벨 MHA 로 write

PENGWIN 케이스는 pelvic 이거나 femur (둘은 disjoint). pelvic 케이스는
Sacrum/LeftHip/RightHip 만 채워지고, femur 케이스는 Femur 만 채워진다.
femur ROI 는 bone-skeleton 분해(pelvic 전용)에 의존하지 않고 Ds539 argmax mask 에서
직접 얻는다.

모델 파일 레이아웃 (model.tar.gz 해제된 트리 구조):
    /opt/ml/model/nnunet/results/Dataset539_PelvicFemurAnatomyV3/
        PengwinTrainerSTUNetBaseAnatomyV301__nnUNetResEncUNetLPlans__3d_fullres/
        fold_0/checkpoint_best.pth
    /opt/ml/model/nnunet/results/Dataset538_PelvicFemurBICMFragmentV5/
        PengwinTrainerSTUNetBaseABBCPhase1V302__nnUNetResEncUNetLPlans__3d_fullres/
        fold_0/checkpoint_best.pth
        PengwinTrainerSTUNetBaseAffinityV308__nnUNetResEncUNetLPlans__3d_fullres/
        fold_0/checkpoint_best.pth
    /opt/ml/model/stage1_router/
        stage1_target_router_fold0.joblib

로컬 테스트: MODEL_ROOT 또는 PENGWIN_MODEL_ROOT 환경변수로 모델 루트를 덮어쓸 수
있다 (예: /workspace/code_task1/result -> result/results/...). 컨테이너 기본값은
/opt/ml/model 로 변하지 않는다.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import SimpleITK as sitk


INPUT_DIR = Path(os.environ.get("PENGWIN_INPUT_DIR", "/input/images/peripelvic-fracture-ct"))
OUTPUT_DIR = Path(os.environ.get("PENGWIN_OUTPUT_DIR", "/output/images/peripelvic-fracture-ct-segmentation"))

# Grand Challenge 의 모델 디렉터리 경로.
# model.tar.gz 는 trailing-dot convention (`tar -C model_payload -czf model.tar.gz .`)
# 으로 패킹되어 있어 내용물이 /opt/ml/model/ 바로 아래로 풀린다 (model_payload/ prefix 없음).
#
# 모델 루트는 환경변수로 덮어쓸 수 있다 (컨테이너 기본값은 /opt/ml/model 그대로):
#   - MODEL_ROOT / PENGWIN_MODEL_ROOT : 모델 루트 (하위에 nnunet/results/... 트리)
#   - nnUNet_results                  : nnUNet 결과 디렉터리를 직접 지정 (최우선)
# 로컬 테스트 예: nnUNet_results=/workspace/code_task1/result/results 로 두면
#   result/results/Dataset539_.../...__...__3d_fullres/fold_0/checkpoint_best.pth 를 찾는다.
MODEL_ROOT = Path(
    os.environ.get("MODEL_ROOT")
    or os.environ.get("PENGWIN_MODEL_ROOT")
    or "/opt/ml/model"
)
NN_RES = Path(os.environ.get("nnUNet_results", str(MODEL_ROOT / "nnunet" / "results")))
NN_PREP = Path(os.environ.get("nnUNet_preprocessed", str(MODEL_ROOT / "nnunet" / "preprocessed")))
NN_RAW = Path(os.environ.get("nnUNet_raw", str(MODEL_ROOT / "nnunet" / "raw")))
os.environ.setdefault("nnUNet_results", str(NN_RES))
os.environ.setdefault("nnUNet_preprocessed", str(NN_PREP))
os.environ.setdefault("nnUNet_raw", str(NN_RAW))
os.environ.setdefault("PENGWIN_ROOT", str(MODEL_ROOT))

# --- Dataset539 (STU-Net): anatomy classifier (5-class). ---
#     0=background, 1=Sacrum, 2=LeftHip, 3=RightHip, 4=Femur
#     (dataset.json labels 및 anatomy_registry.ANATOMY_REGISTRY 와 일치 확인됨).
DS539_DATASET = "Dataset539_PelvicFemurAnatomyV3"
DS539_TRAINER = os.environ.get("PENGWIN_DS539_TRAINER", "PengwinTrainerSTUNetBaseAnatomyV301")
DS539_PLANS = "nnUNetResEncUNetLPlans"
DS539_CONFIG = "3d_fullres"
# fold may be an int ("0") or the string "all" (a -f all / fold_all model). Default "0" = v1.5 behavior.
DS539_FOLD = os.environ.get("PENGWIN_DS539_FOLD", "0")
DS539_OUTPUT_CHANNELS = 5  # background, Sacrum, LeftHip, RightHip, Femur

# --- Dataset538 (STU-Net): anatomy 별 ABBC instance segmenter (V302, leak-free). ---
#     [v1.4] LEAK-FREE: V302 trainer = CT-ONLY 1-channel input (the old sidecar trainer fed a
#     3-channel [CT, Ds539-anatomy-prob, SDF] tensor — the anatomy-prob channel was Stage-A→B
#     leakage). V302 drops it; Stage-B input is now just the bone-LUT CT crop. ABBC 4-class head
#     is identical (0=bg,1=border,2=boundary,3=core) so the inline core-seed watershed decode is
#     reused unchanged. Localization bbox still comes from the Ds539 anatomy prob (routing, not a
#     model input channel — that is legitimate cascade routing, not leakage).
DS538_DATASET = "Dataset538_PelvicFemurBICMFragmentV5"
DS538_TRAINER = os.environ.get("PENGWIN_DS538_TRAINER", "PengwinTrainerSTUNetBaseABBCPhase1V302")
DS538_PLANS = "nnUNetResEncUNetLPlans"
DS538_CONFIG = "3d_fullres"
DS538_FOLD = os.environ.get("PENGWIN_DS538_FOLD", "0")  # int-string or "all" (fold_all); default "0" = v1.5
DS538_OUTPUT_CHANNELS = int(os.environ.get("PENGWIN_DS538_OUT_CH", "4"))  # ABBC 4ch (V302); 13 for V307 affinity head (4 ABBC + 9 affinity)
CHECKPOINT_NAME = "checkpoint_best.pth"

# --- anatomy contract (anatomy_registry 와 일치).
#     Ds539 5-class softmax 의 fg 채널 index = anatomy semantic class index.
#     PENGWIN 전역 instance-ID 범위: Sacrum 1-50, LeftHip 51-100, RightHip 101-150,
#     Femur 151-200. (anatomy_registry.ANATOMY_REGISTRY 와 동일.)
DS539_PROB_CHANNEL = {"Sacrum": 1, "LeftHip": 2, "RightHip": 3, "Femur": 4}
ANATOMY_RANGES = {
    "Sacrum": (1, 50),
    "LeftHip": (51, 100),
    "RightHip": (101, 150),
    "Femur": (151, 200),
}
# 전체 anatomy 순서 (per-anatomy 추론 루프 순서). femur 활성화.
ALL_ANATOMIES = ("Sacrum", "LeftHip", "RightHip", "Femur")
# bone-skeleton 분해는 pelvic 전용(중앙/좌/우 CC 분류)이므로 femur 는 제외한다.
PELVIC_ANATOMIES = ("Sacrum", "LeftHip", "RightHip")

# --- v2.0 target-family router. ---
# The code is committed, but the joblib artifact is packaged in model.tar.gz.
# If enabled, missing/incompatible router artifacts fail fast to avoid silently
# deploying the old Ds539 volume-ratio route.
TARGET_ROUTER_ENABLED = os.environ.get("PENGWIN_TARGET_ROUTER", "0") == "1"
# [v2.4] Below this RF decision margin (|2p-0.5|) the router stops trusting itself and takes a
# 3-way vote with the official rule + Stage-1 anatomy evidence. Measured min margin over all 340
# training cases = 0.9052 (zero cases below 0.90), so the default 0.85 never fires on known data
# and the deployed routing is unchanged. 0 disables the gate entirely.
ROUTER_ABSTAIN_MARGIN = float(os.environ.get("PENGWIN_ROUTER_ABSTAIN_MARGIN", "0.85"))
TARGET_ROUTER_PATH = Path(
    os.environ.get(
        "PENGWIN_TARGET_ROUTER_PATH",
        str(MODEL_ROOT / "stage1_router" / "stage1_target_router_fold0.joblib"),
    )
)
_TARGET_ROUTER_PAYLOAD = None

# --- anatomy routing: process every anatomy whose Ds539 argmax mask is a substantial
#     fraction of the largest present mask (see route_from_ds539_masks). ---
ROI_PAD_VOX = 24
SDF_CLIP_MM = 40.0
BONE_HU_RANGE = (-1000.0, 2000.0)

# --- ABBC decode 하이퍼파라미터 (V0.1 description.md 의 "bw=10 + sr=0.05" 와 동일). ---
BACKGROUND_THRESHOLD = 0.50
CORE_THRESHOLD = 0.50
# [v1.3.3 DEC-2] non-Sacrum size-ratio merge DISABLED (was 0.05). At 0.05 any fragment
# smaller than 5% of the bone's largest fragment was merged away — for a ~400k-voxel hip
# that cutoff (~20k vox) swallowed genuine 1-11 cm^3 fracture fragments, causing the GC
# under-segmentation (instance_recall 0.456, merge_error 0.65, split_error 0). The validated
# offline eval (eval.py) uses 0.0 for non-Sacrum; the deploy decoder had silently diverged.
# Sacrum keeps its deliberate 0.10 precision guard via the max(...) at its call site.
ANATOMY_SIZE_RATIO_KEEP = 0.0
MIN_COMPONENT_VOXELS = 1820  # V5 plan 해상도에서 약 1 cm^3 에 해당

# --- V0.3 견고화 관련 플래그 ---
V03_USE_BONE_SKELETON_FALLBACK = True  # Ds539 실패 시 bone HU mask fallback 사용
V03_USE_TIME_BUDGET = True             # 시간 예산 기반 안전망 활성화

# --- V0.3 견고화 관련 상수 ---
# 시간 예산 설정 (Grand Challenge 의 10분 제한을 안전하게 지키기 위함)
TIME_BUDGET_SECONDS = 480.0   # 8분: I/O / decode / paste 여유분 포함
TIME_BUDGET_HARD_LIMIT = 540.0   # 9분: 이 시점 이후로는 남은 anatomy 모두 skip
BONE_HU_THRESHOLD = 200.0
BONE_MIN_COMPONENT_VOXELS = 10000
SANITY_MAX_BBOX_FRACTION = 0.35
# Routing CC policy is now env-driven at the use-site: PENGWIN_ROUTE_CC_MODE in
# {largest (default, = old LARGEST_CC_KEEP_ONLY=True), union, floor}; floor uses
# PENGWIN_ROUTE_CC_MIN_VOX (default MIN_COMPONENT_VOXELS). See run_per_anatomy().
MIN_DS539_CC_VOXELS = 500  # Ds539 mask 의 sparse outlier CC 제거 임계값 (voxels)
DS539_MORPH_OPENING_ITERS = 1  # Ds539 mask 에 적용할 binary_opening 반복 횟수


def log(msg: str) -> None:
    print(f"[pengwin_v0] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 입출력 관련 헬퍼.
# ---------------------------------------------------------------------------
def find_input_image() -> Path:
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"input directory missing: {INPUT_DIR}")
    candidates = sorted(p for p in INPUT_DIR.iterdir() if p.suffix in {".mha", ".mhd"})
    if not candidates:
        raise FileNotFoundError(f"no .mha input under {INPUT_DIR}; got {list(INPUT_DIR.iterdir())}")
    return candidates[0]


def output_path(input_path: Path) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / input_path.name


def write_label_map(label_arr: np.ndarray, ref_img: sitk.Image, out_path: Path) -> None:
    """ref_img 의 geometry (spacing/origin/direction) 를 유지하며 라벨 MHA 를 저장한다.

    값은 [0, 200] 으로 clip 후 uint8 로 캐스팅한다 (PENGWIN 라벨 범위 보장).
    """
    if label_arr.ndim != 3:
        raise ValueError(f"expected [Z, Y, X] label map, got {label_arr.shape}")
    label_arr = np.clip(label_arr, 0, 200).astype(np.uint8, copy=False)
    out_img = sitk.GetImageFromArray(label_arr)
    out_img.SetSpacing(ref_img.GetSpacing())
    out_img.SetOrigin(ref_img.GetOrigin())
    out_img.SetDirection(ref_img.GetDirection())
    uniq = np.unique(label_arr)
    log(
        f"write_label_map: dtype={label_arr.dtype} shape={label_arr.shape} "
        f"n_unique={len(uniq)} unique_head={uniq[:10].tolist()} -> {out_path}"
    )
    sitk.WriteImage(out_img, str(out_path), useCompression=False)


def write_zero_output(input_path: Path, ref_img: sitk.Image, reason: str) -> None:
    out_path = output_path(input_path)
    zero = np.zeros(sitk.GetArrayFromImage(ref_img).shape, dtype=np.uint8)
    write_label_map(zero, ref_img, out_path)
    log(f"wrote ALL-ZERO output ({reason}) -> {out_path}")


# ---------------------------------------------------------------------------
# 이미지 전처리 헬퍼들. preprocessing.py / utils.py 에서 그대로 복사해 왔다.
# 이렇게 함으로써 런타임에 /opt/app/code_task1 에 의존하지 않아도 되도록 한다.
# ---------------------------------------------------------------------------
def orientation_code(img: sitk.Image) -> str:
    return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img.GetDirection())


def canonicalize_sitk(img: sitk.Image, target: str = "LPS") -> sitk.Image:
    code = orientation_code(img)
    if code == target:
        return img
    return sitk.DICOMOrient(img, target)


def bone_window_clip(arr: np.ndarray,
                     hu_low: float = BONE_HU_RANGE[0],
                     hu_high: float = BONE_HU_RANGE[1]) -> np.ndarray:
    return np.clip(arr.astype(np.float32, copy=False), float(hu_low), float(hu_high))


def canonicalize_and_clip_image(img: sitk.Image) -> tuple[sitk.Image, np.ndarray]:
    img_lps = canonicalize_sitk(img, target="LPS")
    arr = bone_window_clip(sitk.GetArrayFromImage(img_lps))
    return img_lps, arr


def bone_lut_normalize(arr: np.ndarray) -> np.ndarray:
    """결정론적 bone-window LUT (preprocessing._bone_lut_normalize 의 동일 구현).

    학습 시 사용한 piecewise-linear HU → [0,1] 매핑을 그대로 재현해 train/inference
    contract 가 어긋나지 않도록 한다.
    """
    x = arr.astype(np.float32, copy=False)
    xp = np.asarray([-1000.0, -200.0, 150.0, 700.0, 1500.0, 2000.0], dtype=np.float32)
    fp = np.asarray([0.0, 0.05, 0.35, 0.70, 0.92, 1.0], dtype=np.float32)
    return np.interp(np.clip(x, xp[0], xp[-1]), xp, fp).astype(np.float32)


def ndi_distance_transform(mask: np.ndarray,
                           spacing_zyx: tuple[float, float, float]) -> np.ndarray:
    from scipy import ndimage as ndi

    return ndi.distance_transform_edt(mask, sampling=spacing_zyx).astype(np.float32, copy=False)


def selected_anatomy_sdf_from_prob(prob: np.ndarray,
                                   spacing_zyx: tuple[float, float, float],
                                   clip_mm: float = SDF_CLIP_MM) -> np.ndarray:
    """anatomy 확률 prob>=0.5 mask 로부터 signed distance 를 계산해 [-1, 1] 범위로 스케일링한다.

    mask 내부는 양수, 외부는 음수가 되어 Ds538 모델이 위치 정보를 학습된 형태로 받을 수 있다.
    """
    arr = np.asarray(prob, dtype=np.float32)
    mask = arr >= 0.5
    if not mask.any():
        return np.zeros_like(arr, dtype=np.float32)
    inside = ndi_distance_transform(mask, spacing_zyx)
    outside = ndi_distance_transform(~mask, spacing_zyx)
    sdf = np.clip(inside - outside, -float(clip_mm), float(clip_mm)) / float(clip_mm)
    return sdf.astype(np.float32, copy=False)


def bbox_from_mask(mask: np.ndarray, pad_vox: int = ROI_PAD_VOX) -> tuple[slice, slice, slice] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    pad = int(max(0, pad_vox))
    lo = np.maximum(coords.min(axis=0) - pad, 0)
    hi = np.minimum(coords.max(axis=0) + pad + 1, np.asarray(mask.shape))
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# nnU-Net predictor 관련 헬퍼 (출력 채널 수에 일반화된 형태).
# ---------------------------------------------------------------------------
def build_predictor(dataset: str, trainer: str, plans: str, config: str,
                    fold: int, checkpoint_name: str = CHECKPOINT_NAME):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=use_cuda,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    model_dir = NN_RES / dataset / f"{trainer}__{plans}__{config}"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"trained model folder missing: {model_dir}")
    predictor.initialize_from_trained_model_folder(
        str(model_dir),
        # accept both numeric folds ("0") and the all-data model ("all" -> fold_all dir)
        use_folds=(("all" if str(fold).strip().lower() == "all" else int(fold)),),
        checkpoint_name=checkpoint_name,
    )
    # CRITICAL (nnUNet version-independence): nnUNet 2.5.1 — the version pinned for the GC
    # container — builds the network in initialize_from_trained_model_folder() but does NOT
    # apply the checkpoint to predictor.network. It only stashes the weights in
    # predictor.list_of_parameters and defers the load to perform_actual_prediction().
    # Our custom inference path (_predict_custom_logits_from_preprocessed_data) reads
    # predictor.network directly and never calls that method, so on 2.5.1 the network would
    # run with its RANDOM initialization -> catastrophic speckle anatomy (tens of thousands
    # of CCs) -> ~0 score. nnUNet 2.5.2+ loads at init (MIC-DKFZ/nnUNet#2520), which is why
    # the local 2.5.2 env was clean while the 2.5.1 GC container scored 0. Load the trained
    # weights into the network explicitly so this is correct on ANY nnUNet version.
    if getattr(predictor, "list_of_parameters", None):
        predictor.network.load_state_dict(predictor.list_of_parameters[0])
    # DIAG: confirm the RIGHT architecture (STUNet, not ResEnc) + non-random weights loaded.
    # A wrong/random network here is exactly the 73%-one-class / thousands-of-CC garbage signature.
    try:
        import torch as _t
        _net = predictor.network
        _ps = list(_net.parameters())
        _nparam = sum(p.numel() for p in _ps)
        with _t.no_grad():
            _w0 = float(_ps[0].detach().float().abs().sum().cpu()) if _ps else 0.0
        log(f"build_predictor: {dataset} {trainer[-30:]} NET={type(_net).__name__} "
            f"params={_nparam/1e6:.1f}M w0sum={_w0:.4e} device={device}")
    except Exception as _e:
        log(f"build_predictor: {dataset} {trainer[-30:]} device={device} (net-diag err: {_e})")
    return predictor


def predict_logits_full(predictor, image_4d: np.ndarray,
                        image_properties: dict, output_channels: int) -> tuple[np.ndarray, dict]:
    """(C, Z, Y, X) 배열에 sliding-window 추론을 수행해 logit 과 data_properties 를 반환한다.

    image_4d 의 입력 채널 수는 학습된 모델의 plans.channel_names 와 정확히 일치해야 한다.
    그렇지 않으면 encoder 가 입력을 거부한다.
    """
    from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy

    if image_4d.ndim != 4:
        raise ValueError(f"expected (C, Z, Y, X) input, got {image_4d.shape}")
    image_4d = image_4d.astype(np.float32, copy=False)
    ppa = PreprocessAdapterFromNpy(
        [image_4d],
        [None],
        [image_properties],
        [None],
        predictor.plans_manager,
        predictor.dataset_json,
        predictor.configuration_manager,
        num_threads_in_multithreaded=1,
        verbose=False,
    )
    dct = next(ppa)
    data = dct["data"]
    import torch
    if torch.is_tensor(data):
        data_tensor = data.float()
    else:
        data_tensor = torch.from_numpy(np.asarray(data)).float()
    logits = _predict_custom_logits_from_preprocessed_data(
        predictor, data_tensor, output_channels=output_channels,
    )
    logits_np = logits.detach().cpu().numpy().astype(np.float32, copy=False)
    return logits_np, dct["data_properties"]


def _predict_custom_logits_from_preprocessed_data(predictor, data, output_channels: int):
    """custom head 의 출력 채널 수에 맞춰 logit 버퍼를 할당하는 sliding-window 추론.

    nnUNet 기본 구현은 plans.json 의 num_classes 에 의존하지만, 본 프로젝트는
    ABBC 4-channel 헤드처럼 비표준 출력 채널을 사용하므로 output_channels 를
    명시적으로 주입한다.
    """
    import torch
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian
    from nnunetv2.utilities.helpers import dummy_context, empty_cache

    def _internal_predict(data_pad, slicers, do_on_device):
        results_device = predictor.device if do_on_device else torch.device("cpu")
        empty_cache(predictor.device)
        data_work = data_pad.to(results_device)
        predicted_logits = torch.zeros(
            (int(output_channels), *data_work.shape[1:]),
            dtype=torch.half, device=results_device,
        )
        n_predictions = torch.zeros(
            data_work.shape[1:], dtype=torch.half, device=results_device,
        )
        gaussian = compute_gaussian(
            tuple(predictor.configuration_manager.patch_size),
            sigma_scale=1.0 / 8,
            value_scaling_factor=10,
            device=results_device,
        ) if predictor.use_gaussian else 1
        for sl in slicers:
            workon = data_work[sl][None].to(predictor.device)
            prediction = predictor._internal_maybe_mirror_and_predict(workon)[0].to(results_device)
            if int(prediction.shape[0]) != int(output_channels):
                raise RuntimeError(
                    "Custom nnU-Net head mismatch: "
                    f"expected {output_channels} channels, got {tuple(prediction.shape)}"
                )
            if predictor.use_gaussian:
                prediction *= gaussian
            predicted_logits[sl] += prediction
            n_predictions[sl[1:]] += gaussian
        predicted_logits /= n_predictions
        if torch.any(torch.isinf(predicted_logits)):
            raise RuntimeError("inf in custom-head predicted logits")
        return predicted_logits

    with torch.no_grad():
        if data.ndim != 4:
            raise ValueError(f"expected [C, Z, Y, X], got {tuple(data.shape)}")
        predictor.network = predictor.network.to(predictor.device)
        predictor.network.eval()
        empty_cache(predictor.device)
        autocast_ctx = (
            torch.autocast(predictor.device.type, enabled=True)
            if predictor.device.type == "cuda" else dummy_context()
        )
        with autocast_ctx:
            data_pad, slicer_revert = pad_nd_image(
                data,
                predictor.configuration_manager.patch_size,
                "constant", {"value": 0}, True, None,
            )
            slicers = predictor._internal_get_sliding_window_slicers(data_pad.shape[1:])
            do_on_device = (
                predictor.perform_everything_on_device
                and predictor.device.type != "cpu"
            )
            try:
                predicted_logits = _internal_predict(data_pad, slicers, do_on_device)
            except RuntimeError as exc:
                log(f"on-device predict failed; retrying with CPU buffer: {exc}")
                empty_cache(predictor.device)
                predicted_logits = _internal_predict(data_pad, slicers, False)
            empty_cache(predictor.device)
            predicted_logits = predicted_logits[(slice(None), *slicer_revert[1:])]
    return predicted_logits


def softmax_axis0(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float32)
    arr = arr - np.max(arr, axis=0, keepdims=True)
    exp = np.exp(arr).astype(np.float32, copy=False)
    return exp / np.maximum(np.sum(exp, axis=0, keepdims=True), np.float32(1e-8))


# ---------------------------------------------------------------------------
# ABBC 확률 → fragment instance 라벨 디코더 (V288 core-seed watershed).
# ---------------------------------------------------------------------------
def decode_abbc_core_seed_watershed(
    probs: np.ndarray,
    *,
    background_threshold: float = BACKGROUND_THRESHOLD,
    core_threshold: float = CORE_THRESHOLD,
    min_component_voxels: int = MIN_COMPONENT_VOXELS,
    size_ratio_keep: float = ANATOMY_SIZE_RATIO_KEEP,
) -> np.ndarray:
    """V288 core-seed watershed 로 ABBC 확률맵을 fragment instance 라벨로 디코딩한다.

    ABBC 채널 의미: 0=background, 1=border, 2=boundary, 3=core.
    core 영역을 seed 로 두고 background 가 아닌 영역(support) 전체로 watershed 를 확장한다.
    반환값은 uint16 라벨 1..N (배경은 0).
    """
    import scipy.ndimage as ndi

    background = probs[0] >= float(background_threshold)
    support = ~background
    if not support.any():
        return np.zeros(probs.shape[1:], dtype=np.uint16)
    core = (probs[3] >= float(core_threshold)) & support
    core_labels, n_core = ndi.label(core, structure=np.ones((3, 3, 3), dtype=bool))
    core_labels = core_labels.astype(np.int32, copy=False)
    if int(n_core) <= 0:
        labels = np.where(support, 1, 0).astype(np.int32, copy=False)
    else:
        from skimage.segmentation import watershed as _ski_watershed
        priority = ndi.distance_transform_edt(core_labels == 0)
        labels = _ski_watershed(
            priority, markers=core_labels, mask=support,
        ).astype(np.int32, copy=False)
        fill_mask = support & (labels == 0)
        if fill_mask.any():
            nearest_indices = ndi.distance_transform_edt(
                labels == 0, return_indices=True,
            )[1]
            labels[fill_mask] = labels[
                tuple(idx[fill_mask] for idx in nearest_indices)
            ]
    decoded = _merge_small_components(labels, min_component_voxels=int(min_component_voxels))
    if float(size_ratio_keep) > 0.0:
        decoded = _merge_by_size_ratio(decoded, size_ratio_keep=float(size_ratio_keep))
    return decoded.astype(np.uint16, copy=False)


def _merge_small_components(labels: np.ndarray, *, min_component_voxels: int) -> np.ndarray:
    import scipy.ndimage as ndi

    out = np.asarray(labels, dtype=np.int32).copy()
    if int(min_component_voxels) <= 1:
        return out
    ids, counts = np.unique(out, return_counts=True)
    small_ids = [int(i) for i, c in zip(ids, counts) if int(i) > 0 and int(c) < int(min_component_voxels)]
    if not small_ids:
        return out
    small_mask = np.isin(out, small_ids)
    out[small_mask] = 0
    if (out == 0).any() and (out > 0).any():
        original_support = labels > 0
        refill_mask = original_support & (out == 0)
        if refill_mask.any():
            non_zero = out > 0
            if non_zero.any():
                nearest = ndi.distance_transform_edt(~non_zero, return_indices=True)[1]
                out[refill_mask] = out[tuple(idx[refill_mask] for idx in nearest)]
    return out


def _merge_by_size_ratio(labels: np.ndarray, *, size_ratio_keep: float) -> np.ndarray:
    import scipy.ndimage as ndi

    out = np.asarray(labels, dtype=np.int32).copy()
    ids, counts = np.unique(out, return_counts=True)
    pairs = [(int(i), int(c)) for i, c in zip(ids, counts) if int(i) > 0]
    if len(pairs) <= 1:
        return out
    largest = max(c for _, c in pairs)
    threshold = float(size_ratio_keep) * float(largest)
    small_ids = [i for i, c in pairs if float(c) < threshold]
    if not small_ids:
        return out
    refill_mask = np.isin(out, small_ids)
    out[refill_mask] = 0
    non_zero = out > 0
    if non_zero.any() and refill_mask.any():
        nearest = ndi.distance_transform_edt(~non_zero, return_indices=True)[1]
        out[refill_mask] = out[tuple(idx[refill_mask] for idx in nearest)]
    return out


# ---------------------------------------------------------------------------
# fragment 단위 라벨 맵을 preprocessed grid 에서 원본 CT 격자로 resampling.
# ---------------------------------------------------------------------------
def resample_label_map_to_original(
    label_pp: np.ndarray, data_properties: dict, predictor,
) -> np.ndarray:
    from nnunetv2.preprocessing.resampling.default_resampling import (
        resample_data_or_seg_to_shape,
    )

    label_in = label_pp.astype(np.uint16, copy=False)[None]

    shape_after_cropping = tuple(
        int(s) for s in data_properties["shape_after_cropping_and_before_resampling"]
    )
    current_spacing = predictor.configuration_manager.spacing
    transpose_forward = predictor.plans_manager.transpose_forward
    original_spacing = [data_properties["spacing"][i] for i in transpose_forward]

    resampled = resample_data_or_seg_to_shape(
        label_in.astype(np.float32, copy=False),
        shape_after_cropping,
        current_spacing,
        original_spacing,
        is_seg=True,
        order=1,
        order_z=0,
    )
    resampled_np = np.asarray(resampled).astype(np.uint16, copy=False)[0]

    from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice

    shape_before_cropping = tuple(
        int(s) for s in data_properties["shape_before_cropping"]
    )
    bbox = data_properties.get("bbox_used_for_cropping", None)
    if bbox is None:
        full = resampled_np
    else:
        full = np.zeros(shape_before_cropping, dtype=np.uint16)
        slicer = bounding_box_to_slice(bbox)
        full[slicer] = resampled_np

    transpose_backward = predictor.plans_manager.transpose_backward
    full = np.transpose(full, tuple(transpose_backward))
    return full


# ---------------------------------------------------------------------------
# V0.3 견고화된 anatomy ROI 추출 헬퍼 (4-layer 파이프라인용).
# ---------------------------------------------------------------------------
def bone_skeleton_anatomy_decomposition(arr_clipped, spacing_zyx,
                                        hu_threshold=200.0,
                                        min_component_voxels=10000):
    """Layer 1: 학습 데이터 분포에 의존하지 않는 anatomy ROI 분해 (bone HU mask 기반).

    PENGWIN CT 의 골격 부위는 HU > 200 으로 안정적으로 분리 가능하다.
    Ds532 가 OOD (out-of-distribution) 케이스에서 실패하더라도, 본 함수는 항상
    합리적인 Sacrum/LeftHip/RightHip ROI 를 추출할 수 있게 해 주는 안전망이다.

    Args:
        arr_clipped: (Z, Y, X) HU 배열 (이미 [-1000, 2000] 로 clip 된 상태)
        spacing_zyx: physical spacing (z, y, x) mm
        hu_threshold: bone 으로 간주할 최소 HU 값
        min_component_voxels: 너무 작은 connected component 는 무시할 임계값

    Returns:
        dict[str, np.ndarray] — anatomy 별 mask (LPS frame 기준).
        Sacrum 은 중앙 CC, LeftHip 은 +x 쪽, RightHip 은 -x 쪽으로 배정한다.
        추출 실패 시 빈 dict 반환.
    """
    import scipy.ndimage as ndi
    bone_mask = arr_clipped > float(hu_threshold)
    # Morphology 후처리: opening 으로 노이즈 제거하면서 작은 gap 도 함께 정리
    bone_mask = ndi.binary_opening(bone_mask, structure=np.ones((2, 2, 2), dtype=bool), iterations=1)
    labels, n_cc = ndi.label(bone_mask, structure=np.ones((3, 3, 3), dtype=bool))
    if n_cc == 0:
        return {}
    # connected component 를 크기 내림차순으로 정렬
    sizes = ndi.sum(bone_mask, labels, index=np.arange(1, n_cc + 1))
    sorted_idx = np.argsort(sizes)[::-1]
    top_cc_ids = [int(sorted_idx[i]) + 1 for i in range(min(5, n_cc)) if sizes[sorted_idx[i]] >= min_component_voxels]
    if not top_cc_ids:
        return {}
    # 각 CC 의 centroid (z, y, x) 좌표 계산
    centroids_raw = ndi.center_of_mass(bone_mask, labels, top_cc_ids)
    # scipy 버전/입력 개수에 따라 반환 형식이 달라지므로 3-tuple 의 list 로 정규화한다.
    if len(top_cc_ids) == 1:
        # index 가 1개일 때 scipy 가 단일 tuple 을 반환하는 경우가 있어 list 로 감싼다
        if isinstance(centroids_raw, tuple) and len(centroids_raw) == 3 and not isinstance(centroids_raw[0], tuple):
            centroids = [centroids_raw]
        else:
            centroids = list(centroids_raw)
    else:
        centroids = list(centroids_raw)
    # x-centroid 로 좌/중/우 분류 (LPS 좌표계에서는 +x 가 patient left)
    x_dim = float(arr_clipped.shape[2])
    cc_info = []
    for i, cc_id in enumerate(top_cc_ids):
        c = centroids[i]
        if not (hasattr(c, '__len__') and len(c) == 3):
            log(f"bone-skeleton: CC {cc_id} centroid malformed ({c}), skipping")
            continue
        cz, cy, cx = float(c[0]), float(c[1]), float(c[2])
        cc_info.append({
            'cc_id': int(cc_id),
            'cx': float(cx),
            'cy': float(cy),
            'cz': float(cz),
            'size': int(sizes[cc_id - 1]),
        })
    # 크기 내림차순으로 정렬
    cc_info.sort(key=lambda d: -d['size'])

    # 가장 큰 3개 CC 를 사용해 x-centroid 로 anatomy 를 분류한다:
    #   가장 중앙에 가까운 CC = Sacrum
    #   +x 쪽 (이미지 좌측 절반) = LeftHip
    #   -x 쪽 (이미지 우측 절반) = RightHip
    if len(cc_info) >= 3:
        top3 = cc_info[:3]
    else:
        top3 = cc_info  # 3개 미만이면 가능한 만큼만 사용 (best effort)

    # 이미지 중앙 x 좌표
    cx_center = x_dim / 2.0
    # 중앙성 점수 = abs(cx - cx_center). 작을수록 중앙에 가깝다.
    top3_sorted_by_centrality = sorted(top3, key=lambda d: abs(d['cx'] - cx_center))
    if len(top3_sorted_by_centrality) >= 1:
        sacrum_cc = top3_sorted_by_centrality[0]
    else:
        sacrum_cc = None

    remaining = [d for d in top3 if d['cc_id'] != (sacrum_cc['cc_id'] if sacrum_cc else None)]
    if len(remaining) >= 2:
        remaining.sort(key=lambda d: d['cx'])  # x 좌표 오름차순 정렬
        righthip_cc = remaining[0]   # x 가 작은 쪽 = patient right (LPS)
        lefthip_cc = remaining[-1]   # x 가 큰 쪽 = patient left (LPS)
    elif len(remaining) == 1:
        # hip CC 가 1개만 보일 때 centroid 부호를 보고 좌/우 결정
        only_hip = remaining[0]
        if only_hip['cx'] < cx_center:
            righthip_cc = only_hip
            lefthip_cc = None
        else:
            lefthip_cc = only_hip
            righthip_cc = None
    else:
        lefthip_cc = righthip_cc = None

    masks = {}
    if sacrum_cc is not None:
        masks['Sacrum'] = (labels == sacrum_cc['cc_id'])
    if lefthip_cc is not None:
        masks['LeftHip'] = (labels == lefthip_cc['cc_id'])
    if righthip_cc is not None:
        masks['RightHip'] = (labels == righthip_cc['cc_id'])
    log(f"bone-skeleton: {len(masks)} anatomies extracted ({list(masks.keys())})")
    return masks


def ds539_argmax_masks(probs_full):
    """Ds539 5-channel softmax → 서로 배타적인 anatomy mask + morphology cleanup.

    Layer 2 일부: Ds539 5-channel softmax 의 argmax 로 서로 배타적인 anatomy mask 를
    추출하고, sparse outlier 잡음을 제거하기 위해 binary_opening + small CC drop 을
    적용한다. 이는 V0.3.1 GC failure (LeftHip bbox 98.6% / RightHip bbox 100%) 처럼
    largest-CC keep 만으로는 막을 수 없는 sparse fragment 연결 문제를 차단한다.

    채널: 0=bg, 1=Sacrum, 2=LeftHip, 3=RightHip, 4=Femur. Femur 포함 4개 anatomy
    모두에 대해 mask 를 추출한다 (pelvic 케이스는 Femur mask 가 비고, femur 케이스는
    Sacrum/Hip mask 가 비는 식으로 자연히 disjoint 하게 갈린다).

    Args:
        probs_full: (5, Z, Y, X) softmax 확률 텐서.

    Returns:
        dict[str, np.ndarray] — anatomy 별 mask (서로 disjoint 임이 보장됨).
    """
    import scipy.ndimage as ndi
    argmax_map = np.argmax(probs_full, axis=0)  # (Z,Y,X) 0=bg,1=sacrum,2=lh,3=rh,4=femur
    masks = {}
    structure = np.ones((2, 2, 2), dtype=bool)
    cc_struct = np.ones((3, 3, 3), dtype=bool)
    for anatomy, ch_idx in DS539_PROB_CHANNEL.items():
        raw_mask = (argmax_map == ch_idx)
        # Morphology cleanup: opening 으로 얇거나 고립된 노이즈를 제거한다
        if DS539_MORPH_OPENING_ITERS > 0:
            cleaned = ndi.binary_opening(raw_mask, structure=structure,
                                          iterations=DS539_MORPH_OPENING_ITERS)
        else:
            cleaned = raw_mask
        # 작은 CC 제거 (Ds539 sparse outlier)
        cc_labels, n_cc = ndi.label(cleaned, structure=cc_struct)
        if n_cc > 1:
            sizes = ndi.sum(cleaned, cc_labels, index=np.arange(1, n_cc + 1))
            keep_ids = np.where(sizes >= MIN_DS539_CC_VOXELS)[0] + 1
            if len(keep_ids) > 0:
                cleaned = np.isin(cc_labels, keep_ids)
            else:
                cleaned = np.zeros_like(cleaned)
        elif n_cc == 1:
            if int(cleaned.sum()) < MIN_DS539_CC_VOXELS:
                cleaned = np.zeros_like(cleaned)
        masks[anatomy] = cleaned
        log(f"Ds539 argmax: {anatomy} raw={int(raw_mask.sum())} cleaned={int(cleaned.sum())} "
            f"({n_cc} CCs, drop<{MIN_DS539_CC_VOXELS})")
    return masks


def merge_masks_with_sanity(ds539_masks, bone_masks, image_shape,
                            sanity_max_fraction=0.35,
                            bone_fallback_when_sanity_fails=True,
                            anatomies=ALL_ANATOMIES):
    """Layer 2+3: Ds539 mask 와 bone-skeleton mask 를 sanity check 와 함께 병합한다.

    각 anatomy 별 정책:
      1) Ds539 mask 의 voxel 비율이 sanity_max_fraction 미만이면 Ds539 사용 (정확도 우선)
      2) sanity 실패 시 bone skeleton mask 로 fallback (가용한 경우; pelvic 전용)
      3) 둘 다 실패 → 해당 anatomy 는 skip 하고 라벨을 0 으로 둔다

    Femur 는 bone-skeleton 분해(중앙/좌/우 CC)가 만들어 주지 않으므로 bone fallback 이
    없다 → Ds539 argmax mask 에만 의존한다 (femur 케이스는 pelvic 과 disjoint).

    반환:
        dict[str, dict] — { anatomy: { 'mask': arr, 'source': 'ds539' | 'bone_fallback' | 'none' } }
    """
    img_voxels = float(np.prod(image_shape))
    out = {}
    for anatomy in anatomies:
        ds_mask = ds539_masks.get(anatomy)
        bone_mask = bone_masks.get(anatomy)
        ds_fraction = float(ds_mask.sum()) / img_voxels if ds_mask is not None else 1.0
        if ds_mask is not None and ds_mask.any() and ds_fraction < sanity_max_fraction:
            out[anatomy] = {'mask': ds_mask, 'source': 'ds539', 'fraction': ds_fraction}
        elif bone_mask is not None and bone_fallback_when_sanity_fails:
            bone_fraction = float(bone_mask.sum()) / img_voxels
            out[anatomy] = {'mask': bone_mask, 'source': 'bone_fallback', 'fraction': bone_fraction}
            log(f"[{anatomy}] Ds539 sanity fail (covers {ds_fraction*100:.1f}%), bone-skeleton fallback ({bone_fraction*100:.1f}%)")
        else:
            out[anatomy] = {'mask': None, 'source': 'none', 'fraction': ds_fraction}
            reason = ("Ds539 empty" if (ds_mask is None or not ds_mask.any())
                      else f"Ds539 sanity fail ({ds_fraction*100:.1f}%)")
            log(f"[{anatomy}] {reason} and no bone-skeleton fallback — anatomy will be zero")
    return out


def estimate_fracture_inference_seconds(bbox, patch_size=(192, 160, 224), seconds_per_patch=2.5):
    """Layer 3: Ds538 sliding-window 추론 소요 시간 대략 추정.

    nnUNet 의 sliding window 는 tile_step_size=0.5 이므로 patch 의 50% 씩 이동한다.

    ⚠️ [2026-07-21 감사] "plan 의 patch_size [192,160,224] 와 일치" 라는 종전 서술은 **거짓**이다.
    V308/V302 의 plans.json 실측 patch_size 는 **[256,160,160]** 이고, 이 함수는 raw crop 을 타일링하는데
    실제 추론은 plan spacing 으로 리샘플된 crop 을 타일링한다(실 CT z=1.0mm vs plan 0.801mm → ×1.25).
    680개 실 bbox 측정: 평균 추정 4.99 타일 vs 실제 7.31 타일, ROI 의 70.6% 에서 **과소추정** =
    ETA 가 ~46% 낙관적이고 오차 방향이 안전하지 않다(가드가 늦게 걸린다).

    그래도 **값은 고치지 않는다**: 명목 pelvic 케이스가 ~51 타일 ≈ ~130s 로 480s 예산 대비 여유가 크고
    가드가 실제로 걸리지 않는다. 상수를 바꾸면 rank-10 배포본의 런타임 동작이 바뀐다(가드가 더 자주 걸려
    해부부위를 0 으로 내보낼 수 있다). **실질적 교훈은 부정형이다 — 추론에 비용 레버(Stage-1/2 앙상블,
    TTA, tile_step_size↓)를 추가하지 마라.** 실패 모드가 우아한 0 이 아니라 출력조차 없는 하드 타임아웃이다.
    축별 patch 개수 = ceil((dim - patch) / (patch * 0.5)) + 1.
    이를 곱한 총 patch 수에 patch 당 평균 추론 시간을 곱해 ETA 를 구한다.
    """
    if bbox is None:
        return 0.0
    crop_z = bbox[0].stop - bbox[0].start
    crop_y = bbox[1].stop - bbox[1].start
    crop_x = bbox[2].stop - bbox[2].start
    pz, py, px = patch_size
    n_z = max(1, int(np.ceil(max(0, crop_z - pz) / (pz * 0.5)) + 1))
    n_y = max(1, int(np.ceil(max(0, crop_y - py) / (py * 0.5)) + 1))
    n_x = max(1, int(np.ceil(max(0, crop_x - px) / (px * 0.5)) + 1))
    n_patches = n_z * n_y * n_x
    return float(n_patches) * float(seconds_per_patch)


def route_from_ds539_masks(ds539_masks: dict) -> tuple[str, tuple[str, ...]]:
    """Process EVERY anatomy whose Ds539 argmax mask is a substantial fraction of the largest
    present mask. The genuinely-present anatomy is always kept (no misroute -> 0); a small
    hallucination is dropped by the fraction gate, a sizable one becomes a minor FP, never a zero.

    [v1.3.3 note] An absolute-volume + confidence variant was prototyped and REJECTED on the
    evidence: Ds539's softmax confidence does NOT separate a real bone from a confidently-painted
    cross-group hallucination (both 0.97-0.99), and Ds539 anatomy masks for present bones are all
    large (>100 cm^3) whether real or phantom — so no inference-side gate (relative or absolute)
    can drop the LARGE cross-group phantom without dropping real bone. Pure absolute presence
    additionally regressed femur-only cases (it kept small phantom pelvic bones the relative gate
    drops). The cross-group false positive is a MODEL-level issue (Ds539's partial-label marginal
    training) whose fundamental fix is retraining, not a routing threshold. Keep the validated
    relative gate.
    """
    sizes = {a: int(ds539_masks[a].sum()) for a in ALL_ANATOMIES
             if a in ds539_masks and ds539_masks[a] is not None}
    biggest = max(sizes.values(), default=0)
    if biggest <= 0:
        log("route_from_ds539(multi): no Ds539 mask -> pelvic fallback")
        return "pelvic", PELVIC_ANATOMIES
    keep_frac = float(os.environ.get("PENGWIN_ROUTE_KEEP_FRAC", "0.20"))
    kept = tuple(a for a in ALL_ANATOMIES if sizes.get(a, 0) >= keep_frac * biggest)
    log(f"route_from_ds539(multi): sizes={sizes} keep>={keep_frac:.2f}x{biggest} -> {kept}")
    return "multi", kept


def classify_pelvic_femur_rule(spacing_x, spacing_y, spacing_z, physical_x_mm, physical_z_mm):
    """The organizers' official pelvic/femur decision tree, transcribed verbatim from the guidelines.

    Organizers: *"participants can separate pelvic/femur with a rule derived from image spacing and
    physical FOV. **The test sets have been verified to conform to this rule.**"*

    ⚠️ MEASURED ON OUR 340 TRAINING CASES (2026-07-21): this rule agrees with the GT family only
    **172/340 = 50.6%**, and on the 168 cases where it disagrees with the deployed RandomForest the
    rule is right **0/168** while the RF is right **168/168**. Direction: 154 rule=pelvic/GT=femur,
    14 rule=femur/GT=pelvic. It is NOT an orientation bug (all prelim CTs are already LPS).
    Conclusion: the thresholds are tuned to the organizers' curated acquisition protocol, which our
    7-institution training data does not match. The rule is therefore used ONLY as a weak prior in
    the abstention branch below, and NEVER to override a confident RF.
    """
    if physical_x_mm <= 285.35:
        if spacing_x <= 0.71:
            return "pelvic"
        elif spacing_z <= 0.90:
            return "femur"
        else:
            return "pelvic" if spacing_y <= 0.91 else "femur"
    else:
        if spacing_z <= 0.68:
            return "pelvic" if physical_z_mm <= 193.55 else "femur"
        else:
            return "pelvic" if physical_z_mm <= 390.78 else "femur"


def _family_evidence_from_ds539(ds539_masks: dict | None) -> tuple[str | None, float]:
    """Direct image evidence: femur voxels vs pelvic voxels in the Stage-1 anatomy argmax.

    Free — Stage-1 already ran before routing. Returns (family|None, femur_fraction).
    Deliberately NOT used on its own: Ds539 confidently hallucinates cross-family bone (a femur-only
    scan in the GC logs got LeftHip=51,226 voxels across 22 CCs), which is exactly why pure
    Ds539-volume routing scored GC instance F1 0.572 vs 0.940 for the RF.
    """
    if not ds539_masks:
        return None, float("nan")
    fem = int(ds539_masks["Femur"].sum()) if ds539_masks.get("Femur") is not None else 0
    pel = sum(int(ds539_masks[a].sum()) for a in PELVIC_ANATOMIES
              if ds539_masks.get(a) is not None)
    tot = fem + pel
    if tot <= 0:
        return None, float("nan")
    frac = fem / tot
    return ("femur" if frac >= 0.5 else "pelvic"), frac


def route_from_target_family_router(image_path: Path,
                                    ds539_masks: dict | None = None) -> tuple[str, tuple[str, ...]] | None:
    if not TARGET_ROUTER_ENABLED:
        return None

    global _TARGET_ROUTER_PAYLOAD
    if _TARGET_ROUTER_PAYLOAD is None:
        from target_family_router import load_router

        _TARGET_ROUTER_PAYLOAD = load_router(TARGET_ROUTER_PATH)
        names = _TARGET_ROUTER_PAYLOAD.get("feature_names", [])
        labels = _TARGET_ROUTER_PAYLOAD.get("labels", [])
        log(f"target-router: loaded {TARGET_ROUTER_PATH} n_features={len(names)} labels={labels}")

    from target_family_router import predict_family

    family, p_femur = predict_family(image_path, _TARGET_ROUTER_PAYLOAD)
    source = "target-router"

    # --- [v2.4] OOD abstention gate ------------------------------------------------------------
    # Measured on all 340 training cases (2026-07-21): the RF is correct 340/340 and its decision
    # margin |2p-0.5| never drops below 0.9052 -- there are ZERO cases below 0.90. So with the
    # default threshold 0.85 this branch is a provable NO-OP on every case we can measure, and the
    # deployed rank-10 routing is bit-identical.
    #
    # It exists only for a genuinely out-of-distribution test scan where the RF is less certain than
    # anything it ever saw in training. There we do NOT trust the RF blindly; we combine the two
    # independent signals we have -- the organizers' rule (guaranteed on the TEST set, useless on
    # ours) and the Stage-1 anatomy evidence (real image content, but hallucination-prone).
    #
    # Deliberately NOT the symmetric "rule AND RF must agree, else tiebreak" design: the rule
    # disagrees with the RF on 49.4% of our cases and is wrong on 100% of those disagreements, so
    # letting it trigger a tiebreak would hand half of all routing decisions to the 0.572-F1 signal.
    # Set PENGWIN_ROUTER_ABSTAIN_MARGIN=0 to hard-disable, or raise it to widen the safety net.
    margin = abs(float(p_femur) - 0.5) * 2.0 if np.isfinite(p_femur) else 1.0
    if margin < ROUTER_ABSTAIN_MARGIN:
        try:
            _r = sitk.ImageFileReader()
            _r.SetFileName(str(image_path))
            _r.ReadImageInformation()
            sx, sy, sz = _r.GetSpacing()
            dx, dy, dz = _r.GetSize()
            rule_family = classify_pelvic_femur_rule(sx, sy, sz, sx * dx, sz * dz)
        except Exception as exc:  # noqa: BLE001
            rule_family = None
            log(f"target-router: official rule unavailable ({exc})")
        anat_family, femur_frac = _family_evidence_from_ds539(ds539_masks)
        votes = [v for v in (family, rule_family, anat_family) if v]
        winner = max(set(votes), key=votes.count) if votes else family
        log(f"target-router: LOW CONFIDENCE margin={margin:.4f} < {ROUTER_ABSTAIN_MARGIN} "
            f"-> vote rf={family} rule={rule_family} anatomy={anat_family} "
            f"(femur_frac={femur_frac:.3f}) => {winner}")
        if winner != family:
            source = "target-router-abstain"
            family = winner

    if family == "femur":
        anatomies = ("Femur",)
    elif family == "pelvic":
        anatomies = PELVIC_ANATOMIES
    else:
        raise ValueError(f"target-router predicted unsupported family {family!r}")

    if np.isfinite(p_femur):
        log(f"target-router: family={family} p_femur={p_femur:.4f} margin={margin:.4f} "
            f"-> anatomies={anatomies}")
    else:
        log(f"target-router: family={family} p_femur=nan -> anatomies={anatomies}")
    return f"{source}:{family}", anatomies


# ---------------------------------------------------------------------------
# anatomy 별 Ds538 추론 파이프라인 (메인 로직).
# ---------------------------------------------------------------------------
def run_per_anatomy(image_path: Path, ref_img: sitk.Image,
                    prerouted_bone_masks: dict | None = None,
                    anatomies: tuple[str, ...] | None = None) -> np.ndarray:
    """V0.4 STU-Net 2-stage 견고화 파이프라인 (4-layer 구조):
       Layer 1: bone-skeleton anatomy 분해 (HU>200, pelvic 전용) — fallback 확보
       Layer 2: Ds539 5-class argmax refinement       — Ds539 가 합리적이면 우선 사용
       Layer 2b: v2.0 target-family router, or Ds539 argmax volume route fallback
       Layer 3: bbox sanity + 시간 예산 점검          — 안전망 역할
       Layer 4: Ds538 per-anatomy ABBC 추론 + decode + paste

    Femur (151-200) 활성화: Ds539 의 channel 4 argmax mask 로 femur ROI 를 잡는다.
    PENGWIN 케이스는 pelvic 이거나 femur (disjoint). v2.0에서는 CT/FOV/bone-geometry
    router가 target family를 정하고, router가 꺼져 있으면 Ds539 argmax 부피 비율로
    결정한다(spacing 룰은 femur 를 거의 전부 pelvic 으로 오라우팅하므로 사용 안 함).

    anatomies 가 None 이면 target-family router 또는 Ds539 argmax 로 자동 라우팅한다.
    명시되면(테스트/디버그용) 그 subset 만 강제한다.
    """
    import time
    t_start = time.time()

    # === Step 1: LPS 정규화 + bone window HU clip ===
    img_lps, arr_clipped = canonicalize_and_clip_image(ref_img)
    image_npy_lps = arr_clipped.astype(np.float32, copy=False)
    ct_lut_full = bone_lut_normalize(arr_clipped)
    # DIAG: raw (pre-clip) HU stats — detects an OOD / mis-calibrated GC input (e.g. uint16
    # raw pixels needing rescale, or a non-HU intensity range) that would make Ds539 garbage.
    try:
        _raw = sitk.GetArrayFromImage(img_lps)
        log(f"DIAG raw-HU: dtype={_raw.dtype} min={float(_raw.min()):.0f} max={float(_raw.max()):.0f} "
            f"mean={float(_raw.mean()):.0f} p1={float(np.percentile(_raw,1)):.0f} "
            f"p99={float(np.percentile(_raw,99)):.0f} bone(>200)={float((_raw>200).mean()):.4f} "
            f"air(<-500)={float((_raw<-500).mean()):.4f}")
    except Exception as _e:
        log(f"DIAG raw-HU: failed ({_e})")
    spacing_xyz = img_lps.GetSpacing()
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    img_shape = image_npy_lps.shape
    forced_anatomies = anatomies  # None 이면 Ds539 argmax 로 자동 라우팅
    log(f"V0.4 preproc: shape={img_shape} spacing_zyx={spacing_zyx} "
        f"forced_anatomies={forced_anatomies}")

    # === Layer 1: bone HU mask 기반 anatomy 분해 (항상 실행, fallback 용) ===
    if prerouted_bone_masks is not None:
        bone_masks = prerouted_bone_masks
        log(f"L1 bone-skeleton: reusing prerouted masks {list(bone_masks.keys())}")
    else:
        bone_masks = bone_skeleton_anatomy_decomposition(
            arr_clipped, spacing_zyx,
            hu_threshold=BONE_HU_THRESHOLD,
            min_component_voxels=BONE_MIN_COMPONENT_VOXELS,
        )
        log(f"L1 bone-skeleton: {list(bone_masks.keys())}")

    # === Step 2: Ds539 anatomy classifier 추론 (전체 CT, 1-channel HU 입력, 5-class) ===
    ds539_predictor = build_predictor(
        DS539_DATASET, DS539_TRAINER, DS539_PLANS, DS539_CONFIG, DS539_FOLD,
    )
    ds539_image_4d = image_npy_lps[None]
    ds539_props = {"spacing": list(spacing_zyx)}
    log(f"L2 Ds539 predict: image (C,Z,Y,X)={ds539_image_4d.shape}")
    ds539_logits_pp, ds539_data_props = predict_logits_full(
        ds539_predictor, ds539_image_4d, ds539_props, output_channels=DS539_OUTPUT_CHANNELS,
    )
    ds539_probs_pp = softmax_axis0(ds539_logits_pp)
    # nnUNet preprocessed grid 의 prob 를 원본 CT 격자로 다시 resampling (linear interp)
    from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
    from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
    shape_after_cropping = tuple(int(s) for s in ds539_data_props["shape_after_cropping_and_before_resampling"])
    current_spacing = ds539_predictor.configuration_manager.spacing
    transpose_forward = ds539_predictor.plans_manager.transpose_forward
    transpose_backward = ds539_predictor.plans_manager.transpose_backward
    original_spacing = [ds539_data_props["spacing"][i] for i in transpose_forward]
    probs_resampled_axes = resample_data_or_seg_to_shape(
        ds539_probs_pp.astype(np.float32, copy=False),
        shape_after_cropping, current_spacing, original_spacing,
        is_seg=False, order=1, order_z=0,
    )
    shape_before_cropping = tuple(int(s) for s in ds539_data_props["shape_before_cropping"])
    bbox_used = ds539_data_props.get("bbox_used_for_cropping", None)
    probs_full_axes = np.zeros((DS539_OUTPUT_CHANNELS, *shape_before_cropping), dtype=np.float32)
    if bbox_used is None:
        probs_full_axes = np.asarray(probs_resampled_axes, dtype=np.float32)
    else:
        slicer = bounding_box_to_slice(bbox_used)
        probs_full_axes[(slice(None), *slicer)] = np.asarray(probs_resampled_axes, dtype=np.float32)
    # cropping bbox 바깥 영역은 fg 가 비어 있으므로 background prob 를 1 로 채워 정상화
    fg_sum = probs_full_axes[1:].sum(axis=0)
    bg_mask = fg_sum < 1e-6
    if bg_mask.any():
        probs_full_axes[0][bg_mask] = 1.0
    probs_full = np.transpose(probs_full_axes, (0, *(int(a) + 1 for a in transpose_backward)))
    assert probs_full.shape[1:] == img_shape, f"shape mismatch: {probs_full.shape[1:]} vs {img_shape}"

    # === Layer 2: Ds539 5-class argmax 기반 anatomy mask 추출 (Femur 포함) ===
    ds539_masks = ds539_argmax_masks(probs_full)
    log("L2 Ds539 argmax: " + ", ".join(f"{a}={int(ds539_masks[a].sum())}" for a in ALL_ANATOMIES))
    # DIAG: Ds539 health — each anatomy's fraction of the WHOLE volume. The GC failure was
    # RightHip=73% + Sacrum 18141 CCs; a healthy case is each anatomy ~1-2% of volume.
    try:
        _tot = float(np.prod(img_shape))
        _fr = {a: int(ds539_masks[a].sum()) / max(1.0, _tot) for a in ALL_ANATOMIES}
        _maxa = max(_fr, key=_fr.get)
        log("DIAG Ds539 health: frac={" + ", ".join(f"{a}:{_fr[a]:.3f}" for a in ALL_ANATOMIES) + "} "
            f"largest={_maxa}({_fr[_maxa]:.3f}) HEALTH="
            + ("GARBAGE(>50%_one_class!)" if _fr[_maxa] > 0.50 else "ok"))
    except Exception as _e:
        log(f"DIAG Ds539 health: failed ({_e})")

    # === Layer 2b: pelvic vs femur 라우팅 (Ds539 argmax 부피 비율) ===
    used_target_router = False
    if forced_anatomies is not None:
        anatomies = tuple(forced_anatomies)
        log(f"L2b routing: forced anatomies={anatomies}")
    else:
        router_route = route_from_target_family_router(image_path, ds539_masks)
        if router_route is not None:
            _route, anatomies = router_route
            used_target_router = True
        else:
            _route, anatomies = route_from_ds539_masks(ds539_masks)
        log(f"L2b routing: auto route={_route} anatomies={anatomies}")

        # [v1.7 Stage-A recall fix] Reconcile the Ds539-size routing with the GEOMETRIC bone-skeleton
        # decomposition (robust for pelvic PRESENCE + laterality). The GC recall killer (logs: a
        # present hip given Ds539=0 or swapped L<->R -> the 0.20x routing gate drops it -> whole-
        # anatomy ZERO = all its fragments become FN) is fixed in two conservative steps. Default ON;
        # PENGWIN_STAGEA_BONE_RECONCILE=0 to disable. Femur (no bone-skeleton) is untouched.
        if (not used_target_router) and os.environ.get("PENGWIN_STAGEA_BONE_RECONCILE", "1") == "1":
            routed = list(anatomies)
            # A femur-DOMINANT scan has incidental in-frame pelvis that is NOT a GT target (e.g. dev
            # 294: Ds539 Femur 2.28M >> LeftHip 740k, GT = femur-only). Adding the bone-skeleton pelvis
            # there is the femur-case FP regression the v1.3.3 note warns of. So (A) recall-recovery
            # fires ONLY when a PELVIC anatomy is the LARGEST Ds539 mask (= a pelvic case = where the GC
            # whole-hip miss/swap actually happens). Femur-dominant cases are left to the relative gate.
            _sz = {a: int(ds539_masks[a].sum()) for a in ALL_ANATOMIES if ds539_masks.get(a) is not None}
            _pelvic_dominant = bool(_sz) and max(_sz, key=_sz.get) in PELVIC_ANATOMIES
            # (A) recall: on a PELVIC-dominant case, add any pelvic anatomy bone-skeleton found that the
            #     Ds539 size gate dropped (recovers the GC whole-anatomy MISS)
            if _pelvic_dominant:
                for a in PELVIC_ANATOMIES:
                    bm = bone_masks.get(a)
                    if bm is not None and bm.any() and a not in routed:
                        routed.append(a)
                        log(f"L2b +bone-recall: {a} present in bone-skeleton but Ds539 routing dropped it")
            # (B) laterality: a routed Ds539 hip that bone-skeleton did NOT find, whose mask mostly
            #     overlaps the OTHER hip's bone mask, is an L<->R swap -> drop the hallucinated side
            for a, other in (("LeftHip", "RightHip"), ("RightHip", "LeftHip")):
                if a not in routed:
                    continue
                bma = bone_masks.get(a)
                if bma is not None and bma.any():
                    continue  # bone-skeleton confirms this side is real -> keep
                bmo, dma = bone_masks.get(other), ds539_masks.get(a)
                if bmo is not None and bmo.any() and dma is not None and dma.any():
                    inter = int(np.logical_and(dma, bmo).sum())
                    if inter > 0.5 * float(dma.sum()):
                        routed.remove(a)
                        log(f"L2b -laterality: drop {a} (Ds539 mask {inter/max(1,int(dma.sum()))*100:.0f}% "
                            f"inside {other}'s bone mask = left/right swap)")
            anatomies = tuple(x for x in ALL_ANATOMIES if x in routed)
            log(f"L2b bone-reconciled anatomies={anatomies}")

    # === Layer 2+3: Ds539 mask 와 bone fallback 을 sanity check 와 함께 병합 ===
    merged = merge_masks_with_sanity(
        ds539_masks, bone_masks, img_shape,
        sanity_max_fraction=SANITY_MAX_BBOX_FRACTION,
        bone_fallback_when_sanity_fails=V03_USE_BONE_SKELETON_FALLBACK,
        anatomies=anatomies,
    )

    # Ds538 모델 로딩 전에 Ds539 관련 텐서를 해제해 GPU 메모리 확보
    del ds539_predictor, ds539_logits_pp, ds539_probs_pp, probs_resampled_axes, probs_full_axes
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # === Layer 4: anatomy 별 Ds538 추론 → ABBC decode → 전체 라벨에 paste ===
    ds538_predictor = build_predictor(
        DS538_DATASET, DS538_TRAINER, DS538_PLANS, DS538_CONFIG, DS538_FOLD,
    )

    full_label = np.zeros(img_shape, dtype=np.uint16)
    for anatomy in anatomies:
        elapsed = time.time() - t_start
        if V03_USE_TIME_BUDGET and elapsed >= TIME_BUDGET_HARD_LIMIT:
            log(f"[{anatomy}] HARD time budget hit at {elapsed:.0f}s, emit zero for this and remaining anatomies")
            break

        info = merged[anatomy]
        if info['mask'] is None:
            log(f"[{anatomy}] no usable mask, emit zero")
            continue

        mask = info['mask']
        # Routing CC policy (ablatable via PENGWIN_ROUTE_CC_MODE; default 'largest' = deployed
        # v1.3.x behaviour). The largest-CC-only policy is a suspected recall lever: a DISPLACED
        # fracture fragment forms its own CC and is dropped before Stage-B, so its instance is lost.
        #   largest : keep only the biggest CC (current default — drops displaced fragments).
        #   union   : keep ALL CCs -> the ROI bbox spans every fragment of this anatomy (recall fix).
        #   floor   : keep CCs >= PENGWIN_ROUTE_CC_MIN_VOX (drop only noise CCs, keep real ones).
        _cc_mode = os.environ.get("PENGWIN_ROUTE_CC_MODE", "largest").strip().lower()
        if _cc_mode != "union":
            import scipy.ndimage as ndi
            cc_labels, n_cc = ndi.label(mask, structure=np.ones((3,3,3), dtype=bool))
            if n_cc > 1:
                sizes = ndi.sum(mask, cc_labels, index=np.arange(1, n_cc+1))
                if _cc_mode == "floor":
                    min_vox = int(os.environ.get("PENGWIN_ROUTE_CC_MIN_VOX", str(MIN_COMPONENT_VOXELS)))
                    keep = [i + 1 for i, s in enumerate(sizes) if s >= min_vox] or [int(np.argmax(sizes)) + 1]
                    mask = np.isin(cc_labels, keep)
                else:  # 'largest' (default)
                    keep_idx = int(np.argmax(sizes)) + 1
                    mask = (cc_labels == keep_idx)
        # 'union' leaves the multi-CC mask untouched so bbox_from_mask spans every fragment.

        bbox = bbox_from_mask(mask, pad_vox=ROI_PAD_VOX)
        if bbox is None:
            log(f"[{anatomy}] mask empty after CC filter, emit zero")
            continue

        # [v1.3.3 ROB-3] The bbox>50% "sanity" band-aid was removed. It fired on CLEAN large
        # anatomy in a tight FOV (e.g. a hip legitimately covering 53.6% of the volume), shrank
        # the pad and ultimately swapped the soft Ds539 probability ROI for a hard 0/1 bone-HU
        # mask — feeding Ds538 an out-of-distribution channel and degrading the decode. It only
        # existed to bound the speckle-era garbage masks, which the v1.3.2 weight-load fix
        # eliminated. The bbox keeps pad=24 with the soft probability channel; the time-budget
        # gate below is the sole (and sufficient) guard against an oversized/slow crop.

        # 시간 예산 기반 sanity check (Grand Challenge 10분 제한 보호)
        eta = estimate_fracture_inference_seconds(bbox)
        bbox_fraction = float(np.prod([bbox[i].stop - bbox[i].start for i in range(3)])) / float(np.prod(img_shape))
        log(f"[{anatomy}] bbox crop={tuple(bbox[i].stop - bbox[i].start for i in range(3))} "
            f"fraction={bbox_fraction*100:.1f}% source={info['source']} ETA={eta:.0f}s elapsed={elapsed:.0f}s")

        if V03_USE_TIME_BUDGET and (elapsed + eta) > TIME_BUDGET_SECONDS:
            log(f"[{anatomy}] elapsed+ETA={elapsed+eta:.0f}s > budget {TIME_BUDGET_SECONDS}s, emit zero")
            continue

        # [v1.4 LEAK-FREE] Ds538 입력: 1-channel CT-ONLY (bone-LUT CT crop).
        # V302 trains on CT only — the old 3-channel [CT, Ds539-prob, SDF] tensor leaked the
        # Stage-A anatomy probability into Stage-B. The bbox above still uses the Ds539 prob for
        # localization (cascade routing), but the model input is now pure CT.
        ct_lut_crop = ct_lut_full[bbox]
        ds538_image_4d = ct_lut_crop[None]  # (1, Z, Y, X)
        ds538_props = {"spacing": list(spacing_zyx)}

        # Ds538 sliding-window 추론 실행
        ds538_logits_pp, ds538_data_props = predict_logits_full(
            ds538_predictor, ds538_image_4d, ds538_props, output_channels=DS538_OUTPUT_CHANNELS,
        )
        # [TIER-1 experiment, env-gated] V307 affinity head: Ds538 output is 4 ABBC + K affinity
        # channels. Split -> softmax(ABBC[:4]) + sigmoid(affinity[4:]) -> average-linkage
        # agglomeration on the LEARNED affinities. Default OFF -> V302 (4ch) path unchanged.
        _fusion = os.environ.get("PENGWIN_FUSION_DECODE", "0") == "1"
        if os.environ.get("PENGWIN_AFFINITY_DECODE", "0") == "1" or _fusion:
            import sys as _sys
            # agglo_decode.py is VENDORED next to this file so it ships in the GC container; also add
            # the dev experiments/ dir as a fallback for local runs. (numpy/scipy/skimage only.)
            for _p in (os.path.dirname(os.path.abspath(__file__)),
                       os.path.join(os.environ.get("PENGWIN_ROOT", "/nonexistent"),
                                    "code_task1", "experiments")):
                if _p not in _sys.path:
                    _sys.path.insert(0, _p)
            from agglo_decode import decode_affinity_agglo, decode_fusion
            _abbc = softmax_axis0(ds538_logits_pp[:4])
            _aff = 1.0 / (1.0 + np.exp(-np.asarray(ds538_logits_pp[4:], dtype=np.float32)))
            # [13ch raw dump] 4 ABBC softmax + 9 affinity sigmoid -> enables OFFLINE decode/fusion
            # T-sweeps on CPU (no GPU re-run). Split offline as probs13[:4]/probs13[4:].
            _dump_dir = os.environ.get("PENGWIN_DUMP_PROBS", "")
            if _dump_dir:
                os.makedirs(_dump_dir, exist_ok=True)
                np.save(os.path.join(_dump_dir, f"probs13_{anatomy}.npy"),
                        np.concatenate([_abbc, _aff], axis=0).astype(np.float16))
                log(f"[dump] saved 13ch probs {anatomy} -> {_dump_dir}")
            if os.environ.get("PENGWIN_AFF_STATS", "0") == "1":
                _sh = _aff[:3]  # short-range channels
                log(f"[aff:{anatomy}] all: mean={float(_aff.mean()):.3f} frac<0.5={float((_aff<0.5).mean()):.4f} | short: mean={float(_sh.mean()):.3f} frac<0.5={float((_sh<0.5).mean()):.4f} min={float(_sh.min()):.3f}")
            if _fusion:
                # [FUSION] Use the EXACT deployed V302 core-seed watershed as the precise base (incl.
                # the Sacrum-specific aggressive merge), then sub-split ONLY within a base instance
                # where the learned affinity confirms a true fracture surface -> recall up, base
                # boundaries never crossed = precision floor = V302.
                if anatomy == "Sacrum":
                    _base = decode_abbc_core_seed_watershed(
                        _abbc, size_ratio_keep=max(float(ANATOMY_SIZE_RATIO_KEEP), 0.10),
                        min_component_voxels=max(int(MIN_COMPONENT_VOXELS), 250))
                else:
                    _base = decode_abbc_core_seed_watershed(_abbc)
                decoded_pp = decode_fusion(_base, _abbc, _aff,
                    T=float(os.environ.get("PENGWIN_FUSION_T", os.environ.get("PENGWIN_AGGLO_T", "0.45"))),
                    min_ridge_vox=int(os.environ.get("PENGWIN_FUSION_RIDGE_VOX", "3000")))
            else:
                decoded_pp = decode_affinity_agglo(_abbc, _aff, T=float(os.environ.get("PENGWIN_AGGLO_T", "0.45")))
        else:
            ds538_probs = softmax_axis0(ds538_logits_pp)
            _dump_dir = os.environ.get("PENGWIN_DUMP_PROBS", "")
            if _dump_dir:
                os.makedirs(_dump_dir, exist_ok=True)
                np.save(os.path.join(_dump_dir, f"probs_{anatomy}.npy"), ds538_probs.astype(np.float16))
                log(f"[dump] saved ds538_probs {anatomy} -> {_dump_dir}")
            # [2026-06-06] Sacrum over-segments — its predicted core speckles into spurious islands
            # inside one dominant bone -> too many fragments. Apply an anatomy-specific aggressive
            # merge (size-ratio 0.10 + min 250 voxels) for Sacrum only; femur/hips are genuine
            # multi-fragment bones and keep the defaults (a global merge collapsed their recall).
            if anatomy == "Sacrum":
                decoded_pp = decode_abbc_core_seed_watershed(
                    ds538_probs,
                    size_ratio_keep=max(float(ANATOMY_SIZE_RATIO_KEEP), 0.10),
                    min_component_voxels=max(int(MIN_COMPONENT_VOXELS), 250),
                )
            else:
                decoded_pp = decode_abbc_core_seed_watershed(ds538_probs)
        decoded_crop = resample_label_map_to_original(decoded_pp, ds538_data_props, ds538_predictor)

        # 라벨 재매핑 후 전체 볼륨에 paste
        # PENGWIN 스펙: Sacrum [1..50], LeftHip [51..100], RightHip [101..150], Femur [151..200]
        lo, hi = ANATOMY_RANGES[anatomy]
        n_slots = hi - lo + 1
        local_ids = sorted(int(v) for v in np.unique(decoded_crop) if int(v) > 0)
        if not local_ids:
            log(f"[{anatomy}] no fragments after decode")
            continue
        # 슬롯이 부족하면 fragment 크기 순으로 상위 n_slots 개만 살린다
        if len(local_ids) > n_slots:
            sizes_local = [(lid, int((decoded_crop == lid).sum())) for lid in local_ids]
            sizes_local.sort(key=lambda x: -x[1])
            local_ids = [t[0] for t in sizes_local[:n_slots]]
        remap = np.zeros(max(local_ids) + 1, dtype=np.uint16)
        for i, old_id in enumerate(local_ids):
            remap[old_id] = np.uint16(lo + i)
        # 슬롯 초과 ID 는 0 으로 마스킹 (PENGWIN 라벨 범위 보호)
        decoded_crop_filtered = np.where(np.isin(decoded_crop, local_ids), decoded_crop, 0).astype(np.int32, copy=False)
        # remap 인덱스를 안전하게 사용하기 위해 valid 범위로 clip
        decoded_crop_filtered = np.clip(decoded_crop_filtered, 0, max(local_ids))
        remapped_crop = remap[decoded_crop_filtered]

        out_slot = full_label[bbox]
        write_mask = (remapped_crop > 0) & (out_slot == 0)
        # [v1.4.1 FIX] Do not let an anatomy paint into ANOTHER anatomy's Ds539-argmax territory. The
        # padded ROI bboxes of two anatomies overlap in space (notably a femur-only scan where Ds539
        # paints a phantom pelvic bone whose ROI overlaps the real femur). The phantom's watershed
        # regrows over the shared bone support and, with first-anatomy-wins (out_slot==0), BLOCKS the
        # later femur -> femur destroyed (210k -> 7.9k voxels, Dice 0.95 -> 0.0). A voxel-into-its-own
        # mask clip over-clips legitimate regrow (pelvic Dice 0.90 -> 0.76); instead we only forbid
        # painting into OTHER bones' argmax masks. An anatomy may still regrow freely into background;
        # it just cannot invade a neighbouring bone. Ds539 argmax is 1-anatomy-per-voxel (disjoint).
        if os.environ.get("PENGWIN_CONFINE_TO_MASK", "1") == "1":
            other = np.zeros(img_shape, dtype=bool)
            for _a, _inf in merged.items():
                if _a != anatomy and _inf.get('mask') is not None:
                    other |= np.asarray(_inf['mask']).astype(bool)
            ob = other[bbox]
            if ob.shape == write_mask.shape:
                write_mask = write_mask & ~ob
        out_slot[write_mask] = remapped_crop[write_mask]
        full_label[bbox] = out_slot
        log(f"[{anatomy}] painted {int(write_mask.sum())} voxels, {len(local_ids)} fragments in range [{lo},{hi}]")

    # === Step 5: 원본이 LPS 가 아니었다면 원래 방향으로 되돌린다 ===
    # nnUNet 은 LPS 로 추론하므로, 원본 orientation 으로 복구해야
    # Grand Challenge 가 기대하는 좌표계와 일치한다.
    if orientation_code(ref_img) != "LPS":
        lps_label_img = sitk.GetImageFromArray(full_label.astype(np.uint16, copy=False))
        lps_label_img.SetSpacing(img_lps.GetSpacing())
        lps_label_img.SetOrigin(img_lps.GetOrigin())
        lps_label_img.SetDirection(img_lps.GetDirection())
        target_code = orientation_code(ref_img)
        oriented = sitk.DICOMOrient(lps_label_img, target_code)
        full_label = sitk.GetArrayFromImage(oriented).astype(np.uint16, copy=False)
        log(f"reoriented label map: LPS -> {target_code} (shape={full_label.shape})")

    log(f"V0.4 pipeline complete in {time.time() - t_start:.1f}s, unique={sorted(int(v) for v in np.unique(full_label)[:20])}")
    return full_label


# ---------------------------------------------------------------------------
# main() — 컨테이너 진입점
# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    log(f"start: model_root={MODEL_ROOT} nn_results={NN_RES}")
    try:
        image_path = find_input_image()
        log(f"input image: {image_path}")
        ref_img = sitk.ReadImage(str(image_path))

        size = ref_img.GetSize()       # SimpleITK 순서 (X, Y, Z)
        spacing = ref_img.GetSpacing() # SimpleITK 순서 (sx, sy, sz)
        spacing_x, spacing_y, spacing_z = (float(s) for s in spacing)
        physical_x_mm = float(size[0]) * spacing_x
        physical_z_mm = float(size[2]) * spacing_z
        log(
            "spatial: "
            f"size={size} spacing=({spacing_x:.4f},{spacing_y:.4f},{spacing_z:.4f}) "
            f"physical_x_mm={physical_x_mm:.2f} physical_z_mm={physical_z_mm:.2f}"
        )
        # V0.4 라우팅: pelvic vs femur 결정은 Stage-1 Ds539 추론 뒤로 미룬다.
        # 과거 spacing 룰(classify_pelvic_femur)은 femur 케이스를 거의 전부 pelvic 으로
        # 오라우팅한다(검증 ~50%) — femur zero-stub 시절엔 무해했지만 femur 활성화 후엔
        # 치명적이다. 대신 run_per_anatomy 가 Ds539 argmax 부피 비율로 라우팅한다(~90%).
        #
        # bone-skeleton 분해는 pelvic ROI fallback 으로만 쓰이므로 여기서 미리 계산해
        # 넘긴다(femur 케이스에선 자연히 비어 무시된다). spacing 기반 분류는 사용하지 않는다.
        try:
            _img_lps_for_bone, _arr_clipped_for_bone = canonicalize_and_clip_image(ref_img)
            _sp_xyz = _img_lps_for_bone.GetSpacing()
            _sp_zyx = (float(_sp_xyz[2]), float(_sp_xyz[1]), float(_sp_xyz[0]))
            prerouted_bone_masks = bone_skeleton_anatomy_decomposition(
                _arr_clipped_for_bone, _sp_zyx,
                hu_threshold=BONE_HU_THRESHOLD,
                min_component_voxels=BONE_MIN_COMPONENT_VOXELS,
            )
        except Exception as bone_exc:  # noqa: BLE001
            log(f"bone-skeleton preroute failed ({bone_exc}); proceeding without fallback masks")
            prerouted_bone_masks = None

        out_path = output_path(image_path)
        # anatomies=None -> v2.0 target-family router when enabled, otherwise Ds539 volume route.
        label_arr = run_per_anatomy(
            image_path, ref_img, prerouted_bone_masks, anatomies=None,
        )
        write_label_map(label_arr, ref_img, out_path)
        log(f"wrote prediction -> {out_path}")
    except Exception as exc:
        log(f"FATAL: {exc}")
        traceback.print_exc()
        try:
            image_path = find_input_image()
            ref_img = sitk.ReadImage(str(image_path))
            write_zero_output(image_path, ref_img, f"fallback after exception: {exc}")
        except Exception as fallback_exc:
            log(f"fallback also failed: {fallback_exc}")
            traceback.print_exc()
            return 1
    finally:
        dt = time.time() - t0
        log(f"done in {dt:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

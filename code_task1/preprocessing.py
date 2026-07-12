#!/usr/bin/env python3
"""PENGWIN 2026 Task 1 — Dataset preprocessing & sidecar builders.

Active datasets: Ds539 (PelvicFemurAnatomyV3, 5-class anatomy) + Ds538
(PelvicFemurBICMFragmentV5, per-anatomy ABBC fracture). 532/533/537 retired.

Dataset builder functions (driven by preprocessing/gen_nnunet_dataset.py stages):
    generate_anatomy_probability_context  — Stage-A anatomy-prob cache (the only inference step)
    build_anatomy_semantic_dataset(539)   — whole-CT 5-class anatomy target
    build_bicm_v5_dataset(538)            — per-anatomy 3ch ABBC fragment ROIs

CLI (sidecars on the preprocessed grid):
    python preprocessing.py build-instance-sidecars --dataset 538
    python preprocessing.py build-boundary-fragment-v3-sidecars --dataset 538
"""
from __future__ import annotations
import argparse, json, os, tempfile, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from core import (
    DATA_RAW, NN_RAW, NN_PREP, NN_RES, DATASETS, ANATOMY_RANGES, ANATOMY_NAMES,
    is_pelvic, is_femur, case_subject_type, RESULT_DATE, RESULT_REPORT,
    RESULT_VISUALIZE, configure_nnunet_env, get_logger, ABBC_HARD_NEGATIVE_LABEL,
    CONTACT_INSTANCE_CHANNEL_NAMES, FACTOR_INSTANCE_CHANNEL_NAMES,
    FACTOR_INSTANCE_SIDECAR_DIR, _instance_edge_break_target,
)
from utils import (
    find_case_dir, list_cases, inst_to_anat, crop_save_mha, save_crop_array_mha,
    save_full_mha, canonicalize_sitk,
    prepare_lps_ct_for_nnunet, orientation_code, clip_ct_hu,
    ABBC_DISTANCE_THRESHOLD_VOX, ABBC_DIVERGENCE_THRESHOLD, ABBC_FRACTURE_DISK_RADIUS,
    ABBC_OFFICIAL_LABELS, ABBC_BORDER_LABEL, ABBC_CORE_LABEL,
    audit_official_target, compute_abbc_official_target, get_contact_surface_regions,
    BFV3_LABELS, BFV3_CLASS_NAMES, BoundaryFragmentParams, compute_boundary_fragment_target,
    BICMV5Params, V5_ANATOMY_RANGES, V5_ANATOMY_RANGES_WITH_FEMUR,
    V5_LABELS, V5_TARGET_PROFILES,
    anatomy_range, anatomy_mask_from_instances, bbox_from_mask, compute_bicm_v5_target,
    label_distribution,
)
# Registry single source. The per-anatomy BICM V5 sidecar builder handles Femur
# ROIs (151-200) so it uses the FULL view; legacy pelvic-only V3/V4 builders keep
# the explicit pelvic_only=True view (their "drop femur" intent stays visible).
from utils import (
    MAX_INSTANCE_ID,
    PELVIC_MAX_INSTANCE_ID,
    valid_instance_mask,
)
configure_nnunet_env()
log = get_logger(__name__)


V5_INPUT_VARIANTS = ("ct_lut", "ct_lut_anat_sdf")
# [V0.x][FIX:B1+B2][2026-05-31] Dataset539 5-class anatomy 의 softmax 채널 인덱스.
# Ds532 4-class (Sacrum=1/LeftHip=2/RightHip=3) 는 그대로 호환되고, Dataset539 의
# 5-class (Femur=4) 채널이 추가된다. 본 dict 는 Dataset537 (3-anatomy) 과
# Dataset538 (4-anatomy) 모두에서 anatomy-specific Ds532/539 prob 채널을 찾을 때
# 사용된다.
V5_DATASET532_PROB_CHANNEL = {"Sacrum": 1, "LeftHip": 2, "RightHip": 3, "Femur": 4}


def _bone_lut_normalize(arr: np.ndarray) -> np.ndarray:
    """Deterministic bone-window LUT for the CT-LUT ablation."""
    x = arr.astype(np.float32, copy=False)
    xp = np.asarray([-1000.0, -200.0, 150.0, 700.0, 1500.0, 2000.0], dtype=np.float32)
    fp = np.asarray([0.0, 0.05, 0.35, 0.70, 0.92, 1.0], dtype=np.float32)
    return np.interp(np.clip(x, xp[0], xp[-1]), xp, fp).astype(np.float32)


# =============================================================================
# Bone-CT preprocessing helpers
# =============================================================================
def bone_window_clip(img_arr: np.ndarray,
                     hu_low: float = -1000.0,
                     hu_high: float = 2000.0) -> np.ndarray:
    """Clip CT HU to a bone-relevant range BEFORE any further normalization.

    v8 best practice (PENGWIN'26 dataset 실측 기반):
        2026 audit: Pelvic bone HU p1=-69, p99=1313, max=1963 (rare metal artifact
        gets up to 19242). Femur bone p1=-71, p99=1464, max=1597.
        → Clipping at [-1000, 2000] preserves 99.99% of bone signal while
        truncating extreme metal artifacts (causes z-score to inflate).
        nnU-Net's default CT-percentile normalization handles z-score AFTER this.

    Why we clip BEFORE nnU-Net normalization:
        Default nnU-Net uses percentile-based clip (0.5/99.5%) computed PER-CASE,
        which works but inflates the z-score variance when a few metal artifacts
        push the 99.5% to >5000. Pre-clipping at -1000/2000 produces tighter,
        more uniform z-scores across cases.

    Args:
        img_arr: input CT volume (z, y, x), HU values (int or float).
        hu_low, hu_high: clip range. Defaults [-1000, 2000].

    Returns:
        np.ndarray, same shape, same dtype. In-range values unchanged; outliers clipped.
    """
    return clip_ct_hu(img_arr, (hu_low, hu_high))


def canonicalize_and_clip_image(img: sitk.Image) -> tuple[sitk.Image, np.ndarray]:
    """Return an LPS-oriented CT image and its bone-window-clipped array.

    Audit note:
        The previous raw nnU-Net datasets were generated while source cases
        still had mixed LPS/RAS directions. That silently poisons left/right
        anatomy learning and makes old checkpoints unsuitable as final
        evidence. Every builder must pass CT through this helper before writing
        `imagesTr`, so rebuild audits can assert all images are LPS.
    """
    img_lps = canonicalize_sitk(img)
    arr = bone_window_clip(sitk.GetArrayFromImage(img_lps))
    return img_lps, arr


def assert_lps_image(img: sitk.Image, context: str) -> None:
    """Fail fast if a generated image is not in the canonical LPS orientation."""
    code = orientation_code(img)
    if code != "LPS":
        raise RuntimeError(f"{context}: expected LPS orientation, got {code}")


def assert_matching_geometry(img: sitk.Image, lbl: sitk.Image, context: str) -> None:
    """Fail fast if CT/label geometry diverges after canonicalization."""
    if img.GetSize() != lbl.GetSize():
        raise RuntimeError(f"{context}: image/label size mismatch {img.GetSize()} != {lbl.GetSize()}")
    if not np.allclose(img.GetSpacing(), lbl.GetSpacing(), rtol=0, atol=1e-5):
        raise RuntimeError(f"{context}: image/label spacing mismatch {img.GetSpacing()} != {lbl.GetSpacing()}")
    if not np.allclose(img.GetOrigin(), lbl.GetOrigin(), rtol=0, atol=1e-4):
        raise RuntimeError(f"{context}: image/label origin mismatch {img.GetOrigin()} != {lbl.GetOrigin()}")
    if not np.allclose(img.GetDirection(), lbl.GetDirection(), rtol=0, atol=1e-5):
        raise RuntimeError(f"{context}: image/label direction mismatch")


def _select_case_subset(cases: list[Path], case_subset: list[str] | None) -> list[Path]:
    """Filter source case directories by zero-padded case IDs."""
    if not case_subset:
        return cases
    wanted = {str(c).zfill(3) for c in case_subset}
    selected = [cd for cd in cases if cd.name.zfill(3) in wanted]
    missing = sorted(wanted - {cd.name.zfill(3) for cd in selected})
    if missing:
        raise FileNotFoundError(f"case subset missing source cases: {missing}")
    return selected












def _load_probability_npz(path: Path) -> np.ndarray:
    data = np.load(path)
    for key in ("probabilities", "softmax"):
        if key in data:
            arr = data[key]
            break
    else:
        keys = list(data.keys())
        if not keys:
            raise RuntimeError(f"empty probability npz: {path}")
        arr = data[keys[0]]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 4:
        raise RuntimeError(f"expected probability array (C,Z,Y,X) in {path}, got {arr.shape}")
    return arr


def ndi_distance_transform(mask: np.ndarray,
                           spacing_zyx: tuple[float, float, float]) -> np.ndarray:
    from scipy import ndimage as ndi

    return ndi.distance_transform_edt(mask, sampling=spacing_zyx).astype(np.float32, copy=False)


def _selected_anatomy_sdf_from_prob(prob: np.ndarray,
                                    spacing_zyx: tuple[float, float, float],
                                    clip_mm: float = 40.0) -> np.ndarray:
    """Signed distance for one Dataset532 anatomy probability channel.

    [AUDIT][Risk:Major][Scope:v7_input_context]
    This channel is deterministic inference context, not a target-derived
    feature. It may come from a trained Dataset532 checkpoint, so the build
    report records the checkpoint and input profile. For production validation,
    this must be regenerated out-of-fold; six-case root-cause overfits use it
    only to test whether anatomy support context fixes CT-only support leakage.

    [QC][Invariant:range]
    Values are clipped/scaled to [-1, 1] and saved as nnU-Net `nonorm`
    channels. Keeping the range bounded prevents the SDF from dominating the
    CT-LUT channel solely by numeric scale.
    """
    arr = np.asarray(prob, dtype=np.float32)
    mask = arr >= 0.5
    if not mask.any():
        return np.zeros_like(arr, dtype=np.float32)
    inside = ndi_distance_transform(mask, spacing_zyx)
    outside = ndi_distance_transform(~mask, spacing_zyx)
    sdf = np.clip(inside - outside, -float(clip_mm), float(clip_mm)) / float(clip_mm)
    return sdf.astype(np.float32, copy=False)


def _anatomy_context_root(foundation_ds_id: int = 532) -> Path:
    """[V0.x][FIX:C2][2026-06-01] anatomy-prob 캐시 경로를 foundation_ds_id 로 키잉.

    이전: 경로가 항상 ".../anatomy_context_ds532_checkpoint_best" 로 고정되어, Ds537
    빌드(foundation=Ds532, 4-class)와 Ds538 빌드(foundation=Ds539, 5-class w/ Femur)가
    같은 캐시 dir 을 공유했다. 먼저 돈 빌드의 확률맵을 다음 빌드가 (잘못된 foundation 의
    출력임에도) image.npz 존재만 보고 조용히 재사용하는 cross-foundation 오염 위험.

    현재: 기본 경로를 ".../anatomy_context_ds{foundation_ds_id}_checkpoint_best" 로 분리.
    명시적 PENGWIN_ABBC_ANATOMY_CONTEXT_ROOT override 는 사용자 의도이므로 그대로 존중한다.
    """
    override = os.environ.get("PENGWIN_ABBC_ANATOMY_CONTEXT_ROOT", "").strip()
    if override:
        return Path(override)
    return Path(str(RESULT_VISUALIZE / f"anatomy_context_ds{foundation_ds_id}_checkpoint_best"))


def generate_anatomy_probability_context(cases: list[Path],
                                         foundation_ds_id: int = 539,
                                         force: bool = False,
                                         gpu: int = 0,
                                         checkpoint: str = "checkpoint_best.pth") -> Path:
    """EXPLICIT Stage-A inference step — the ONLY place build-time inference runs.

    Runs the foundation anatomy model (Ds532 4-class pelvic, or Ds539 5-class
    with Femur) over `cases` and writes the per-case softmax probability cache
    `<root>/<cid>/image.npz`. This was previously the lazy tail of
    `_ensure_ds532_anatomy_probability_context`; it is now a standalone,
    user-invoked step so the dataset build never infers implicitly.

    인수:
        cases: 처리할 case Path 목록.
        foundation_ds_id: anatomy 모델의 Dataset ID. Dataset537 빌드 시 532,
                          Dataset538 빌드 시 539 사용.
        force: 기존 cache 무시하고 재생성.
        gpu: CUDA 디바이스 인덱스.
        checkpoint: nnUNet checkpoint 파일 이름.

    반환: 캐시 root 경로.
    """
    out_root = _anatomy_context_root(foundation_ds_id)
    missing = []
    for cd in cases:
        cid = cd.name.zfill(3)
        if force or not (out_root / cid / "image.npz").exists():
            missing.append(cd)
    if not missing:
        print(f"  anatomy context cache ready (foundation_ds={foundation_ds_id}): {out_root}")
        return out_root

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    import torch

    # [FIX:B3] DATASETS[532] 하드코딩 → DATASETS[foundation_ds_id] 동적 lookup.
    foundation_cfg = DATASETS[foundation_ds_id]
    # [V0.x][FIX:FT][2026-06-02] foundation trainer 이름을 env 로 override 가능하게.
    # registry 의 DATASETS[539]["trainer"]="PengwinTrainer" 와 달리, STU-Net 백본 전환 후엔
    # 실제 학습 trainer 가 PengwinTrainerSTUNetBaseAnatomyV301 이다. checkpoint 경로는
    # {trainer}__nnUNetResEncUNetLPlans__3d_fullres 로 구성되므로 trainer 이름이 맞아야 한다.
    # PENGWIN_FOUNDATION_TRAINER 로 지정(없으면 registry 기본값).
    foundation_trainer = os.environ.get(
        "PENGWIN_FOUNDATION_TRAINER", foundation_cfg["trainer"]
    )
    model_root = (
        NN_RES / foundation_cfg["name"]
        / f"{foundation_trainer}__nnUNetResEncUNetLPlans__3d_fullres"
    )
    if not (model_root / "fold_0" / checkpoint).exists():
        raise FileNotFoundError(
            f"Foundation dataset {foundation_ds_id} ({foundation_cfg['name']}) checkpoint missing: "
            f"{model_root / 'fold_0' / checkpoint}"
        )
    device = torch.device("cuda", int(gpu)) if torch.cuda.is_available() else torch.device("cpu")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_root), use_folds=(0,), checkpoint_name=checkpoint,
    )
    print(f"  generating Ds{foundation_ds_id} probability context for {len(missing)} cases -> {out_root}")
    source_lists = []
    output_truncated = []
    meta_rows = []
    foundation_raw = NN_RAW / foundation_cfg["name"] / "imagesTr"
    for cd in missing:
        cid = cd.name.zfill(3)
        out_dir = out_root / cid
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_input = foundation_raw / f"PENGWIN_{cid}_0000.mha"
        if not raw_input.exists():
            # [DATA][Scope:anatomy_context][Risk:Major]
            # Aggressive cleanup may remove nnU-Net raw Dataset532 while keeping
            # the trained checkpoint. Rebuilding the full anatomy raw dataset
            # just to run a six-case V7 diagnostic wastes disk. Materialize only
            # the canonicalized CT required by this context cache; labels are not
            # needed for inference and no Dataset537 target is derived from this
            # temporary input.
            img, arr_img = canonicalize_and_clip_image(sitk.ReadImage(str(cd / "image.mha")))
            raw_input.parent.mkdir(parents=True, exist_ok=True)
            save_full_mha(arr_img, img, raw_input, dtype=arr_img.dtype)
        source_lists.append([str(raw_input)])
        output_truncated.append(str(out_dir / "image"))
        meta_rows.append((cid, cd, raw_input, out_dir))
    predictor.predict_from_files(
        source_lists,
        output_truncated,
        save_probabilities=True,
        overwrite=True,
        num_processes_preprocessing=max(1, min(8, int(os.environ.get("PENGWIN_ABBC_CONTEXT_PREPROC", "4")))),
        num_processes_segmentation_export=max(1, min(4, int(os.environ.get("PENGWIN_ABBC_CONTEXT_EXPORT", "2")))),
    )
    for cid, cd, raw_input, out_dir in meta_rows:
        (out_dir / "context_meta.json").write_text(json.dumps({
            "case": cid,
            "source_image": str(cd / "image.mha"),
            "prepared_input": str(raw_input),
            "dataset": 532,
            "checkpoint": checkpoint,
            "model_root": str(model_root),
            "channels": ["Sacrum", "LeftHip", "RightHip"],
        }, indent=2))
    return out_root


def require_anatomy_probability_context(cases: list[Path],
                                        foundation_ds_id: int = 539) -> Path:
    """READ-ONLY guard — returns the anatomy-prob cache root, NEVER infers.

    The build consumes this cache. If any case is missing
    `<root>/<cid>/image.npz`, raise a clear FileNotFoundError telling the user
    to run the explicit `generate-anatomy-prob` stage first. This function does
    not load nnUNet, does not touch the GPU, and never writes anything.
    """
    out_root = _anatomy_context_root(foundation_ds_id)
    missing = [cd.name.zfill(3) for cd in cases
               if not (out_root / cd.name.zfill(3) / "image.npz").exists()]
    if missing:
        raise FileNotFoundError(
            f"anatomy-prob cache missing for {len(missing)} cases under {out_root}; "
            f"run: python -m gen_nnunet_dataset --stage generate-anatomy-prob "
            f"--foundation {foundation_ds_id}  "
            "(the build no longer runs inference implicitly)"
        )
    return out_root


def build_anatomy_semantic_dataset(ds_id: int, force: bool = False) -> int:
    """Build a trusted CT → semantic anatomy nnU-Net raw dataset.

    Contract:
        - Use only original GT labels from `/workspace/data/task1_2/extracted`.
        - Collapse fragment IDs into the dataset's local anatomy IDs without
          inventing supervision.
        - Keep background as one class; air/soft tissue/table/other-bone labels
          are not verified in Task1/2.
        - Canonicalize both CT and label to LPS so LH/RH semantics are stable.
    """
    cfg = DATASETS[ds_id]
    dst = NN_RAW / cfg["name"]
    (dst / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dst / "labelsTr").mkdir(parents=True, exist_ok=True)
    cases = list_cases(case_filter=None if cfg["filter"] == "all" else cfg["filter"])
    local_label_map = {
        anat: local_idx for local_idx, anat in enumerate(cfg["anatomies"], start=1)
    }
    global_label_map = {
        anat: ANATOMY_NAMES.index(anat) + 1 for anat in cfg["anatomies"]
    }
    print(
        f"[anatomy] Ds{ds_id} {cfg['name']} — {len(cases)} trusted labeled CT cases "
        f"anatomies={cfg['anatomies']}"
    )

    # [V0.x][FIX:W1][2026-06-01] --overwrite(force) 시 현재 case 집합에 없는 orphan
    # 파일만 정리한다. force 는 기존엔 case 파일을 덮어쓰기만 하고 이전 빌드의 다른
    # case 집합에서 남은 파일은 그대로 두어, "깨끗한 rebuild" 가 아니었다. blanket
    # rmtree 대신 expected 파일명에 없는 것만 제거해 안전하게 stale 만 지운다.
    if force:
        expected_imgs = {f"PENGWIN_{int(cd.name):03d}_0000.mha" for cd in cases}
        expected_lbls = {f"PENGWIN_{int(cd.name):03d}.mha" for cd in cases}
        removed = 0
        for f in (dst / "imagesTr").glob("*.mha"):
            if f.name not in expected_imgs:
                f.unlink(); removed += 1
        for f in (dst / "labelsTr").glob("*.mha"):
            if f.name not in expected_lbls:
                f.unlink(); removed += 1
        if removed:
            print(f"  [overwrite] removed {removed} stale orphan file(s) from {cfg['name']}")

    for i, cd in enumerate(cases):
        cid = int(cd.name)
        out_img = dst / "imagesTr" / f"PENGWIN_{cid:03d}_0000.mha"
        out_lbl = dst / "labelsTr" / f"PENGWIN_{cid:03d}.mha"
        if out_img.exists() and out_lbl.exists() and not force:
            continue

        img, arr_img = canonicalize_and_clip_image(sitk.ReadImage(str(cd / "image.mha")))
        lbl_img = canonicalize_sitk(sitk.ReadImage(str(cd / "label.mha")))
        assert_lps_image(img, f"Ds{ds_id} case {cid:03d} CT")
        assert_lps_image(lbl_img, f"Ds{ds_id} case {cid:03d} label")
        assert_matching_geometry(img, lbl_img, f"Ds{ds_id} case {cid:03d}")

        inst = sitk.GetArrayFromImage(lbl_img)
        global_anat = inst_to_anat(inst)
        anat = np.zeros_like(global_anat, dtype=np.uint8)
        for name in cfg["anatomies"]:
            anat[global_anat == global_label_map[name]] = local_label_map[name]
        values = set(int(v) for v in np.unique(anat))
        allowed = set(range(int(cfg["n_classes"])))
        if not values.issubset(allowed):
            raise RuntimeError(
                f"Ds{ds_id} case {cid:03d}: labels {sorted(values)} outside {sorted(allowed)}"
            )
        if not (anat > 0).any():
            raise RuntimeError(f"Ds{ds_id} case {cid:03d}: empty foreground after anatomy remap")

        save_full_mha(arr_img, img, out_img, dtype=arr_img.dtype)
        save_full_mha(anat, lbl_img, out_lbl, dtype=np.uint8)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(cases)}]")

    labels = {"background": 0}
    labels.update({name: idx for name, idx in local_label_map.items()})
    ds_json = {
        "channel_names": {"0": "CT"},
        "labels": labels,
        "numTraining": len(cases),
        "file_ending": ".mha",
        "description": (
            f"PENGWIN 2026 — {cfg['name']} split semantic anatomy target. "
            f"Original fragment IDs are collapsed to background/{'/'.join(cfg['anatomies'])}; "
            "no TotalSegmentator, Task3, pseudo-label, or unlabeled source is included."
        ),
    }
    (dst / "dataset.json").write_text(json.dumps(ds_json, indent=2))
    print(f"  ✅ {cfg['name']} ready")
    return len(cases)


def build_bicm_v5_dataset(ds_id: int,
                          force: bool = False,
                          v5_input: str = "ct_lut",
                          v5_target_profile: str = "v5_tiny_marker",
                          v5_core_ball_radius_mm: float = 2.5,
                          v5_core_body_mm: float = 3.0,
                          v5_contact_band_mm: float = 2.0,
                          label_mode: str = "instance",
                          case_subset: list[str] | None = None) -> int:
    """Dataset537 per-anatomy BICM V5 raw 데이터를 만든다.

    [AUDIT][Risk:High][Scope:pipeline_reset]
    V5에서는 V4의 factorized / global-pelvis 경로를 더 이상 쓰지 않는다.
    대신 anatomy 정보를 이용해 Sacrum / LeftHip / RightHip 각각에 대해
    샘플당 ROI를 하나씩 crop한다. 기본 모델 입력은 CT-LUT 단일 채널이고,
    V7 진단용 입력 프로파일에서는 target과 trainer는 그대로 둔 채
    선택된 Dataset532 anatomy probability / SDF 채널을 추가할 수 있다.

    [DATA][Leakage]
    라벨은 오직 원본 Task1 GT instance map에서만 생성된다. `ct_lut_anat_sdf`
    가 지정되어 Dataset532 prediction이 사용될 때도, 그 값은 input context
    로만 저장될 뿐 라벨 생성이나 케이스 선정에는 절대 영향을 주지 않는다.

    [DECOUPLE][2026-06-12]
    빌드는 더 이상 build-time 추론을 실행하지 않는다. anatomy-prob 캐시는
    require_anatomy_probability_context() 로 READ-ONLY 소비되며, 캐시가 없으면
    명확한 FileNotFoundError 로 generate-anatomy-prob 스테이지 실행을 안내한다.
    """
    if v5_input not in V5_INPUT_VARIANTS:
        raise ValueError(f"--v5-input must be one of {V5_INPUT_VARIANTS}, got {v5_input!r}")
    if v5_target_profile not in V5_TARGET_PROFILES:
        raise ValueError(f"--v5-target-profile must be one of {V5_TARGET_PROFILES}, got {v5_target_profile!r}")
    cfg = DATASETS[ds_id]
    if cfg["kind"] != "bicm_v5":
        raise ValueError(f"Dataset{ds_id} is not a BICM V5 dataset")
    dst = NN_RAW / cfg["name"]
    images_dir = dst / "imagesTr"
    labels_dir = dst / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    # [V0.x][FIX:B2][2026-05-31] cfg["filter"] 을 반영한 case 선정.
    # 이전: 하드코딩 list_cases("pelvic") — 170 femur-only 케이스 누락.
    # 현재: cfg.filter ("all" / "pelvic" / "femur") 을 그대로 list_cases 에 전달.
    cases = _select_case_subset(list_cases(cfg["filter"]), case_subset)
    params = BICMV5Params(
        target_profile=v5_target_profile,
        core_ball_radius_mm=float(v5_core_ball_radius_mm),
        core_body_mm=float(v5_core_body_mm),
        contact_band_mm=float(v5_contact_band_mm),
    )
    pad_vox = int(os.environ.get("PENGWIN_V5_ROI_PAD_VOX", "24"))
    uses_anatomy_context = (v5_input == "ct_lut_anat_sdf") and label_mode != "instance"
    context_checkpoint = os.environ.get("PENGWIN_V7_CONTEXT_CHECKPOINT", "checkpoint_best.pth")
    context_root = None
    if uses_anatomy_context:
        # [AUDIT][Risk:High][Scope:input_ablation]
        # V7 root-cause 실험에서 이 변수 하나만 바꾼다. target, decoder, trainer,
        # loss, ROI 샘플은 그대로 고정한 채, 모델만 선택된 Dataset532 support
        # context를 추가로 받는다. 캐시 경로와 checkpoint를 아래에 기록해 두면,
        # 이후 fold0 작업에서 fold0를 몰래 재사용하는 일 없이 이 부분을
        # out-of-fold anatomy probability로 교체할 수 있다.
        # [V0.x][FIX:B3][2026-05-31] cfg["foundation_dataset"] 전달.
        # Ds537 (foundation=532) → Ds532 anatomy prob 사용 (기존 동작).
        # Ds538 (foundation=539) → Ds539 anatomy prob 사용 (Femur 포함 5-class).
        # [DECOUPLE][2026-06-12] _ensure_ds532_anatomy_probability_context(...) 호출을
        # require_anatomy_probability_context(...) 로 교체 — 빌드는 캐시를 READ-ONLY 로만
        # 소비하고 절대 추론하지 않는다. 캐시 생성은 generate-anatomy-prob 스테이지가 담당.
        foundation_ds = int(cfg["foundation_dataset"] or 532)
        context_root = require_anatomy_probability_context(cases, foundation_ds_id=foundation_ds)
    if uses_anatomy_context:
        channel_names = {"0": "nonorm", "1": "nonorm", "2": "nonorm"}
        input_contract = [
            "ct_lut",
            "Dataset532_selected_anatomy_probability",
            "Dataset532_selected_anatomy_sdf",
        ]
    else:
        channel_names = {"0": "nonorm"}
        input_contract = ["ct_lut"]
    rows = []
    max_k = 0  # instance 모드: dataset.json 라벨 선언용 (관측된 최대 fragment 수)
    # [V0.x][FIX:B2][2026-05-31] anatomies 카운트는 cfg["anatomies"] 기반.
    # 단 cfg_anatomies 는 아래 loop 안에서 정의되므로 여기선 cfg 직접 참조.
    _cfg_anat_count = len(cfg["anatomies"]) if cfg["anatomies"] else len(V5_ANATOMY_RANGES_WITH_FEMUR)
    _cfg_filter_desc = {"pelvic": "pelvic", "femur": "femur", "all": "pelvic+femur"}.get(cfg["filter"], cfg["filter"])
    print(
        f"[bicm_v5] Ds{ds_id} {cfg['name']} — {len(cases)} {_cfg_filter_desc} CT cases "
        f"x {_cfg_anat_count} anatomy ROI samples input={v5_input} "
        f"target={v5_target_profile}"
    )
    for cd in cases:
        cid = int(cd.name)
        img, arr_img = canonicalize_and_clip_image(sitk.ReadImage(str(cd / "image.mha")))
        lbl_img = canonicalize_sitk(sitk.ReadImage(str(cd / "label.mha")))
        assert_lps_image(img, f"Ds{ds_id} case {cid:03d} CT")
        assert_lps_image(lbl_img, f"Ds{ds_id} case {cid:03d} label")
        assert_matching_geometry(img, lbl_img, f"Ds{ds_id} case {cid:03d}")
        inst_full = sitk.GetArrayFromImage(lbl_img).astype(np.uint16)
        spacing_zyx = tuple(float(v) for v in lbl_img.GetSpacing()[::-1])
        image_channel = _bone_lut_normalize(arr_img)
        context_probs: dict[str, np.ndarray] = {}
        if uses_anatomy_context:
            assert context_root is not None
            prob_path = context_root / f"{cid:03d}" / "image.npz"
            probs = _load_probability_npz(prob_path)
            if tuple(probs.shape[1:]) != tuple(arr_img.shape):
                raise RuntimeError(
                    f"Dataset532 context shape mismatch for case {cid:03d}: "
                    f"prob={tuple(probs.shape[1:])}, ct={tuple(arr_img.shape)}"
                )
            # [V0.x][FIX:B1+B2][2026-05-31] cfg.anatomies 에 포함된 채널만 검증.
            # 이전: V5_DATASET532_PROB_CHANNEL 의 max (=4, Femur 채널 포함) 와 비교.
            # 현재: cfg.anatomies 에 실제로 필요한 채널의 max 와만 비교.
            # Ds537 (foundation=Ds532, 4-class) 는 Femur 채널 없이도 동작,
            # Ds538 (foundation=Ds539, 5-class) 는 Femur 채널까지 검증.
            cfg_required_anats = cfg["anatomies"] if cfg["anatomies"] else list(V5_DATASET532_PROB_CHANNEL)
            cfg_required_channels = {a: V5_DATASET532_PROB_CHANNEL[a] for a in cfg_required_anats if a in V5_DATASET532_PROB_CHANNEL}
            if not cfg_required_channels:
                raise RuntimeError(f"Ds{ds_id} cfg.anatomies={cfg_required_anats} 에 해당하는 prob 채널이 없음")
            if probs.shape[0] <= max(cfg_required_channels.values()):
                raise RuntimeError(
                    f"foundation dataset context for case {cid:03d} has {probs.shape[0]} channels; "
                    f"need anatomy channels {cfg_required_channels}"
                )
            for anatomy, prob_channel in cfg_required_channels.items():
                prob = np.asarray(probs[int(prob_channel)], dtype=np.float32)
                context_probs[anatomy] = prob
        # [V0.x][FIX:B2][2026-05-31] cfg.anatomies 기반 반복 (Femur 포함 시 4개).
        # 이전: 하드코딩 V5_ANATOMY_RANGES (3 anatomies) — Femur 미지원.
        # 현재: cfg["anatomies"] 의 4-anatomy (Dataset538) 또는 3-anatomy (Dataset537) 양쪽 동작.
        # bbox 가 비어있을 경우, femur-only 케이스에서 Sacrum/LH/RH 의 빈 ROI 가 발생할 수 있으므로
        # raise 대신 continue 로 건너뛴다 (Dataset538 "all" filter 동작).
        cfg_anatomies = list(cfg["anatomies"]) if cfg["anatomies"] else list(V5_ANATOMY_RANGES_WITH_FEMUR)
        for anatomy in cfg_anatomies:
            if anatomy not in V5_ANATOMY_RANGES_WITH_FEMUR:
                raise ValueError(f"Ds{ds_id} unknown anatomy {anatomy!r}; supported: {sorted(V5_ANATOMY_RANGES_WITH_FEMUR)}")
            sample_id = f"PENGWIN_{cid:03d}_{anatomy}"
            out_imgs = [images_dir / f"{sample_id}_{idx:04d}.mha" for idx in range(len(channel_names))]
            out_lbl = labels_dir / f"{sample_id}.mha"
            if all(p.exists() for p in out_imgs) and out_lbl.exists() and not force:
                continue
            anat_mask = anatomy_mask_from_instances(inst_full, anatomy)
            bbox = bbox_from_mask(anat_mask, pad_vox=pad_vox)
            if bbox is None:
                # cfg.filter="all" 시 femur-only 케이스에서 pelvic 빈 ROI 발생 가능.
                # 또는 pelvic-only 케이스에서 Femur 빈 ROI 발생 가능. 양쪽 모두 정상 skip.
                continue
            lo, hi = V5_ANATOMY_RANGES_WITH_FEMUR[anatomy]
            inst_roi = inst_full[bbox]
            inst_roi = np.where((inst_roi >= lo) & (inst_roi <= hi), inst_roi, 0).astype(np.uint16, copy=False)
            if label_mode == "instance":
                # leak-free target: the per-anatomy fragment instance map itself, relabeled to
                # contiguous 1..K (bg=0). No ABBC-semantic conversion, no sidecar — the custom loss
                # reads this label directly. Anatomy identity is carried by the ROI (sample_id), not
                # by an input channel. See docs/Plan.md Phase 0 / [[pengwin-instance-label-nosidecar]].
                uniq = [int(v) for v in np.unique(inst_roi) if v != 0]
                if not uniq:
                    continue  # this anatomy has no fragment in the ROI -> skip
                remap = {v: i + 1 for i, v in enumerate(uniq)}
                target = np.zeros(inst_roi.shape, dtype=np.uint8)
                for v, n in remap.items():
                    target[inst_roi == v] = n
                max_k = max(max_k, len(uniq))
            else:
                target = compute_bicm_v5_target(inst_roi, spacing_zyx=spacing_zyx, params=params)
                values = set(int(v) for v in np.unique(target))
                allowed = set(V5_LABELS.values())
                if not values.issubset(allowed):
                    raise RuntimeError(f"{sample_id}: V5 labels {sorted(values)} outside {sorted(allowed)}")
                if not (target == V5_LABELS["core"]).any():
                    raise RuntimeError(f"{sample_id}: no V5 core voxels")
            crop_save_mha(image_channel, img, bbox, out_imgs[0], dtype=np.float32)
            row_extra = {}
            if uses_anatomy_context:
                prob = context_probs[anatomy]
                # [DATA][Leakage]
                # 선택된 anatomy 채널은 이번 6-케이스 overfit 진단을 위해 CT와
                # 동일한 GT ROI로 crop된다. 타겟 생성에는 쓰이지 않지만, 이
                # 프로파일을 정식 승격(promotion) 근거로 쓰려면 production
                # fold0에서는 캐시를 out-of-fold anatomy prediction으로 반드시
                # 교체해야 한다.
                crop_save_mha(prob, img, bbox, out_imgs[1], dtype=np.float32)
                prob_roi = prob[bbox]
                # [QC][Perf:roi_sdf]
                # 6개의 큰 CT 전체 볼륨에 대해 EDT를 돌리면 수 분이 걸리는데,
                # 실험의 결론에는 영향을 주지 않는다. 모델은 이 crop 밖의 voxel을
                # 절대 보지 않으므로, 실제 materialize된 ROI 내부에서만 SDF를
                # 계산해 시간을 아낀다.
                sdf_roi = _selected_anatomy_sdf_from_prob(prob_roi, spacing_zyx=spacing_zyx)
                save_crop_array_mha(sdf_roi, img, bbox, out_imgs[2], dtype=np.float32)
                row_extra = {
                    "anatomy_prob_mean": float(np.mean(prob_roi)),
                    "anatomy_prob_fg_fraction": float(np.mean(prob_roi >= 0.5)),
                    "anatomy_sdf_mean": float(np.mean(sdf_roi)),
                }
            save_crop_array_mha(target, lbl_img, bbox, out_lbl, dtype=np.uint8)
            row = {
                "case": f"{cid:03d}",
                "sample": sample_id,
                "anatomy": anatomy,
                "bbox_zyx": [[int(s.start), int(s.stop)] for s in bbox],
                "labels": label_distribution(target),
                "input_channels": len(channel_names),
            }
            row.update(row_extra)
            rows.append(row)
    ds_json = {
        "channel_names": channel_names,
        "labels": (
            {"background": 0, **{f"fragment_{i:02d}": i for i in range(1, max(max_k, 1) + 1)}}
            if label_mode == "instance" else dict(V5_LABELS)
        ),
        # [V0.x][FIX:NT][2026-06-03] numTraining 은 실제 written ROI 수(len(rows)).
        # 이전: len(cases) * len(cfg_anatomies) — pelvic 케이스엔 Femur ROI 가 없고
        # femur 케이스엔 pelvic 3-anat ROI 가 없어 빈 ROI 가 skip 되므로 과대계산되어
        # nnUNetv2_plan_and_preprocess --verify_dataset_integrity 가 실패했다
        # (subset 3×4=12 vs 실제 5; full 340×4=1360 vs 실제 680). rows 는 위 루프에서
        # 실제로 기록된 ROI 샘플만 모으므로 dataset.json 의 파일 수와 정확히 일치한다.
        "numTraining": len(rows),
        "file_ending": ".mha",
        "v5_contract": {
            "input": v5_input,
            "input_contract": input_contract,
            "roi_source": (
                "GT anatomy IDs for raw target/oracle ROI; Dataset532 context, "
                "if present, is input-only and must be out-of-fold for promotion"
            ),
            "anatomy_context": {
                "enabled": uses_anatomy_context,
                # [V0.x][FIX:B3][2026-05-31] foundation_dataset 동적 기록.
                "dataset": DATASETS[int(cfg["foundation_dataset"] or 532)]["name"] if uses_anatomy_context else None,
                "foundation_dataset_id": int(cfg["foundation_dataset"] or 532) if uses_anatomy_context else None,
                "checkpoint": context_checkpoint if uses_anatomy_context else None,
                "cache_root": str(context_root) if uses_anatomy_context and context_root is not None else None,
                "channel_map": V5_DATASET532_PROB_CHANNEL if uses_anatomy_context else None,
                "sdf_clip_mm": 40.0 if uses_anatomy_context else None,
            },
            "anatomies": list(cfg_anatomies),
            "target_params": {
                "exterior_mm": params.exterior_mm,
                "core_mm": params.core_mm,
                "core_fallback_radius_vox": params.core_fallback_radius_vox,
                "target_profile": params.target_profile,
                "core_ball_radius_mm": params.core_ball_radius_mm,
                "core_body_mm": params.core_body_mm,
                "contact_band_mm": params.contact_band_mm,
                "roi_pad_vox": pad_vox,
            },
            "decoder": "core connected components + contact-surface watershed, no threshold sweep",
        },
        "description": (
            f"PENGWIN 2026 — {cfg['name']} V5 per-anatomy BICM target. "
            "One sample is one CT-LUT anatomy ROI. Labels are background, "
            "exterior context, interior shell, core, and contact surface."
        ),
    }
    (dst / "dataset.json").write_text(json.dumps(ds_json, indent=2))
    audit_path = RESULT_REPORT / f"build_bicm_v5_dataset{ds_id}_{RESULT_DATE}.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({
        "dataset": cfg["name"],
        "input_variant": v5_input,
        "input_contract": input_contract,
        "anatomy_context_root": str(context_root) if context_root is not None else None,
        "anatomy_context_checkpoint": context_checkpoint if uses_anatomy_context else None,
        "target_profile": params.target_profile,
        "core_ball_radius_mm": params.core_ball_radius_mm,
        "core_body_mm": params.core_body_mm,
        "contact_band_mm": params.contact_band_mm,
        "samples": rows,
    }, indent=2))
    print(f"  raw audit: {audit_path}")
    print(f"  ✅ {cfg['name']} ready")
    # [V0.x][FIX:B2][2026-05-31] 실제 written sample 수 (rows) 반환.
    # 이전: len(cases) × 3 (V5_ANATOMY_RANGES 길이) — Femur 누락 + skip 미반영.
    # 현재: 실제 빌드된 sample 수. Dataset538 "all" 시 femur-only 케이스에서
    # pelvic 3 anat skip 되므로 정확한 count 반환.
    return len(rows)





















































BICM_V5_INSTANCE_SIDECAR_DIR = "bicm_v5_instance_targets"
BOUNDARY_FRAGMENT_V3_TARGET_SIDECAR_DIR = "boundary_fragment_v3_targets"


PENGWIN_BICM_V5_SIDECAR_MAX_LOCATIONS = int(
    os.environ.get("PENGWIN_BICM_V5_SIDECAR_MAX_LOCATIONS", "30000")
)


def _instance_centers_and_sizes(inst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-fragment (MAX_INSTANCE_ID+1, 3) centroids and voxel counts.

    Index 0 is reserved for background and stays NaN/0. Indices outside the valid
    1..MAX_INSTANCE_ID range (registry: Sacrum 1-50 ... Femur 151-200) are also
    left as NaN/0 so the sidecar contract matches the BICM V5 sidecar loader which
    addresses fragments by global PENGWIN fragment ID. Sized to the full registry
    range so per-anatomy Femur ROIs (IDs 151-200) no longer overflow the array.
    """
    n_slots = MAX_INSTANCE_ID + 1
    centers = np.full((n_slots, 3), np.nan, dtype=np.float32)
    sizes = np.zeros((n_slots,), dtype=np.int64)
    valid = valid_instance_mask(inst)
    if not valid.any():
        return centers, sizes
    flat_ids = inst[valid].astype(np.int64, copy=False)
    coords = np.argwhere(valid).astype(np.float32, copy=False)
    sizes_view = np.bincount(flat_ids, minlength=n_slots).astype(np.int64, copy=False)
    sizes[: sizes_view.shape[0]] = sizes_view[: sizes.shape[0]]
    sum_z = np.bincount(flat_ids, weights=coords[:, 0], minlength=n_slots)
    sum_y = np.bincount(flat_ids, weights=coords[:, 1], minlength=n_slots)
    sum_x = np.bincount(flat_ids, weights=coords[:, 2], minlength=n_slots)
    nonzero = np.flatnonzero(sizes_view[:n_slots] > 0)
    nonzero = nonzero[valid_instance_mask(nonzero)]
    if nonzero.size > 0:
        denom = sizes_view[nonzero].astype(np.float32)
        centers[nonzero, 0] = (sum_z[nonzero] / denom).astype(np.float32)
        centers[nonzero, 1] = (sum_y[nonzero] / denom).astype(np.float32)
        centers[nonzero, 2] = (sum_x[nonzero] / denom).astype(np.float32)
    return centers, sizes


def _fast_same_anatomy_contact_mask(inst: np.ndarray) -> np.ndarray:
    """Same-anatomy adjacency contact mask on a single anatomy ROI.

    Each V5 raw sample is already cropped to one anatomy (Sacrum / LeftHip /
    RightHip), so any pair of nonzero fragment voxels touching across a
    6-neighbor face is by construction same-anatomy. Keeping the same-anatomy
    guard makes the helper safe even when called on a full-pelvic instance
    array.
    """
    inst = inst.astype(np.int32, copy=False)
    out = np.zeros(inst.shape, dtype=bool)
    for axis in range(3):
        a_sl = [slice(None), slice(None), slice(None)]
        b_sl = [slice(None), slice(None), slice(None)]
        a_sl[axis] = slice(1, None)
        b_sl[axis] = slice(None, -1)
        a = inst[tuple(a_sl)]
        b = inst[tuple(b_sl)]
        same_anatomy = (
            (a > 0) & (b > 0) & (a != b)
            & (((a - 1) // 50) == ((b - 1) // 50))
        )
        if same_anatomy.any():
            out_a = out[tuple(a_sl)]
            out_b = out[tuple(b_sl)]
            out_a[same_anatomy] = True
            out_b[same_anatomy] = True
            out[tuple(a_sl)] = out_a
            out[tuple(b_sl)] = out_b
    return out


def _sample_locations(mask: np.ndarray,
                      max_locations: int = PENGWIN_BICM_V5_SIDECAR_MAX_LOCATIONS,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Deterministically sample up to `max_locations` (z, y, x) coords from `mask`.

    Returns an (N, 3) int32 array. The sampler shrinks the mask down to a cap
    so the sidecar stays bounded for very large ROIs without losing positional
    diversity.
    """
    if mask.size == 0 or not np.any(mask):
        return np.zeros((0, 3), dtype=np.int32)
    coords = np.argwhere(mask).astype(np.int32, copy=False)
    cap = int(max(0, max_locations))
    if cap <= 0 or coords.shape[0] <= cap:
        return coords
    rng = rng if rng is not None else np.random.default_rng(0)
    idx = rng.choice(coords.shape[0], size=cap, replace=False)
    idx.sort()
    return coords[idx]


def _resample_instance_to_preprocessed(inst_roi: np.ndarray,
                                       raw_label_img: sitk.Image,
                                       pkl_properties: dict,
                                       target_shape_zyx: tuple[int, int, int],
                                       target_spacing_zyx: tuple[float, float, float]) -> np.ndarray:
    """Resample an anatomy-ROI instance ID array to the preprocessed grid.

    The instance ID grid is aligned in voxel space to the raw V5 label MHA
    (since both come from the same anatomy bbox). nnUNet preprocessing changes
    spacing/shape but keeps origin/direction. We mirror that by constructing a
    reference image at the preprocessed spacing/shape with the same origin and
    direction, then nearest-neighbor resampling.

    `target_shape_zyx` is the preprocessed grid shape (matches the `seg` tensor
    in the preprocessed `.npz`) and `target_spacing_zyx` is the plan spacing
    from `nnUNetPlans.json`. Both must be passed explicitly because the per-
    sample pkl stores only the *source* spacing of the original case.
    """
    src = sitk.GetImageFromArray(inst_roi.astype(np.uint16, copy=False))
    src.SetSpacing(raw_label_img.GetSpacing())
    src.SetOrigin(raw_label_img.GetOrigin())
    src.SetDirection(raw_label_img.GetDirection())

    target_spacing_xyz = (
        float(target_spacing_zyx[2]),
        float(target_spacing_zyx[1]),
        float(target_spacing_zyx[0]),
    )
    out_size_xyz = (
        int(target_shape_zyx[2]),
        int(target_shape_zyx[1]),
        int(target_shape_zyx[0]),
    )

    bbox = pkl_properties.get("bbox_used_for_cropping")
    if bbox is not None:
        crop_lo_zyx = (int(bbox[0][0]), int(bbox[1][0]), int(bbox[2][0]))
    else:
        crop_lo_zyx = (0, 0, 0)
    direction = raw_label_img.GetDirection()
    direction_mat = np.array(direction, dtype=np.float64).reshape(3, 3)
    # bbox_used_for_cropping is in (z, y, x); convert to (x, y, z) for sitk origin shift.
    crop_lo_xyz = np.array(
        [crop_lo_zyx[2], crop_lo_zyx[1], crop_lo_zyx[0]], dtype=np.float64
    )
    raw_spacing_xyz = np.array(raw_label_img.GetSpacing(), dtype=np.float64)
    shift_world = direction_mat @ (crop_lo_xyz * raw_spacing_xyz)
    new_origin = tuple(float(o + s) for o, s in zip(raw_label_img.GetOrigin(), shift_world))

    ref = sitk.Image(out_size_xyz, sitk.sitkUInt16)
    ref.SetSpacing(target_spacing_xyz)
    ref.SetOrigin(new_origin)
    ref.SetDirection(direction)

    resampled = sitk.Resample(
        src,
        ref,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt16,
    )
    return sitk.GetArrayFromImage(resampled).astype(np.uint16, copy=False)


def _build_one_bicm_v5_instance_sidecar(sample_id: str,
                                        raw_label_path: Path,
                                        source_case_dir: Path,
                                        preprocessed_pkl: Path,
                                        preprocessed_npz: Path,
                                        out_path: Path,
                                        max_locations: int,
                                        plan_target_spacing_zyx: tuple[float, float, float]) -> dict:
    """Build the .npz sidecar for one anatomy ROI sample."""
    import pickle

    parts = sample_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"unexpected sample_id format: {sample_id!r}")
    cid = int(parts[1])
    anatomy = "_".join(parts[2:])
    # Full 4-anatomy view: Ds538 builds a Femur ROI sample (anatomy="Femur",
    # IDs 151-200). Ds537 (pelvic-only) simply never passes "Femur", so the
    # superset is safe and the femur ROI is no longer rejected here.
    if anatomy not in V5_ANATOMY_RANGES_WITH_FEMUR:
        raise ValueError(f"{sample_id}: anatomy {anatomy!r} not in {sorted(V5_ANATOMY_RANGES_WITH_FEMUR)}")
    lo, hi = V5_ANATOMY_RANGES_WITH_FEMUR[anatomy]

    raw_lbl_img = canonicalize_sitk(sitk.ReadImage(str(raw_label_path)))
    raw_v5_arr = sitk.GetArrayFromImage(raw_lbl_img)

    src_lbl_img = canonicalize_sitk(sitk.ReadImage(str(source_case_dir / "label.mha")))
    inst_full = sitk.GetArrayFromImage(src_lbl_img).astype(np.uint16, copy=False)
    anat_mask = anatomy_mask_from_instances(inst_full, anatomy)
    pad_vox = int(os.environ.get("PENGWIN_V5_ROI_PAD_VOX", "24"))
    bbox = bbox_from_mask(anat_mask, pad_vox=pad_vox)
    if bbox is None:
        raise RuntimeError(f"{sample_id}: empty anatomy ROI")
    inst_roi = inst_full[bbox]
    inst_roi = np.where((inst_roi >= lo) & (inst_roi <= hi), inst_roi, 0).astype(np.uint16, copy=False)
    if inst_roi.shape != raw_v5_arr.shape:
        raise RuntimeError(
            f"{sample_id}: raw instance ROI shape {inst_roi.shape} != raw V5 label "
            f"shape {raw_v5_arr.shape} (bbox replay mismatch)"
        )

    with open(preprocessed_pkl, "rb") as fh:
        props = pickle.load(fh)
    preprocessed = np.load(preprocessed_npz)
    seg = np.asarray(preprocessed["seg"])
    if seg.ndim == 4:
        seg = seg[0]
    seg = seg.astype(np.int16, copy=False)

    target_shape_zyx = tuple(int(v) for v in seg.shape)
    instance = _resample_instance_to_preprocessed(
        inst_roi, raw_lbl_img, props,
        target_shape_zyx=target_shape_zyx,
        target_spacing_zyx=plan_target_spacing_zyx,
    )
    if instance.shape != seg.shape:
        raise RuntimeError(
            f"{sample_id}: resampled instance shape {instance.shape} != preprocessed seg shape {seg.shape}"
        )
    # The V5 raw target labels (0..4) define support/exterior/contact regions and
    # are the source of truth for the spatial location pools below. Instance IDs
    # add fragment identity but do not redefine the location masks.
    bg = V5_LABELS["background"]
    ext = V5_LABELS["exterior_context"]
    shell = V5_LABELS["interior_shell"]
    core = V5_LABELS["core"]
    contact = V5_LABELS["contact_surface"]
    support_mask = (seg == shell) | (seg == core) | (seg == contact)
    contact_mask = (seg == contact)
    edge_mask = _fast_same_anatomy_contact_mask(instance)
    if not contact_mask.any() and edge_mask.any():
        # Some V5 target profiles emit no `contact_surface` voxels even when
        # adjacent same-anatomy fragments touch; treat the adjacency mask as a
        # contact-location fallback in that case so the sidecar pool is never
        # empty for multi-fragment ROIs.
        contact_mask = edge_mask
    hard_negative_mask = (seg == ext)
    # Tiny fragments: per-fragment support below a small voxel threshold.
    centers_zyx, fragment_sizes = _instance_centers_and_sizes(instance)
    tiny_threshold = int(os.environ.get("PENGWIN_BICM_V5_TINY_FRAGMENT_VOXELS", "512"))
    tiny_mask = np.zeros_like(support_mask, dtype=bool)
    if support_mask.any():
        for fid in range(1, MAX_INSTANCE_ID + 1):
            sz = int(fragment_sizes[fid])
            if 0 < sz <= tiny_threshold:
                tiny_mask |= (instance == fid)

    rng = np.random.default_rng(int(cid) * 1000 + (lo // 50))
    contact_locations = _sample_locations(contact_mask, max_locations, rng)
    tiny_locations = _sample_locations(tiny_mask, max_locations, rng)
    support_locations = _sample_locations(support_mask & ~contact_mask, max_locations, rng)
    hard_negative_locations = _sample_locations(hard_negative_mask, max_locations, rng)
    edge_locations = _sample_locations(edge_mask, max_locations, rng)

    valid_fragments = [int(fid) for fid in range(1, MAX_INSTANCE_ID + 1) if int(fragment_sizes[fid]) > 0]
    audit = {
        "sample_id": sample_id,
        "case_id": f"{cid:03d}",
        "anatomy": anatomy,
        "anatomy_range": [int(lo), int(hi)],
        "raw_v5_shape": list(int(v) for v in raw_v5_arr.shape),
        "preprocessed_shape": list(int(v) for v in seg.shape),
        "fragment_count": len(valid_fragments),
        "valid_fragment_ids": valid_fragments,
        "voxel_counts": {
            "support": int(support_mask.sum()),
            "contact": int(contact_mask.sum()),
            "edge_adjacency": int(edge_mask.sum()),
            "tiny": int(tiny_mask.sum()),
            "hard_negative": int(hard_negative_mask.sum()),
            "background": int((seg == bg).sum()),
        },
        "location_counts": {
            "contact_locations": int(contact_locations.shape[0]),
            "tiny_locations": int(tiny_locations.shape[0]),
            "support_locations": int(support_locations.shape[0]),
            "hard_negative_locations": int(hard_negative_locations.shape[0]),
            "edge_locations": int(edge_locations.shape[0]),
        },
        "tiny_fragment_voxel_threshold": tiny_threshold,
        "max_locations": int(max_locations),
        "v5_label_map": dict(V5_LABELS),
        "validations": {
            "instance_shape_matches_seg": bool(instance.shape == seg.shape),
            "support_covers_all_instance_voxels": bool(((instance > 0) & ~support_mask).sum() == 0),
        },
    }
    audit_json = np.array(json.dumps(audit, sort_keys=True), dtype=object)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez appends `.npz` to filenames without that suffix; write to a
    # `.partial.npz` next to the final path so concurrent writers cannot read
    # a half-written sidecar.
    tmp_stem = out_path.with_suffix("")
    tmp_path = tmp_stem.with_name(tmp_stem.name + ".partial")
    tmp_npz = tmp_path.with_suffix(".partial.npz")
    if tmp_npz.exists():
        tmp_npz.unlink()
    np.savez(
        str(tmp_path),
        centers_zyx=centers_zyx,
        fragment_sizes=fragment_sizes,
        contact_locations=contact_locations,
        hard_negative_locations=hard_negative_locations,
        tiny_locations=tiny_locations,
        support_locations=support_locations,
        edge_locations=edge_locations,
        instance=instance,
        audit_json=audit_json,
    )
    os.replace(str(tmp_npz), str(out_path))
    return audit


def build_bicm_v5_instance_sidecars(ds_id: int = 537,
                                    force: bool = False,
                                    case_subset: list[str] | None = None,
                                    plan_configuration: str = "nnUNetPlans_3d_fullres",
                                    ) -> dict:
    """Build BICM V5 per-sample instance sidecars for the requested dataset.

    For each `PENGWIN_<cid>_<anatomy>.mha` in the raw `labelsTr` directory the
    function:
      1. Reads the raw V5 target MHA (the cropped anatomy ROI label, values 0..4).
      2. Reads the original case `label.mha` and rebuilds the anatomy-ROI
         instance ID array using the same bbox/pad_vox contract as
         `build_bicm_v5_dataset`.
      3. Resamples the instance ID ROI onto the preprocessed grid recorded in
         `nnUNetPlans_3d_fullres/<sample_id>.pkl` (nearest-neighbor).
      4. Combines the resampled instance map with the preprocessed V5 seg
         (`<sample_id>.npz['seg']`) to derive centers, sizes, and contact /
         tiny / support / hard-negative / edge location pools.
      5. Saves the sidecar `.npz` under
         `<preprocessed>/<plan_configuration>/bicm_v5_instance_targets/`.

    Returns a dict with `dataset_id`, `output_dir`, list of written `samples`,
    and `audit_path` pointing at the per-sample audit JSON.
    """
    cfg = DATASETS[ds_id]
    if cfg["kind"] != "bicm_v5":
        raise ValueError(f"Dataset{ds_id} is not a BICM V5 dataset")

    preprocessed_root = NN_PREP / cfg["name"]

    # Resolve the plan, then take the preprocessed-DATA directory from the plan's
    # `data_identifier` — never assume the data dir name equals the plan name.
    # nnUNet's ResEnc planners reuse the DEFAULT preprocessor, so the plan FILE
    # (e.g. nnUNetResEncUNetLPlans.json) and the preprocessed-data dir
    # (data_identifier, e.g. "nnUNetPlans_3d_fullres") have DIFFERENT names. The
    # old code hardcoded "nnUNetPlans.json" + used plan_configuration as the data
    # dir, which (a) missed the ResEncL plan's spacing and (b) on a ResEncL-only
    # tree (no nnUNetPlans.json) failed outright. plan_configuration names the
    # PLAN: "<plans_name>_<config>" (e.g. "nnUNetResEncUNetLPlans_3d_fullres").
    if "_" in plan_configuration:
        plans_name, plan_cfg_key = plan_configuration.split("_", 1)
    else:
        plans_name, plan_cfg_key = "nnUNetPlans", plan_configuration
    plans_path = preprocessed_root / f"{plans_name}.json"
    if not plans_path.exists():
        raise FileNotFoundError(f"Missing {plans_name}.json under {preprocessed_root}")
    plans = json.loads(plans_path.read_text())
    plan_cfg = plans.get("configurations", {}).get(plan_cfg_key)
    if plan_cfg is None:
        raise KeyError(
            f"Plan configuration {plan_cfg_key!r} not found in {plans_path}; "
            f"available: {sorted(plans.get('configurations', {}))}"
        )
    plan_target_spacing_zyx = tuple(float(v) for v in plan_cfg["spacing"])
    data_identifier = plan_cfg.get("data_identifier", plan_configuration)

    raw_labels_dir = NN_RAW / cfg["name"] / "labelsTr"
    preprocessed_dir = preprocessed_root / data_identifier
    output_dir = preprocessed_dir / BICM_V5_INSTANCE_SIDECAR_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_label_files = sorted(raw_labels_dir.glob("PENGWIN_*.mha"))
    if case_subset:
        wanted = {str(c).zfill(3) for c in case_subset}
        raw_label_files = [
            p for p in raw_label_files
            if p.name.split("_")[1] in wanted
        ]
    if not raw_label_files:
        raise FileNotFoundError(
            f"No raw V5 label MHA found under {raw_labels_dir} for subset={case_subset!r}"
        )

    max_locations = PENGWIN_BICM_V5_SIDECAR_MAX_LOCATIONS
    log.info(
        "[bicm_v5_sidecars] Ds%d %s — %d samples → %s (max_locations=%d)",
        ds_id, cfg["name"], len(raw_label_files), output_dir, max_locations,
    )

    audits: list[dict] = []
    written = 0
    skipped = 0
    for raw_lbl in raw_label_files:
        sample_id = raw_lbl.stem  # PENGWIN_003_LeftHip
        out_path = output_dir / f"{sample_id}.npz"
        if out_path.exists() and not force:
            skipped += 1
            continue
        cid_str = sample_id.split("_")[1]
        case_dir = find_case_dir(cid_str)
        if case_dir is None:
            raise FileNotFoundError(f"{sample_id}: source case {cid_str} not found under {DATA_RAW}")
        preprocessed_pkl = preprocessed_dir / f"{sample_id}.pkl"
        preprocessed_npz = preprocessed_dir / f"{sample_id}.npz"
        if not preprocessed_pkl.exists() or not preprocessed_npz.exists():
            raise FileNotFoundError(
                f"{sample_id}: missing preprocessed pkl/npz under {preprocessed_dir}; "
                "run `nnUNetv2_plan_and_preprocess` for the dataset first."
            )
        audit = _build_one_bicm_v5_instance_sidecar(
            sample_id=sample_id,
            raw_label_path=raw_lbl,
            source_case_dir=case_dir,
            preprocessed_pkl=preprocessed_pkl,
            preprocessed_npz=preprocessed_npz,
            out_path=out_path,
            max_locations=max_locations,
            plan_target_spacing_zyx=plan_target_spacing_zyx,
        )
        audits.append(audit)
        written += 1
        log.info("  [%d/%d] %s", written + skipped, len(raw_label_files), sample_id)

    audit_path = RESULT_REPORT / f"build_bicm_v5_instance_sidecars_ds{ds_id}_{RESULT_DATE}.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({
        "dataset_id": ds_id,
        "dataset_name": cfg["name"],
        "output_dir": str(output_dir),
        "plan_configuration": plan_configuration,
        "max_locations": max_locations,
        "force": bool(force),
        "case_subset": case_subset,
        "written": written,
        "skipped": skipped,
        "samples": audits,
    }, indent=2))
    log.info(
        "[bicm_v5_sidecars] wrote=%d skipped=%d audit=%s",
        written, skipped, audit_path,
    )
    return {
        "dataset_id": ds_id,
        "output_dir": str(output_dir),
        "written": written,
        "skipped": skipped,
        "samples": audits,
        "audit_path": str(audit_path),
    }


def generate_boundary_fragment_v3_target_sidecars(
    ds_id: int = 537,
    force: bool = False,
    case_subset: list[str] | None = None,
    plan_configuration: str = "nnUNetPlans_3d_fullres",
) -> dict:
    """Build BoundaryFragment V3 (5-class) target sidecars on the preprocessed ROI grid.

    For each preprocessed sample `PENGWIN_<cid>_<anatomy>.npz` the function:
      1. Loads the per-sample BICM V5 instance sidecar (`bicm_v5_instance_targets/<sample_id>.npz`)
         whose `instance` array is the global PENGWIN fragment-ID grid already
         resampled onto the preprocessed plan spacing.
      2. Calls `compute_boundary_fragment_target` with the plan target spacing
         to derive the dense V3 semantic target
         (0=background, 1=external_context, 2=fracture_barrier,
          3=fragment_shell, 4=fragment_core).
      3. Validates that every emitted label is within {0..4} and that every GT
         fragment receives at least one core voxel (same invariant the raw
         per-case worker enforces).
      4. Saves the sidecar `.npz` containing `target` (uint8, shape matches the
         preprocessed seg grid) under
         `<preprocessed>/<plan_configuration>/boundary_fragment_v3_targets/`.

    Returns a dict with `dataset_id`, `output_dir`, the list of written
    `samples`, and `audit_path` pointing at the per-sample audit JSON.
    """
    cfg = DATASETS[ds_id]
    if cfg["kind"] != "bicm_v5":
        raise ValueError(
            f"Dataset{ds_id} ({cfg['name']}) is not a BICM V5 dataset; cannot "
            "derive BoundaryFragment V3 targets from instance sidecars."
        )

    preprocessed_root = NN_PREP / cfg["name"]

    # Resolve plan, then take the data dir from the plan's data_identifier (see
    # build_bicm_v5_instance_sidecars: under ResEnc the plan-file name and the
    # data-dir name differ). plan_configuration names the PLAN "<plans_name>_<config>".
    if "_" in plan_configuration:
        plans_name, plan_cfg_key = plan_configuration.split("_", 1)
    else:
        plans_name, plan_cfg_key = "nnUNetPlans", plan_configuration
    plans_path = preprocessed_root / f"{plans_name}.json"
    if not plans_path.exists():
        raise FileNotFoundError(f"Missing {plans_name}.json under {preprocessed_root}")
    plans = json.loads(plans_path.read_text())
    plan_cfg = plans.get("configurations", {}).get(plan_cfg_key)
    if plan_cfg is None:
        raise KeyError(
            f"Plan configuration {plan_cfg_key!r} not found in {plans_path}; "
            f"available: {sorted(plans.get('configurations', {}))}"
        )
    plan_target_spacing_zyx = tuple(float(v) for v in plan_cfg["spacing"])
    data_identifier = plan_cfg.get("data_identifier", plan_configuration)

    preprocessed_dir = preprocessed_root / data_identifier
    instance_dir = preprocessed_dir / BICM_V5_INSTANCE_SIDECAR_DIR
    output_dir = preprocessed_dir / BOUNDARY_FRAGMENT_V3_TARGET_SIDECAR_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not instance_dir.is_dir():
        raise FileNotFoundError(
            f"BICM V5 instance sidecar directory missing: {instance_dir}. "
            "Run `build-instance-sidecars` before deriving BoundaryFragment V3 targets."
        )
    instance_files = sorted(instance_dir.glob("PENGWIN_*.npz"))
    if case_subset:
        wanted = {str(c).zfill(3) for c in case_subset}
        instance_files = [
            p for p in instance_files
            if p.name.split("_")[1] in wanted
        ]
    if not instance_files:
        raise FileNotFoundError(
            f"No BICM V5 instance sidecars found under {instance_dir} for subset={case_subset!r}"
        )

    params = BoundaryFragmentParams()
    log.info(
        "[bfv3_targets] Ds%d %s — %d samples → %s (spacing_zyx=%s)",
        ds_id, cfg["name"], len(instance_files), output_dir, plan_target_spacing_zyx,
    )

    audits: list[dict] = []
    written = 0
    skipped = 0
    for inst_path in instance_files:
        sample_id = inst_path.stem  # PENGWIN_003_LeftHip
        out_path = output_dir / f"{sample_id}.npz"
        if out_path.exists() and not force:
            skipped += 1
            continue

        preprocessed_npz = preprocessed_dir / f"{sample_id}.npz"
        if not preprocessed_npz.exists():
            raise FileNotFoundError(
                f"{sample_id}: missing preprocessed npz {preprocessed_npz}; "
                "run `nnUNetv2_plan_and_preprocess` for the dataset first."
            )

        with np.load(inst_path) as payload:
            if "instance" not in payload:
                raise KeyError(
                    f"{inst_path} does not contain `instance`; rebuild bicm_v5 sidecars."
                )
            instance = np.asarray(payload["instance"], dtype=np.uint16)

        # Sanity-check the instance grid shape against the preprocessed seg.
        with np.load(preprocessed_npz) as prep:
            seg = np.asarray(prep["seg"])
        if seg.ndim == 4:
            seg = seg[0]
        if instance.shape != seg.shape:
            raise RuntimeError(
                f"{sample_id}: instance sidecar shape {instance.shape} != "
                f"preprocessed seg shape {seg.shape}"
            )

        target, target_audit = compute_boundary_fragment_target(
            instance,
            spacing_zyx=plan_target_spacing_zyx,
            params=params,
        )
        target = target.astype(np.uint8, copy=False)
        if target.shape != instance.shape:
            raise RuntimeError(
                f"{sample_id}: BFV3 target shape {target.shape} != instance shape {instance.shape}"
            )
        invalid = set(int(v) for v in np.unique(target)) - set(BFV3_LABELS.values())
        if invalid:
            raise RuntimeError(
                f"{sample_id}: invalid BFV3 labels {sorted(invalid)} (allowed {sorted(BFV3_LABELS.values())})"
            )
        if (instance > 0).any() and not (target == BFV3_LABELS["fragment_core"]).any():
            raise RuntimeError(
                f"{sample_id}: no class-4 core voxels emitted for a non-empty instance ROI"
            )

        audit = {
            "sample_id": sample_id,
            "preprocessed_shape": list(int(v) for v in target.shape),
            "plan_target_spacing_zyx": list(float(v) for v in plan_target_spacing_zyx),
            "target_audit": target_audit,
        }
        audit_json = np.array(json.dumps(audit, sort_keys=True), dtype=object)

        # Atomic write via `.partial.npz` so concurrent readers never see a
        # half-written sidecar (mirrors the BICM V5 sidecar contract).
        tmp_stem = out_path.with_suffix("")
        tmp_path = tmp_stem.with_name(tmp_stem.name + ".partial")
        tmp_npz = tmp_path.with_suffix(".partial.npz")
        if tmp_npz.exists():
            tmp_npz.unlink()
        np.savez(
            str(tmp_path),
            target=target,
            audit_json=audit_json,
        )
        os.replace(str(tmp_npz), str(out_path))

        audits.append(audit)
        written += 1
        if written % 25 == 0 or (written + skipped) == len(instance_files):
            log.info("  [%d/%d] %s", written + skipped, len(instance_files), sample_id)

    audit_path = RESULT_REPORT / f"build_boundary_fragment_v3_target_sidecars_ds{ds_id}_{RESULT_DATE}.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({
        "dataset_id": ds_id,
        "dataset_name": cfg["name"],
        "output_dir": str(output_dir),
        "plan_configuration": plan_configuration,
        "plan_target_spacing_zyx": list(float(v) for v in plan_target_spacing_zyx),
        "force": bool(force),
        "case_subset": case_subset,
        "written": written,
        "skipped": skipped,
        "params": {
            "target_profile": params.target_profile,
            "contact_search_mm": float(params.contact_search_mm),
            "contact_ridge_mm": float(params.contact_ridge_mm),
            "contact_same_anatomy_only": bool(params.contact_same_anatomy_only),
            "external_band_mm": float(params.external_band_mm),
            "shell_mm": float(params.shell_mm),
            "tiny_core_radius_mm": float(params.tiny_core_radius_mm),
        },
        "samples": audits,
    }, indent=2))
    log.info(
        "[bfv3_targets] wrote=%d skipped=%d audit=%s",
        written, skipped, audit_path,
    )
    return {
        "dataset_id": ds_id,
        "output_dir": str(output_dir),
        "written": written,
        "skipped": skipped,
        "samples": audits,
        "audit_path": str(audit_path),
    }


















# =============================================================================
# Unpack (.npz → .npy after nnUNet preprocess)
# =============================================================================


# =============================================================================
# Statistics (analyze_dataset 통합)
# =============================================================================






# =============================================================================
# Sanity case inspection (absorbed from data/preprocess/inspect_cases.py)
# =============================================================================


# =============================================================================
# CLI
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    """Command-line dispatcher for preprocessing utilities."""
    parser = argparse.ArgumentParser(
        prog="preprocessing",
        description="PENGWIN 2026 Task 1 preprocessing utilities.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sidecar = sub.add_parser(
        "build-instance-sidecars",
        help="Build BICM V5 instance sidecars under <preprocessed>/<plan>/bicm_v5_instance_targets/.",
    )
    p_sidecar.add_argument("--dataset", type=int, default=538, help="nnUNet dataset ID (default 538).")
    p_sidecar.add_argument("--force", action="store_true", help="Overwrite existing sidecars.")
    p_sidecar.add_argument(
        "--case-subset",
        nargs="*",
        default=None,
        help="Optional list of source case IDs (zero-padded) to restrict the build.",
    )
    p_sidecar.add_argument(
        "--plan-configuration",
        default="nnUNetPlans_3d_fullres",
        help="Preprocessed plan configuration subfolder (default nnUNetPlans_3d_fullres).",
    )

    p_bfv3 = sub.add_parser(
        "build-boundary-fragment-v3-sidecars",
        help=(
            "Build BoundaryFragment V3 (5-class) target sidecars under "
            "<preprocessed>/<plan>/boundary_fragment_v3_targets/."
        ),
    )
    p_bfv3.add_argument("--dataset", type=int, default=538, help="nnUNet dataset ID (default 538).")
    p_bfv3.add_argument("--force", action="store_true", help="Overwrite existing sidecars.")
    p_bfv3.add_argument(
        "--case-subset",
        nargs="*",
        default=None,
        help="Optional list of source case IDs (zero-padded) to restrict the build.",
    )
    p_bfv3.add_argument(
        "--plan-configuration",
        default="nnUNetPlans_3d_fullres",
        help="Preprocessed plan configuration subfolder (default nnUNetPlans_3d_fullres).",
    )

    args = parser.parse_args(argv)

    if args.cmd == "build-instance-sidecars":
        result = build_bicm_v5_instance_sidecars(
            ds_id=int(args.dataset),
            force=bool(args.force),
            case_subset=args.case_subset,
            plan_configuration=args.plan_configuration,
        )
        print(json.dumps({
            "dataset_id": result["dataset_id"],
            "output_dir": result["output_dir"],
            "written": result["written"],
            "skipped": result["skipped"],
            "audit_path": result["audit_path"],
        }, indent=2))
        return 0
    if args.cmd == "build-boundary-fragment-v3-sidecars":
        result = generate_boundary_fragment_v3_target_sidecars(
            ds_id=int(args.dataset),
            force=bool(args.force),
            case_subset=args.case_subset,
            plan_configuration=args.plan_configuration,
        )
        print(json.dumps({
            "dataset_id": result["dataset_id"],
            "output_dir": result["output_dir"],
            "written": result["written"],
            "skipped": result["skipped"],
            "audit_path": result["audit_path"],
        }, indent=2))
        return 0
    parser.error(f"unknown command: {args.cmd!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

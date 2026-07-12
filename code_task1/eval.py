#!/usr/bin/env python3
"""Active split-anatomy evaluation helpers."""
from __future__ import annotations

import argparse
import heapq
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.special import expit

import sys
sys.path.insert(0, str(Path(__file__).parent))

from core import (
    ANATOMY_RANGES, ANATOMY_TO_INDEX, DATASETS, NN_PREP, NN_RAW, NN_RES,
    RESULT_REPORT, RESULT_VISUALIZE, configure_nnunet_env, is_femur, is_pelvic,
    ABBC_HARD_NEGATIVE_LABEL, BICM_V6_OUTPUT_CHANNELS, BICM_V8_OUTPUT_CHANNELS,
    BICM_V38_OUTPUT_CHANNELS, BICM_V68_OUTPUT_CHANNELS,
    BFV3_BINARY_BARRIER_OUTPUT_CHANNELS, BFV3_BINARY_BARRIER_SEED_OUTPUT_CHANNELS,
    BFV3_XYZ_AFFINITY_OUTPUT_CHANNELS, BFV3_AFFINITY13_SEED_OUTPUT_CHANNELS,
    BFV3_MUTEX13_SEED_OUTPUT_CHANNELS, BFV3_CENTER_FLOW_OUTPUT_CHANNELS,
    BFV3_NO_CONTACT_CENTER_FLOW_OUTPUT_CHANNELS, BFV3_SPATIAL_EMBEDDING_OUTPUT_CHANNELS,
    BFV3_NO_CONTACT_PAIRWISE_V273_OUTPUT_CHANNELS, BFV3_FRAGMENT_POSITION_V275_OUTPUT_CHANNELS,
    BFV3_SEPARATOR_GAP_V277_OUTPUT_CHANNELS,
    BFV3_SEPARATOR_ENERGY_V278_OUTPUT_CHANNELS,
    BFV3_SEPARATOR_SOFTMAX_V287_OUTPUT_CHANNELS,
    BFV3_ABBC_V288_OUTPUT_CHANNELS,
    BFV3_ABBC_SDF_V289_OUTPUT_CHANNELS,
    BFV3_ABBC_SDF_FDM_V290_OUTPUT_CHANNELS,
    BFV3_QUERY_MASK_V280_OUTPUT_CHANNELS,
    BFV3_QUERY_MASK_PN_V281_OUTPUT_CHANNELS,
    BFV3_FREE_EMBEDDING_V282_OUTPUT_CHANNELS,
    BFV3_GLOBAL_COORD_FREE_EMBEDDING_V283_OUTPUT_CHANNELS,
    FACTOR_INSTANCE_OUTPUT_CHANNELS, BOUNDARY_FRAGMENT_V3_TARGET_SIDECAR_DIR,
    _pelvic_same_anatomy_contact_mask,
)
from utils import (
    ORIENTATION_CONTRACT_VERSION, canonicalize_sitk, find_case_dir, inst_to_anat,
    prepare_lps_ct_for_nnunet,
    ABBC_BORDER_LABEL, ABBC_CORE_LABEL, ABBC_OFFICIAL_LABELS,
    compute_abbc_official_target, decode_official_abbc, get_contact_surface_regions,
    BFV3_LABELS, BFV3_SUPPORT_LABELS, BoundaryFragmentParams, binary_surface_metrics,
    compute_boundary_fragment_target, decode_boundary_fragment, instance_iouf,
    oracle_topology_diagnostics,
    BICMV5Params, V5_ANATOMY_RANGES, V5_ANATOMY_RANGES_WITH_FEMUR,
    PENGWIN_OFFICIAL_ANATOMY_RANGES,
    V5_LABELS, V5_SUPPORT_LABELS, V5_TARGET_PROFILES,
    anatomy_mask_from_instances, bbox_from_mask, compute_bicm_v5_target,
    decode_bicm_v5, decode_bicm_v5_seed_healed, label_distribution,
)
# Registry single-source helpers. Active decode/scoring uses the FULL (femur-
# inclusive) view; legacy per-version oracle proxies stay 3-anatomy via the
# explicit pelvic_only=True view (intent visible, not hidden behind a magic 150).
from utils import (
    NUM_ANATOMIES,
    MIN_INSTANCE_ID,
    PELVIC_MAX_INSTANCE_ID,
    valid_instance_mask,
    anatomy_start_ids,
)

configure_nnunet_env()


HARD5_PELVIC_CASES = ["003", "011", "017", "018", "025"]
HARD5_FEMUR_CASES = ["253", "261", "267", "268", "275"]
HARD10_CASES = HARD5_PELVIC_CASES + HARD5_FEMUR_CASES
SOTA_CASE_SETS = {
    "hard5_pelvic": HARD5_PELVIC_CASES,
    "hard5_femur": HARD5_FEMUR_CASES,
    "hard10": HARD10_CASES,
}



def _nnunet_results_root() -> Path:
    """Return the active nnU-Net results root for model-loading commands.

    [REPRO][Risk:High][Scope:evaluation_checkpoint]
    Short diagnostics often isolate checkpoints under a temporary
    `nnUNet_results` root. Prediction must honor that environment variable;
    otherwise eval silently reloads an older checkpoint from the default
    `/workspace/nnunet/results` tree and the reported metrics no longer match
    the just-finished experiment.
    """
    return Path(os.environ.get("nnUNet_results", str(NN_RES)))


def _binary_prf(pred: np.ndarray, target: np.ndarray, beta: float = 1.0) -> dict[str, float]:
    pred = pred.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    tp = float((pred & target).sum())
    fp = float((pred & ~target).sum())
    fn = float((~pred & target).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    beta2 = float(beta) ** 2
    f = (1.0 + beta2) * precision * recall / max(beta2 * precision + recall, 1e-8)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f": float(f),
        "pred_voxels": int(pred.sum()),
        "target_voxels": int(target.sum()),
        "pred_target_ratio": float(pred.sum() / max(1, int(target.sum()))),
    }


def run_boundary_fragment_eval(cases: list[str],
                               out_path: Path,
                               pred_root: Path | None = None,
                               oracle: bool = False,
                               decoder_profile: str = "hard_barrier",
                               target_profile: str = "v3_thin_ridge",
                               contact_band_mm: float = 3.0,
                               contact_search_mm: float = 3.0,
                               contact_ridge_mm: float = 1.0,
                               contact_same_anatomy_only: bool = False,
                               shell_mm: float = 8.0) -> dict:
    """Evaluate fixed BoundaryFragment V3 decoder on source-space labels.

    `--oracle` feeds the GT-derived V3 target to the decoder. This is the first
    gate: if oracle IoU-F is bad, training is forbidden because the target or
    decoder definition is wrong.
    """
    rows = []
    for case in [str(c).zfill(3) for c in cases]:
        cd = find_case_dir(case)
        if cd is None:
            raise FileNotFoundError(f"source case not found: {case}")
        img = canonicalize_sitk(sitk.ReadImage(str(cd / "image.mha")))
        lbl_img = canonicalize_sitk(sitk.ReadImage(str(cd / "label.mha")))
        gt = sitk.GetArrayFromImage(lbl_img)
        spacing_zyx = _spacing_zyx_from_image(img)
        target, target_audit = compute_boundary_fragment_target(
            gt,
            spacing_zyx=spacing_zyx,
            params=BoundaryFragmentParams(
                target_profile=target_profile,
                contact_band_mm=float(contact_band_mm),
                contact_search_mm=float(contact_search_mm),
                contact_ridge_mm=float(contact_ridge_mm),
                contact_same_anatomy_only=bool(contact_same_anatomy_only),
                shell_mm=float(shell_mm),
            ),
        )
        if oracle:
            pred_labels = target
            pred_source = "oracle_target"
        else:
            if pred_root is None:
                raise ValueError("--pred-root is required unless --oracle is set")
            pred_path = pred_root / f"PENGWIN_{case}.mha"
            if not pred_path.exists():
                raise FileNotFoundError(f"prediction not found: {pred_path}")
            pred_labels = sitk.GetArrayFromImage(canonicalize_sitk(sitk.ReadImage(str(pred_path)))).astype(np.uint8)
            pred_source = str(pred_path)
        decoded = decode_boundary_fragment(pred_labels, profile=decoder_profile)
        iouf = instance_iouf(decoded, gt)
        topology = oracle_topology_diagnostics(decoded, gt, target, iouf)
        support_metrics = binary_surface_metrics(
            np.isin(pred_labels, [2, 3, 4]),
            np.isin(target, [2, 3, 4]),
            spacing_zyx,
            nsd_tolerance_mm=2.0,
        )
        barrier_metrics = _binary_prf(pred_labels == 2, target == 2, beta=0.5)
        row = {
            "case": case,
            "prediction": pred_source,
            "oracle": bool(oracle),
            "iou_f": iouf,
            "support": support_metrics,
            "class2_barrier": barrier_metrics,
            "target_audit": target_audit,
            "topology_diagnostics": topology,
        }
        rows.append(row)
    summary = {
        "iou_f_mean": float(np.mean([r["iou_f"]["iou_f_mean"] for r in rows])) if rows else 0.0,
        "iou_f_min": float(np.min([r["iou_f"]["iou_f_mean"] for r in rows])) if rows else 0.0,
        "support_dice_mean": float(np.mean([r["support"]["dice"] for r in rows])) if rows else 0.0,
        "support_hd95_mean": float(np.mean([
            r["support"]["hd95_mm"] for r in rows if r["support"]["hd95_mm"] is not None
        ])) if any(r["support"]["hd95_mm"] is not None for r in rows) else None,
        "barrier_precision_mean": float(np.mean([r["class2_barrier"]["precision"] for r in rows])) if rows else 0.0,
        "barrier_recall_mean": float(np.mean([r["class2_barrier"]["recall"] for r in rows])) if rows else 0.0,
        "barrier_ratio_mean": float(np.mean([r["class2_barrier"]["pred_target_ratio"] for r in rows])) if rows else 0.0,
        "missing_seed_cases": [
            r["case"] for r in rows
            if r["target_audit"]["label_counts"].get(str(BFV3_LABELS["fragment_core"]), 0) == 0
        ],
        "decoder_profile": decoder_profile,
        "target_profile": target_profile,
        "contact_band_mm": float(contact_band_mm),
        "contact_search_mm": float(contact_search_mm),
        "contact_ridge_mm": float(contact_ridge_mm),
        "contact_same_anatomy_only": bool(contact_same_anatomy_only),
        "shell_mm": float(shell_mm),
    }
    result = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "boundary_fragment_v3_oracle" if oracle else "boundary_fragment_v3_prediction",
        "dataset": DATASETS[538]["name"],
        "decoder_profile": decoder_profile,
        "target_profile": target_profile,
        "contact_band_mm": float(contact_band_mm),
        "contact_search_mm": float(contact_search_mm),
        "contact_ridge_mm": float(contact_ridge_mm),
        "contact_same_anatomy_only": bool(contact_same_anatomy_only),
        "shell_mm": float(shell_mm),
        "cases": rows,
        "summary": summary,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_sanitize(result), indent=2, allow_nan=False))
    return result


def _json_sanitize(value: Any) -> Any:
    """Convert numpy/path values into JSON-safe plain Python values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def model_dir(ds_id: int,
              plans: str = "nnUNetResEncUNetLPlans",
              config: str = "3d_fullres",
              trainer: str | None = None) -> Path:
    """Return the nnU-Net result directory for an active dataset."""
    cfg = DATASETS[ds_id]
    trainer = trainer or cfg["trainer"]
    return NN_RES / cfg["name"] / f"{trainer}__{plans}__{config}"


def _spacing_zyx_from_image(img: sitk.Image) -> tuple[float, float, float]:
    sx, sy, sz = [float(v) for v in img.GetSpacing()]
    return (sz, sy, sx)


def case_dataset_id(case_id: str | int) -> int:
    """Return the active split dataset ID for a case."""
    cid = int(case_id)
    if is_pelvic(cid):
        return 532
    if is_femur(cid):
        return 533
    raise ValueError(f"cannot infer active dataset for case {case_id}")


def binary_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> dict:
    gm = gt_mask.astype(bool, copy=False)
    pm = pred_mask.astype(bool, copy=False)
    tp = int((gm & pm).sum())
    fp = int((~gm & pm).sum())
    fn = int((gm & ~pm).sum())
    denom_d = int(gm.sum() + pm.sum())
    denom_i = int((gm | pm).sum())
    return {
        "target_voxels": int(gm.sum()),
        "pred_voxels": int(pm.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": 1.0 if denom_d == 0 else float(2.0 * tp / denom_d),
        "iou": 1.0 if denom_i == 0 else float(tp / denom_i),
        "precision": 1.0 if tp + fp == 0 else float(tp / (tp + fp)),
        "recall": 1.0 if tp + fn == 0 else float(tp / (tp + fn)),
    }






def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """26-connected component labeling for decoder seeds/components."""
    from scipy import ndimage as ndi

    return ndi.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))




def _load_probability_npz(prob_path: Path) -> np.ndarray:
    """Load nnU-Net probability output as (C, Z, Y, X)."""
    if not prob_path.exists():
        raise FileNotFoundError(f"missing probability cache: {prob_path}")
    data = np.load(prob_path)
    for key in ("probabilities", "softmax"):
        if key in data:
            arr = data[key]
            break
    else:
        keys = list(data.keys())
        if not keys:
            raise RuntimeError(f"empty probability npz: {prob_path}")
        arr = data[keys[0]]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 4:
        raise RuntimeError(f"expected 4D probability array in {prob_path}, got {arr.shape}")
    return arr


def ndi_distance_indices(mask: np.ndarray,
                         spacing_zyx: tuple[float, float, float]) -> np.ndarray:
    """Return nearest-background indices for a binary mask via SciPy EDT."""
    from scipy import ndimage as ndi

    return ndi.distance_transform_edt(
        mask,
        sampling=spacing_zyx,
        return_distances=False,
        return_indices=True,
    )


def _watershed_assign_energy(markers: np.ndarray,
                             support: np.ndarray,
                             energy: np.ndarray) -> np.ndarray:
    """Assign support to markers using contact probability as watershed energy."""
    if not support.any() or not (markers > 0).any():
        return np.zeros_like(markers, dtype=np.uint16)
    try:
        from skimage.segmentation import watershed
        assigned = watershed(
            energy.astype(np.float32, copy=False),
            markers.astype(np.int32, copy=False),
            mask=support.astype(bool, copy=False),
            connectivity=1,
            watershed_line=False,
        )
        return assigned.astype(np.uint16, copy=False)
    except Exception:
        nearest_idx = ndi_distance_indices(markers == 0, (1.0, 1.0, 1.0))
        nearest = markers[tuple(nearest_idx)]
        out = np.zeros_like(markers, dtype=np.uint16)
        out[support] = nearest[support]
        return out


def _iou_for_ids(gt: np.ndarray, pred: np.ndarray,
                 gt_id: int, pred_id: int) -> float:
    gm = gt == gt_id
    pm = pred == pred_id
    union = int((gm | pm).sum())
    if union == 0:
        return 0.0
    return float((gm & pm).sum() / union)


def fragment_matching_metrics(gt_instances: np.ndarray,
                              pred_instances: np.ndarray,
                              dataset_ids: list[int]) -> dict:
    """Compute official-style best-IoU and Hungarian fragment diagnostics."""
    from scipy.optimize import linear_sum_assignment

    ranges = [DATASETS[ds_id]["global_label_range"] for ds_id in dataset_ids]
    ranges = [r for r in ranges if r is not None]
    gt_ids = [
        int(v) for v in np.unique(gt_instances)
        if any(lo <= int(v) <= hi for lo, hi in ranges)
    ]
    pred_ids = [
        int(v) for v in np.unique(pred_instances)
        if any(lo <= int(v) <= hi for lo, hi in ranges)
    ]
    gt_ids.sort()
    pred_ids.sort()
    matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
    for i, gt_id in enumerate(gt_ids):
        for j, pred_id in enumerate(pred_ids):
            matrix[i, j] = _iou_for_ids(gt_instances, pred_instances, gt_id, pred_id)

    best_rows = []
    best_values = []
    for i, gt_id in enumerate(gt_ids):
        if len(pred_ids) == 0:
            best_pred = None
            best_iou = 0.0
        else:
            j = int(np.argmax(matrix[i]))
            best_pred = pred_ids[j]
            best_iou = float(matrix[i, j])
        best_values.append(best_iou)
        best_rows.append({"gt_id": gt_id, "pred_id": best_pred, "iou": best_iou})
    official_unmatched = [row for row in best_rows if row["iou"] <= 0.0]

    hungarian_matches = []
    matched_gt = set()
    matched_pred = set()
    if matrix.size:
        row_ind, col_ind = linear_sum_assignment(-matrix)
        for i, j in zip(row_ind, col_ind):
            iou = float(matrix[i, j])
            if iou <= 0:
                continue
            matched_gt.add(gt_ids[int(i)])
            matched_pred.add(pred_ids[int(j)])
            hungarian_matches.append({
                "gt_id": gt_ids[int(i)],
                "pred_id": pred_ids[int(j)],
                "iou": iou,
            })
    hungarian_sum = sum(float(row["iou"]) for row in hungarian_matches)
    return {
        "gt_fragment_count": len(gt_ids),
        "pred_fragment_count": len(pred_ids),
        "official_style": {
            "mean_best_iou": float(np.mean(best_values)) if best_values else 1.0,
            "unmatched_count": len(official_unmatched),
            "unmatched_gt_ids": [int(row["gt_id"]) for row in official_unmatched],
            "best_matches": best_rows,
        },
        "hungarian": {
            "mean_iou_over_gt": float(hungarian_sum / len(gt_ids)) if gt_ids else 1.0,
            "matched_count": len(hungarian_matches),
            "unmatched_gt_ids": [int(v) for v in gt_ids if v not in matched_gt],
            "extra_pred_ids": [int(v) for v in pred_ids if v not in matched_pred],
            "matches": hungarian_matches,
        },
    }


def _surface_distance_metrics(pred: np.ndarray,
                              target: np.ndarray,
                              spacing_zyx: tuple[float, float, float]) -> dict[str, float | None]:
    """Binary Dice/HD95/ASSD in millimeters for Task1-aligned proxy scoring.

    Raises:
        ValueError: If pred/target shape differs or spacing is invalid.
    """
    if pred.shape != target.shape:
        raise ValueError(f"surface metric shape mismatch: pred={pred.shape} target={target.shape}")
    if len(spacing_zyx) != 3 or any(float(v) <= 0.0 for v in spacing_zyx):
        raise ValueError(f"invalid spacing_zyx for surface metrics: {spacing_zyx}")

    pred = pred.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    inter = int((pred & target).sum())
    denom = int(pred.sum() + target.sum())
    dice = 1.0 if denom == 0 else float(2.0 * inter / denom)
    if pred.shape == target.shape and bool(np.array_equal(pred, target)):
        return {"dice": float(dice), "hd95_mm": 0.0, "assd_mm": 0.0}
    if not pred.any() and not target.any():
        return {"dice": float(dice), "hd95_mm": 0.0, "assd_mm": 0.0}
    if not pred.any() or not target.any():
        return {"dice": float(dice), "hd95_mm": None, "assd_mm": None}
    crop_bbox = bbox_from_mask(pred | target, pad_vox=2)
    if crop_bbox is not None:
        pred = pred[crop_bbox]
        target = target[crop_bbox]

    pred_border = pred ^ ndi.binary_erosion(pred)
    target_border = target ^ ndi.binary_erosion(target)
    dist_to_target = ndi.distance_transform_edt(~target_border, sampling=spacing_zyx)
    dist_to_pred = ndi.distance_transform_edt(~pred_border, sampling=spacing_zyx)
    distances = np.concatenate([dist_to_target[pred_border], dist_to_pred[target_border]])
    if distances.size == 0:
        return {"dice": float(dice), "hd95_mm": 0.0, "assd_mm": 0.0}
    return {
        "dice": float(dice),
        "hd95_mm": float(np.percentile(distances, 95)),
        "assd_mm": float(np.mean(distances)),
    }


# =============================================================================
# PENGWIN 2026 Task 1 official-aligned proxy v2
# [METRIC][Risk:Blocker][Scope:official_metric_alignment_v2]
# Reason: the v1 proxy (compute_task1_official_proxy_metrics) is pelvic-only,
# global Hungarian, pooled fracture-dice, no GT-volume filter, and no 1 cm^3
# CC pruning. v2 follows the PENGWIN 2026 official Task 1 spec more closely
# while we still wait for the official evaluator script to be published.
# The v1 function is intentionally NOT modified so existing CLI eval JSONs
# (eval_task1_v288_prediction.json, etc.) stay byte-stable for regressions.
# =============================================================================
def _voxel_volume_mm3(spacing_zyx: tuple[float, float, float]) -> float:
    """Per-voxel volume in mm^3 from z/y/x spacing."""
    if len(spacing_zyx) != 3 or any(float(v) <= 0.0 for v in spacing_zyx):
        raise ValueError(f"invalid spacing_zyx: {spacing_zyx}")
    return float(spacing_zyx[0]) * float(spacing_zyx[1]) * float(spacing_zyx[2])


def _prune_small_components(instance_map: np.ndarray,
                            spacing_zyx: tuple[float, float, float],
                            min_mm3: float) -> np.ndarray:
    """instance별 CC 중 `min_mm3`보다 작은 것을 제거한다 (각 ID를 독립적으로 relabel).

    [METRIC][Risk:High][Scope:official_proxy_v2_cc_prune]
    PENGWIN 2026 Task 1 evaluator는 metric 계산 전에 약 1 cm^3 (1000 mm^3) 미만 connected
    component를 GT와 prediction 양쪽에서 모두 가지치기한다. 여기서는 map을 instance ID 단위로
    쪼개 ID별로 CC를 구한 뒤, voxel 수가 spacing을 고려한 threshold 미만인 CC를 0으로 만든다.
    다른 ID는 그대로 보존된다.
    """
    instance_map = np.asarray(instance_map)
    if float(min_mm3) <= 0.0:
        return instance_map.astype(instance_map.dtype, copy=True)
    vox_mm3 = _voxel_volume_mm3(spacing_zyx)
    voxel_threshold = int(max(1, np.ceil(float(min_mm3) / max(vox_mm3, 1e-9))))
    out = instance_map.copy()
    for inst_id in [int(v) for v in np.unique(instance_map) if int(v) > 0]:
        mask = instance_map == inst_id
        labeled, n_comp = ndi.label(mask)
        if n_comp == 0:
            continue
        sizes = np.bincount(labeled.ravel())
        # labeling 결과의 0번 index는 background를 의미한다.
        for cc_label in range(1, int(n_comp) + 1):
            if int(sizes[cc_label]) < voxel_threshold:
                out[labeled == cc_label] = 0
    return out


def _filter_gt_fragments_by_volume(gt_instances: np.ndarray,
                                   spacing_zyx: tuple[float, float, float],
                                   min_mm3: float = 500.0) -> tuple[np.ndarray, list[int]]:
    """전체 부피가 min_mm3 미만인 GT fragment ID를 제거한다.

    [METRIC][Risk:High][Scope:official_proxy_v2_gt_omit]
    PENGWIN 2026 공식 평가에서는 부피가 500 mm^3 미만인 fragment를 matching/metric 계산 전에
    GT 측에서 제외한다. 이 필터는 per-ID 단위로만 동작하며, per-CC 단위가 아님을 주의하자.
    같은 fragment 안의 작은 CC들은 _prune_small_components가 따로 처리한다.
    """
    gt_instances = np.asarray(gt_instances)
    if float(min_mm3) <= 0.0:
        return gt_instances.astype(gt_instances.dtype, copy=True), []
    vox_mm3 = _voxel_volume_mm3(spacing_zyx)
    voxel_threshold = int(max(1, np.ceil(float(min_mm3) / max(vox_mm3, 1e-9))))
    out = gt_instances.copy()
    dropped: list[int] = []
    for gt_id in [int(v) for v in np.unique(gt_instances) if int(v) > 0]:
        size = int((gt_instances == gt_id).sum())
        if size < voxel_threshold:
            out[gt_instances == gt_id] = 0
            dropped.append(int(gt_id))
    return out, dropped


def _present_anatomies(gt_instances: np.ndarray,
                       anatomy_ranges: dict[str, tuple[int, int]] | None = None) -> list[str]:
    """GT voxel이 한 개 이상 존재하는 anatomy key들의 리스트를 반환한다."""
    if anatomy_ranges is None:
        anatomy_ranges = PENGWIN_OFFICIAL_ANATOMY_RANGES
    present: list[str] = []
    for name, (lo, hi) in anatomy_ranges.items():
        if bool(((gt_instances >= int(lo)) & (gt_instances <= int(hi))).any()):
            present.append(str(name))
    return present


def _per_anatomy_argmax_match(gt_part: np.ndarray,
                              pred_part: np.ndarray,
                              iou_threshold: float) -> dict[str, Any]:
    """For each GT in this anatomy, pred = argmax_pred IoU(gt, pred) within same anatomy.

    [METRIC][Risk:High][Scope:official_proxy_v2_matching]
    The official evaluator does per-anatomy argmax IoU matching (each GT picks
    its best pred independently). This is NOT global Hungarian: two GTs may
    point to the same pred (which is what triggers a Merge Error in v2 counts).
    """
    gt_ids = sorted(int(v) for v in np.unique(gt_part) if int(v) > 0)
    pred_ids = sorted(int(v) for v in np.unique(pred_part) if int(v) > 0)
    if not gt_ids:
        return {
            "gt_ids": [],
            "pred_ids": pred_ids,
            "matches": [],
            "iou_matrix_shape": [0, len(pred_ids)],
        }
    if not pred_ids:
        return {
            "gt_ids": gt_ids,
            "pred_ids": [],
            "matches": [{"gt_id": int(g), "pred_id": None, "iou": 0.0} for g in gt_ids],
            "iou_matrix_shape": [len(gt_ids), 0],
        }
    gt_lookup = {gid: i for i, gid in enumerate(gt_ids)}
    pred_lookup = {pid: i for i, pid in enumerate(pred_ids)}
    matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    gt_pos = gt_part > 0
    pred_pos = pred_part > 0
    gt_sizes = np.zeros((len(gt_ids),), dtype=np.float64)
    pred_sizes = np.zeros((len(pred_ids),), dtype=np.float64)
    for gid, idx in gt_lookup.items():
        gt_sizes[idx] = float((gt_part == gid).sum())
    for pid, idx in pred_lookup.items():
        pred_sizes[idx] = float((pred_part == pid).sum())
    both = gt_pos & pred_pos
    if bool(both.any()):
        gt_flat = gt_part[both].astype(np.int64, copy=False)
        pred_flat = pred_part[both].astype(np.int64, copy=False)
        for gid, pid in zip(gt_flat.tolist(), pred_flat.tolist()):
            if gid in gt_lookup and pid in pred_lookup:
                matrix[gt_lookup[gid], pred_lookup[pid]] += 1.0
    unions = gt_sizes[:, None] + pred_sizes[None, :] - matrix
    iou = np.zeros_like(matrix)
    valid = unions > 0
    iou[valid] = matrix[valid] / unions[valid]
    matches: list[dict[str, Any]] = []
    for i, gid in enumerate(gt_ids):
        best_j = int(np.argmax(iou[i, :]))
        best_iou = float(iou[i, best_j])
        if best_iou >= float(iou_threshold):
            matches.append({"gt_id": int(gid), "pred_id": int(pred_ids[best_j]), "iou": best_iou})
        else:
            matches.append({"gt_id": int(gid), "pred_id": None, "iou": best_iou})
    return {
        "gt_ids": gt_ids,
        "pred_ids": pred_ids,
        "matches": matches,
        "iou_matrix_shape": [len(gt_ids), len(pred_ids)],
    }


def _per_fragment_hd95_assd(gt_mask: np.ndarray,
                            pred_mask: np.ndarray,
                            spacing_zyx: tuple[float, float, float]) -> dict[str, float | None]:
    """Per-fragment binary HD95/ASSD + Dice in mm.

    Returns NaN-safe nulls when either side is empty.
    """
    return _surface_distance_metrics(pred_mask.astype(bool, copy=False),
                                     gt_mask.astype(bool, copy=False),
                                     spacing_zyx)


def _mean_skip_none(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not clean:
        return None
    return float(np.mean(clean))


def compute_task1_official_aligned_v2_metrics(pred_instances: np.ndarray,
                                              gt_instances: np.ndarray,
                                              spacing_zyx: tuple[float, float, float],
                                              *,
                                              gt_fragment_min_mm3: float = 500.0,
                                              cc_prune_mm3: float = 1000.0,
                                              iou_match_threshold: float = 0.10,
                                              local_radius_mm: float = 20.0) -> dict[str, Any]:
    """PENGWIN 2026 Task 1 공식 스펙에 정렬한 proxy v2 metric을 계산한다.

    v1 (compute_task1_official_proxy_metrics) 대비 달라진 점:
    - Femur (151-200) anatomy range를 포함한다.
    - 500 mm^3 미만의 GT fragment를 필터링한다 (gt_fragment_min_mm3로 조정 가능).
    - matching 이전에 1 cm^3 미만의 predicted CC를 가지치기한다 (cc_prune_mm3로 조정 가능).
    - per-anatomy argmax matching을 사용한다 (global Hungarian을 대체).
    - HD95/ASSD를 fragment별로 계산한 뒤 part 안에서 평균하고, 다시 part끼리 평균한다.
    - Instance F1/Recall/Precision을 part별로 계산하고 cohort macro로 모은다.
    - Merge/Split은 COUNT만 내보낸다 (_rate field는 더 이상 없다).
    - cohort 집계를 위해 case_failed (bool)와 failure_mode를 함께 태그한다.

    [METRIC][Risk:Blocker][Scope:official_proxy_v2]
    PENGWIN 2026 Task 1 공식 evaluator script는 2026-05-29 기준으로 아직 공개되지 않았다.
    이 구현은 공개된 스펙을 따른 것이므로, 리더보드용 주장을 하려면 공식 스크립트가 공개된
    뒤 반드시 그것으로 다시 검증해야 한다.
    """
    if pred_instances.shape != gt_instances.shape:
        raise ValueError(
            f"Task1 official-aligned v2 shape mismatch: pred={pred_instances.shape} gt={gt_instances.shape}"
        )
    if len(spacing_zyx) != 3 or any(float(v) <= 0.0 for v in spacing_zyx):
        raise ValueError(f"invalid spacing_zyx for Task1 v2 metrics: {spacing_zyx}")
    if not (0.0 <= float(iou_match_threshold) <= 1.0):
        raise ValueError(f"iou_match_threshold must be in [0,1], got {iou_match_threshold}")

    anatomy_ranges = PENGWIN_OFFICIAL_ANATOMY_RANGES
    case_failed = False
    failure_mode: str | None = None
    vox_mm3 = _voxel_volume_mm3(spacing_zyx)

    # Step 1: 공식 anatomy ID range (1..200) 안으로 제한하고, 그 밖의 noise ID는 제거한다.
    valid_lo = min(int(lo) for lo, _ in anatomy_ranges.values())
    valid_hi = max(int(hi) for _, hi in anatomy_ranges.values())
    gt_full = np.where(
        (gt_instances >= valid_lo) & (gt_instances <= valid_hi),
        gt_instances,
        0,
    ).astype(np.int32, copy=False)
    pred_full = np.where(pred_instances > 0, pred_instances, 0).astype(np.int32, copy=False)

    # Step 2: GT 측에서 500 mm^3 미만 fragment를 떨궈낸다.
    gt_filtered, gt_dropped_ids = _filter_gt_fragments_by_volume(
        gt_full, spacing_zyx, min_mm3=float(gt_fragment_min_mm3)
    )

    # Step 3: GT와 prediction 양쪽에서 ID별로 1 cm^3 미만 CC를 가지치기한다.
    try:
        gt_pruned = _prune_small_components(gt_filtered, spacing_zyx, min_mm3=float(cc_prune_mm3))
        pred_pruned = _prune_small_components(pred_full, spacing_zyx, min_mm3=float(cc_prune_mm3))
    except Exception as exc:  # noqa: BLE001
        case_failed = True
        failure_mode = f"cc_prune_failed: {exc!r}"
        gt_pruned = gt_filtered
        pred_pruned = pred_full

    # Step 4: anatomy별로 metric을 계산한다.
    per_anatomy: dict[str, dict[str, Any]] = {}
    present = _present_anatomies(gt_pruned, anatomy_ranges)

    for anatomy, (lo, hi) in anatomy_ranges.items():
        in_anatomy_gt = (gt_pruned >= int(lo)) & (gt_pruned <= int(hi))
        in_anatomy_pred = (pred_pruned >= int(lo)) & (pred_pruned <= int(hi))
        gt_part = np.where(in_anatomy_gt, gt_pruned, 0).astype(np.int32, copy=False)
        pred_part = np.where(in_anatomy_pred, pred_pruned, 0).astype(np.int32, copy=False)

        gt_ids = sorted(int(v) for v in np.unique(gt_part) if int(v) > 0)
        pred_ids = sorted(int(v) for v in np.unique(pred_part) if int(v) > 0)
        n_gt = int(len(gt_ids))
        n_pred = int(len(pred_ids))

        # GT가 없는 anatomy: metric 계산은 건너뛰지만, presence 정보는 남겨둔다.
        if n_gt == 0:
            per_anatomy[str(anatomy)] = {
                "present": False,
                "gt_instance_count": 0,
                "pred_instance_count": n_pred,
                "fracture_iou_per_fragment": None,
                "fracture_dice_per_fragment": None,
                "local_dice_per_fragment_20mm": None,
                "hd95_mm_per_fragment": None,
                "assd_mm_per_fragment": None,
                "instance_recall": None,
                "instance_precision": None,
                "instance_f1": None,
                "merge_error_count": 0,
                "split_error_count": 0,
                "topology_consistency": None,
                "matches": [],
            }
            continue

        # anatomy별 argmax matching을 한다 (global Hungarian은 쓰지 않는다).
        match_result = _per_anatomy_argmax_match(gt_part, pred_part, iou_threshold=float(iou_match_threshold))
        matches = match_result["matches"]

        # fragment별 surface metric (Dice/HD95/ASSD)과 local Dice를 계산한다.
        dice_values: list[float | None] = []
        iou_values: list[float | None] = []   # fracture-wise IoU (IoU-F) — the OFFICIAL headline metric
        hd95_values: list[float | None] = []
        assd_values: list[float | None] = []
        local_dice_values: list[float | None] = []

        # part별 local-band (어떤 GT fragment 기준 20 mm 이내 영역)를 계산한다.
        gt_part_bin = gt_part > 0
        local_band = np.zeros_like(gt_part_bin, dtype=bool)
        if bool(gt_part_bin.any()):
            pad_vox = int(np.ceil(float(local_radius_mm) / max(min(float(v) for v in spacing_zyx), 1e-6))) + 2
            band_bbox = bbox_from_mask(gt_part_bin, pad_vox=pad_vox)
            if band_bbox is not None:
                dist_crop = ndi.distance_transform_edt(~gt_part_bin[band_bbox], sampling=spacing_zyx)
                local_band[band_bbox] = dist_crop <= float(local_radius_mm)

        for row in matches:
            gid = int(row["gt_id"])
            pid = row["pred_id"]
            gt_mask = (gt_part == gid)
            pred_mask = np.zeros_like(gt_mask, dtype=bool) if pid is None else (pred_part == int(pid))
            surf = _per_fragment_hd95_assd(gt_mask, pred_mask, spacing_zyx)
            dice_values.append(surf.get("dice"))
            # fracture-wise IoU (IoU-F): the PENGWIN official headline metric. Over-segmentation
            # (a GT fragment split across pred ids) drives the matched IoU down harder than Dice.
            _inter = float((gt_mask & pred_mask).sum())
            _union = float((gt_mask | pred_mask).sum())
            iou_values.append((_inter / _union) if _union > 0 else 0.0)
            hd95_values.append(surf.get("hd95_mm"))
            assd_values.append(surf.get("assd_mm"))
            local_gt = gt_mask & local_band
            local_pred = pred_mask & local_band
            local_dice_values.append(float(_binary_dice(local_pred, local_gt)))

        # part별로 Instance Recall/Precision/F1을 계산한다.
        matched_gt_ids = {int(m["gt_id"]) for m in matches if m["pred_id"] is not None}
        matched_pred_ids = {int(m["pred_id"]) for m in matches if m["pred_id"] is not None}
        tp = int(len(matched_gt_ids))
        fn = int(n_gt - tp)
        fp = int(n_pred - len(matched_pred_ids))
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        f1 = float(2.0 * precision * recall / max(precision + recall, 1e-8))

        # Merge: 여러 개의 매칭된 GT가 같은 pred id로 몰린 경우 -> 초과된 충돌 횟수를 센다.
        # Split: 하나의 GT에 여러 pred CC가 겹치는 경우
        # (매칭된 GT와 겹치는, 자기 자신은 매칭되지 않은 pred 하나당 split 1로 센다).
        pred_assignment_counts: dict[int, int] = {}
        for m in matches:
            if m["pred_id"] is not None:
                pred_assignment_counts[int(m["pred_id"])] = pred_assignment_counts.get(int(m["pred_id"]), 0) + 1
        merge_error_count = int(sum(max(0, c - 1) for c in pred_assignment_counts.values()))
        # split: GT마다, 이 anatomy 안에서 GT mask와 일정 이상 겹치는 pred CC 수
        # (CC prune 이후 기준)를 세고, 거기서 1 (매칭된 CC 자신)을 뺀다.
        split_error_count = 0
        for row in matches:
            gid = int(row["gt_id"])
            gt_mask = (gt_part == gid)
            if not bool(gt_mask.any()):
                continue
            overlapping_pred_ids = {int(v) for v in np.unique(pred_part[gt_mask]) if int(v) > 0}
            if len(overlapping_pred_ids) > 1:
                split_error_count += int(len(overlapping_pred_ids) - 1)

        # Topology consistency proxy: 1대1로 깔끔하게 매칭된 (충돌도 split도 없는)
        # pred를 가진 GT fragment의 비율을 잰다.
        good = 0
        for row in matches:
            pid = row["pred_id"]
            if pid is None:
                continue
            if pred_assignment_counts.get(int(pid), 0) != 1:
                continue
            gid = int(row["gt_id"])
            gt_mask = (gt_part == gid)
            overlapping_pred_ids = {int(v) for v in np.unique(pred_part[gt_mask]) if int(v) > 0}
            if len(overlapping_pred_ids) <= 1:
                good += 1
        topology_consistency = float(good / max(n_gt, 1))

        per_anatomy[str(anatomy)] = {
            "present": True,
            "gt_instance_count": n_gt,
            "pred_instance_count": n_pred,
            "fracture_iou_per_fragment": _mean_skip_none(iou_values),
            "fracture_dice_per_fragment": _mean_skip_none(dice_values),
            "local_dice_per_fragment_20mm": _mean_skip_none(local_dice_values),
            "hd95_mm_per_fragment": _mean_skip_none(hd95_values),
            "assd_mm_per_fragment": _mean_skip_none(assd_values),
            "instance_recall": float(recall),
            "instance_precision": float(precision),
            "instance_f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "merge_error_count": int(merge_error_count),
            "split_error_count": int(split_error_count),
            "topology_consistency": float(topology_consistency),
            "matches": matches,
        }

    # Cohort macro: 실제로 존재하는 anatomy들에 대해서만 평균을 낸다 (None은 건너뛴다).
    def _macro(key: str) -> float | None:
        values = [
            per_anatomy[name][key]
            for name in present
            if per_anatomy.get(name, {}).get(key) is not None
        ]
        if not values:
            return None
        return float(np.mean([float(v) for v in values]))

    def _macro_sum(key: str) -> int:
        return int(sum(int(per_anatomy[name].get(key, 0) or 0) for name in present))

    cohort_macro = {
        "present_anatomies": list(present),
        "fracture_iou": _macro("fracture_iou_per_fragment"),   # IoU-F — OFFICIAL headline
        "fracture_dice": _macro("fracture_dice_per_fragment"),
        "local_dice_20mm": _macro("local_dice_per_fragment_20mm"),
        "hd95_mm": _macro("hd95_mm_per_fragment"),
        "assd_mm": _macro("assd_mm_per_fragment"),
        "instance_recall": _macro("instance_recall"),
        "instance_precision": _macro("instance_precision"),
        "instance_f1": _macro("instance_f1"),
        "topology_consistency": _macro("topology_consistency"),
        "merge_error_count_total": _macro_sum("merge_error_count"),
        "split_error_count_total": _macro_sum("split_error_count"),
    }

    return {
        "metric_contract": "task1_official_aligned_proxy_v2",
        "official_mean_position_score": None,
        "official_score_reproducible": False,
        "anatomy_ranges": {str(k): [int(v[0]), int(v[1])] for k, v in anatomy_ranges.items()},
        "spacing_zyx_mm": [float(v) for v in spacing_zyx],
        "voxel_volume_mm3": float(vox_mm3),
        "gt_fragment_min_mm3": float(gt_fragment_min_mm3),
        "cc_prune_mm3": float(cc_prune_mm3),
        "iou_match_threshold": float(iou_match_threshold),
        "local_radius_mm": float(local_radius_mm),
        "gt_dropped_below_min_volume_ids": [int(v) for v in gt_dropped_ids],
        "per_anatomy": per_anatomy,
        "cohort_macro": cohort_macro,
        "case_failed": bool(case_failed),
        "failure_mode": failure_mode,
        "limitations": [
            "Official PENGWIN 2026 Task 1 evaluator script is not published as of 2026-05-29.",
            "Per-anatomy argmax matching follows the public spec; collision semantics for Merge are proxy.",
            "Topology Consistency is a 1-to-1 overlap-graph proxy; the official reference is not available.",
            "Femur 151-200 is included (V5_ANATOMY_RANGES_WITH_FEMUR).",
        ],
    }














TASK1_V288_OUTPUT_CHANNELS = 4















def _task1_v277_merge_small_components(labels: np.ndarray,
                                       *,
                                       min_component_voxels: int) -> np.ndarray:
    """separator로 인해 생긴 너무 작은 fragment를 가장 가까운 큰 component에 병합한다."""
    out = np.asarray(labels).astype(np.int32, copy=True)
    if int(min_component_voxels) <= 0:
        return out.astype(np.uint16, copy=False)
    ids = [int(v) for v in np.unique(out) if int(v) > 0]
    if not ids:
        return out.astype(np.uint16, copy=False)
    sizes = {component_id: int((out == component_id).sum()) for component_id in ids}
    small_ids = [component_id for component_id in ids if int(sizes[component_id]) < int(min_component_voxels)]
    large_ids = [component_id for component_id in ids if int(sizes[component_id]) >= int(min_component_voxels)]
    if not small_ids or not large_ids:
        return out.astype(np.uint16, copy=False)
    large_mask = np.isin(out, np.asarray(large_ids, dtype=np.int32))
    nearest_indices = ndi.distance_transform_edt(~large_mask, return_indices=True)[1]
    for component_id in small_ids:
        mask = out == int(component_id)
        if not mask.any():
            continue
        nearest_ids = out[tuple(axis_indices[mask] for axis_indices in nearest_indices)]
        nearest_ids = nearest_ids[nearest_ids > 0]
        if nearest_ids.size <= 0:
            continue
        values, counts = np.unique(nearest_ids, return_counts=True)
        out[mask] = int(values[int(np.argmax(counts))])
    compact = np.zeros_like(out, dtype=np.uint16)
    for new_id, old_id in enumerate([int(v) for v in np.unique(out) if int(v) > 0], start=1):
        compact[out == int(old_id)] = np.uint16(new_id)
    return compact




def _task1_v288_anatomy_size_ratio_merge(labels: np.ndarray,
                                         *,
                                         size_ratio_keep: float) -> np.ndarray:
    """`size_ratio_keep * 가장 큰 component 크기`보다 작은 component를 가장 가까운 큰 component에 병합한다.

    [DATA][Risk:High][Scope:v288_anatomical_refinement]
    PENGWIN 1st (MIC-DKFZ)의 "merging smaller fragments with closest anatomically
    correct instance" 후처리 단계다. 같은 anatomy ROI 안에서, 가장 큰 component 대비 크기
    비율이 `size_ratio_keep` 미만인 component를 가장 가까운 큰 component로 흡수시킨다.
    이렇게 하면 한 fragment가 여러 CC로 쪼개진 GT label 아티팩트(예: 003 Sacrum이 본래 GT 1이지만
    multi-CC로 저장된 경우)를 inference 단계에서도 동일 anatomy 안에서 자동으로 통합할 수 있다.
    """
    if float(size_ratio_keep) <= 0.0:
        return np.asarray(labels).astype(np.uint16, copy=False)
    out = np.asarray(labels).astype(np.int32, copy=True)
    sizes = {int(c): int((out == c).sum()) for c in np.unique(out) if int(c) > 0}
    if len(sizes) <= 1:
        return out.astype(np.uint16, copy=False)
    largest_size = max(sizes.values())
    threshold = max(1, int(float(size_ratio_keep) * float(largest_size)))
    small_ids = [c for c, sz in sizes.items() if sz < threshold]
    large_ids = [c for c, sz in sizes.items() if sz >= threshold]
    if not small_ids or not large_ids:
        return out.astype(np.uint16, copy=False)
    large_mask = np.isin(out, np.asarray(large_ids, dtype=np.int32))
    nearest_indices = ndi.distance_transform_edt(~large_mask, return_indices=True)[1]
    for c in small_ids:
        mask = out == int(c)
        if not mask.any():
            continue
        nearest_ids = out[tuple(axis_indices[mask] for axis_indices in nearest_indices)]
        nearest_ids = nearest_ids[nearest_ids > 0]
        if nearest_ids.size <= 0:
            continue
        vals, counts = np.unique(nearest_ids, return_counts=True)
        out[mask] = int(vals[int(np.argmax(counts))])
    compact = np.zeros_like(out, dtype=np.uint16)
    for new_id, old_id in enumerate([int(c) for c in np.unique(out) if int(c) > 0], start=1):
        compact[out == int(old_id)] = np.uint16(new_id)
    return compact


def decode_task1_v288_abbc(fields: np.ndarray,
                           *,
                           background_threshold: float = 0.50,
                           core_threshold: float = 0.50,
                           min_component_voxels: int = 100,
                           anatomy_size_ratio_keep: float = 0.0) -> tuple[np.ndarray, dict[str, Any]]:
    """V288 ABBC field를 디코드한다: core CC를 seed로 만든 뒤, boundary는 가장 가까운 seed로 watershed merge.

    [METRIC][Scope:task1_primary_gate]
    background와 core 모두 고정된 0.50 threshold를 사용한다. core CC를 watershed seed로 쓰고,
    boundary/border voxel은 가장 가까운 seed에 흡수된다. 너무 작은 component는 V277 helper로
    인접 component에 병합한다.
    """
    arr = np.asarray(fields)
    if arr.ndim != 4 or int(arr.shape[0]) != int(TASK1_V288_OUTPUT_CHANNELS):
        raise ValueError(f"V288 decoder expects [4,Z,Y,X], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("V288 decoder received NaN/Inf fields")
    background = arr[0] >= float(background_threshold)
    support = ~background
    core = (arr[3] >= float(core_threshold)) & support
    core_labels, n_core = ndi.label(core, structure=np.ones((3, 3, 3), dtype=bool))
    core_labels = core_labels.astype(np.int32, copy=False)
    if not support.any():
        decoded = np.zeros(arr.shape[1:], dtype=np.uint16)
        trace = {
            "decoder": "task1_v288_abbc_core_seed_watershed",
            "background_threshold": float(background_threshold),
            "core_threshold": float(core_threshold),
            "min_component_voxels": int(min_component_voxels),
            "support_voxels": 0,
            "core_voxels": 0,
            "core_seed_components_before_merge": 0,
            "components_before_small_merge": 0,
            "pred_fragment_count": 0,
            "contract_has_contact_probability": False,
            "contract_has_seed_channel": True,
            "contract_has_pairwise_graph": False,
        }
        return decoded, trace
    if int(n_core) <= 0:
        labels = np.where(support, 1, 0).astype(np.int32, copy=False)
    else:
        from skimage.segmentation import watershed as _ski_watershed
        priority = ndi.distance_transform_edt(core_labels == 0)
        labels = _ski_watershed(
            priority,
            markers=core_labels,
            mask=support,
        ).astype(np.int32, copy=False)
        fill_mask = support & (labels == 0)
        if fill_mask.any():
            nearest_indices = ndi.distance_transform_edt(labels == 0, return_indices=True)[1]
            labels[fill_mask] = labels[tuple(axis_indices[fill_mask] for axis_indices in nearest_indices)]
    before_merge_count = int(len([v for v in np.unique(labels) if int(v) > 0]))
    decoded = _task1_v277_merge_small_components(labels, min_component_voxels=int(min_component_voxels))
    components_after_size_merge = int(len([v for v in np.unique(decoded) if int(v) > 0]))
    if float(anatomy_size_ratio_keep) > 0.0:
        decoded = _task1_v288_anatomy_size_ratio_merge(
            decoded, size_ratio_keep=float(anatomy_size_ratio_keep),
        )
    trace = {
        "decoder": "task1_v288_abbc_core_seed_watershed",
        "background_threshold": float(background_threshold),
        "core_threshold": float(core_threshold),
        "min_component_voxels": int(min_component_voxels),
        "anatomy_size_ratio_keep": float(anatomy_size_ratio_keep),
        "support_voxels": int(support.sum()),
        "core_voxels": int(core.sum()),
        "core_seed_components_before_merge": int(n_core),
        "components_before_small_merge": int(before_merge_count),
        "components_after_small_merge": int(components_after_size_merge),
        "pred_fragment_count": int(len([v for v in np.unique(decoded) if int(v) > 0])),
        "contract_has_contact_probability": False,
        "contract_has_seed_channel": True,
        "contract_has_pairwise_graph": False,
    }
    return decoded.astype(np.uint16, copy=False), trace



























































































def task1_v288_probabilities_from_logits(raw: np.ndarray) -> np.ndarray:
    """Convert V288 ABBC softmax logits to 4-channel probability fields."""
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim != 4 or int(arr.shape[0]) != int(BFV3_ABBC_V288_OUTPUT_CHANNELS):
        raise ValueError(f"V288 logits expect [4,Z,Y,X], got {arr.shape}")
    shifted = arr - np.max(arr, axis=0, keepdims=True)
    exp = np.exp(shifted).astype(np.float32, copy=False)
    prob = exp / np.maximum(np.sum(exp, axis=0, keepdims=True), np.float32(1e-8))
    return prob.astype(np.float32, copy=False)


def _aggregate_task1_v2_samples(per_sample: list[dict[str, Any]]) -> dict[str, Any]:
    """Macro-average per-sample official-aligned v2 metrics across held-out val
    samples, with a per-anatomy breakdown. Each sample is one held-out
    (case, anatomy) ROI scored against that anatomy's GT fragments only."""
    mean_keys = ["fracture_iou", "fracture_dice", "local_dice_20mm", "hd95_mm", "assd_mm",
                 "instance_recall", "instance_precision", "instance_f1", "topology_consistency"]

    def _macro(rows: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"n_samples": len(rows)}
        for k in mean_keys:
            vals = [r["metrics"].get(k) for r in rows if r["metrics"].get(k) is not None]
            out[k] = float(np.mean(vals)) if vals else None
        for k in ("merge_error_count_total", "split_error_count_total"):
            out[k] = int(sum(int(r["metrics"].get(k, 0) or 0) for r in rows))
        return out

    by_anatomy = {a: _macro([r for r in per_sample if r["anatomy"] == a])
                  for a in sorted({r["anatomy"] for r in per_sample})}
    return {"overall": _macro(per_sample), "by_anatomy": by_anatomy}


def run_task1_abbc_eval(*,
                        dataset_id: int,
                        trainer: str,
                        out_path: Path,
                        samples: list[str] | None = None,
                        checkpoint: str = "checkpoint_best.pth",
                        fold: int = 0,
                        plans: str = "nnUNetResEncUNetLPlans",
                        config: str = "3d_fullres",
                        preprocessed_plans_dir: str = "nnUNetPlans_3d_fullres",
                        roi_pad_vox: int = 24,
                        background_threshold: float = 0.50,
                        core_threshold: float = 0.50,
                        min_component_voxels: int = 100,
                        anatomy_size_ratio_keep: float = 0.0,
                        decoder_support_ratio_cap: float = 6.0,
                        gt_fragment_min_mm3: float = 500.0,
                        cc_prune_mm3: float = 1000.0,
                        iou_match_threshold: float = 0.10) -> dict[str, Any]:
    """Per-sample held-out Stage-2 eval for a 4-class ABBC model, scored with the
    PENGWIN 2026 official-aligned v2 proxy.

    The Ds538 split is per-ROI — a case can have some anatomies in TRAIN and others
    in VAL — so a full-volume per-case eval would leak training ROIs. Here every
    held-out (case, anatomy) val sample is scored independently: its ROI is run
    through sliding-window inference, decoded with the core-seed watershed
    (decode_task1_v288_abbc), placed into a single-anatomy instance volume, and
    scored against that anatomy's GT fragments only (global ranges Sacrum 1-50,
    LeftHip 51-100, RightHip 101-150, Femur 151-200). `samples` defaults to the
    fold's held-out val list from splits_final.json; pass an explicit chunk for
    multi-process parallelism. See [[pengwin-task1-eval-metric]].
    """
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    dataset_name = DATASETS[int(dataset_id)]["name"]
    model_dir = _nnunet_results_root() / dataset_name / f"{trainer}__{plans}__{config}"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"trained model folder missing: {model_dir}")
    config_dir = NN_PREP / dataset_name / preprocessed_plans_dir
    if samples is None:
        splits = json.loads((NN_PREP / dataset_name / "splits_final.json").read_text())
        samples = list(splits[int(fold)]["val"])

    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=False, verbose_preprocessing=False, allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_dir), use_folds=(int(fold),), checkpoint_name=checkpoint)
    # [CRITICAL 2026-06-08] nnUNet 2.5.1 (pinned) does NOT load weights into predictor.network at
    # init — it stashes them in list_of_parameters and defers the load to perform_actual_prediction.
    # Our custom predict (_predict_custom_logits_from_preprocessed_data) reads predictor.network
    # DIRECTLY, so without this explicit load the eval runs a RANDOM network -> speckle -> ~0 score
    # (the exact GC 0-point bug; this eval previously "worked" only on 2.5.2 which loads at init).
    if getattr(predictor, "list_of_parameters", None):
        predictor.network.load_state_dict(predictor.list_of_parameters[0])
    try:
        _w0 = float(list(predictor.network.parameters())[0].detach().float().abs().sum().cpu())
        print(f"[abbc-eval] NET={type(predictor.network).__name__} w0sum={_w0:.3f} (loaded; ~random if <95)")
    except Exception:
        pass

    gt_cache: dict[str, tuple[np.ndarray, tuple]] = {}
    per_sample: list[dict[str, Any]] = []
    for sample in samples:
        stem = str(sample).replace("PENGWIN_", "")
        case, anatomy = stem.split("_", 1)
        case = case.zfill(3)
        if anatomy not in V5_ANATOMY_RANGES_WITH_FEMUR:
            raise ValueError(f"unknown anatomy in sample {sample!r}")
        lo, hi = V5_ANATOMY_RANGES_WITH_FEMUR[anatomy]

        if case not in gt_cache:
            cd = find_case_dir(case)
            if cd is None:
                raise FileNotFoundError(f"source case not found: {case}")
            lbl_img = canonicalize_sitk(sitk.ReadImage(str(cd / "label.mha")))
            gt_cache[case] = (
                sitk.GetArrayFromImage(lbl_img).astype(np.uint16, copy=False),
                tuple(float(v) for v in lbl_img.GetSpacing()[::-1]),
            )
        inst_full, spacing_zyx = gt_cache[case]

        # GT and prediction restricted to THIS anatomy only (the held-out unit) so
        # no other anatomy of the same patient (which may be a TRAIN ROI) is scored.
        gt_anat = np.where((inst_full >= int(lo)) & (inst_full <= int(hi)), inst_full, 0).astype(np.uint16, copy=False)
        pred_anat = np.zeros(inst_full.shape, dtype=np.uint16)

        anat_mask = anatomy_mask_from_instances(inst_full, anatomy)
        bbox = bbox_from_mask(anat_mask, pad_vox=int(roi_pad_vox))
        decoded = False
        if bbox is not None and (
            (config_dir / f"{sample}.npy").exists() or (config_dir / f"{sample}.npz").exists()
        ):
            gt_support = int(((inst_full[bbox] >= int(lo)) & (inst_full[bbox] <= int(hi))).sum())
            data = _load_preprocessed_sample(config_dir, sample)
            logits = _predict_custom_logits_from_preprocessed_data(
                predictor, torch.from_numpy(np.asarray(data).copy()).float(),
                output_channels=int(TASK1_V288_OUTPUT_CHANNELS))
            logits_np = logits.detach().cpu().numpy() if torch.is_tensor(logits) else np.asarray(logits)
            probs = task1_v288_probabilities_from_logits(logits_np)
            support_ratio = int((probs[0] < float(background_threshold)).sum()) / max(1, gt_support)
            if support_ratio <= float(decoder_support_ratio_cap):
                roi_shape = tuple(int(v) for v in inst_full[bbox].shape)
                if tuple(probs.shape[1:]) != roi_shape:
                    probs = _resize_channel_first_probabilities(probs, roi_shape)
                    probs = probs / np.maximum(np.sum(probs, axis=0, keepdims=True), np.float32(1e-8))
                # [2026-06-06] Anatomy-specific over-segmentation control. Sacrum is a single
                # dominant bone whose predicted core speckles into spurious islands -> over-split
                # (precision 0.49); an aggressive size-ratio + min-component merge fixes it
                # (F1 0.585->0.892). Femur/hips are genuine multi-fragment bones and keep the
                # defaults (a global aggressive merge collapsed their recall).
                _ratio = max(float(anatomy_size_ratio_keep), 0.10) if anatomy == "Sacrum" else float(anatomy_size_ratio_keep)
                _minvox = max(int(min_component_voxels), 250) if anatomy == "Sacrum" else int(min_component_voxels)
                decoded_roi, _trace = decode_task1_v288_abbc(
                    probs, background_threshold=background_threshold, core_threshold=core_threshold,
                    min_component_voxels=_minvox, anatomy_size_ratio_keep=_ratio)
                view = pred_anat[bbox]
                for local_id in sorted(int(v) for v in np.unique(decoded_roi) if int(v) > 0):
                    global_id = int(lo) + local_id - 1
                    if global_id > int(hi):
                        continue
                    view[decoded_roi == local_id] = np.uint16(global_id)
                pred_anat[bbox] = view
                decoded = True

        m = compute_task1_official_aligned_v2_metrics(
            pred_anat, gt_anat, spacing_zyx,
            gt_fragment_min_mm3=gt_fragment_min_mm3, cc_prune_mm3=cc_prune_mm3,
            iou_match_threshold=iou_match_threshold)
        per_sample.append({"sample": str(sample), "anatomy": anatomy,
                           "decoded": decoded, "metrics": m["cohort_macro"]})

    report = {
        "task": "task1_abbc_official_aligned_v2_eval_persample",
        "dataset": dataset_name, "trainer": trainer, "checkpoint": checkpoint, "fold": int(fold),
        "n_samples": len(per_sample),
        "decoder": {"background_threshold": background_threshold, "core_threshold": core_threshold,
                    "min_component_voxels": min_component_voxels,
                    "anatomy_size_ratio_keep": anatomy_size_ratio_keep,
                    "decoder_support_ratio_cap": decoder_support_ratio_cap},
        "metric": {"gt_fragment_min_mm3": gt_fragment_min_mm3, "cc_prune_mm3": cc_prune_mm3,
                   "iou_match_threshold": iou_match_threshold},
        "cohort": _aggregate_task1_v2_samples(per_sample),
        "samples": per_sample,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_sanitize(report), indent=2))
    return report








































def _binary_dice(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    denom = int(pred.sum()) + int(target.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * int((pred & target).sum()) / denom)








def _predict_custom_logits_from_preprocessed_data(predictor: Any,
                                                  data: "torch.Tensor",
                                                  output_channels: int) -> "torch.Tensor":
    """Run nnU-Net sliding-window inference for non-semantic custom heads.

    [QC][Invariant:custom_head_channels]
    nnU-Net's default exporter allocates output channels from `label_manager`.
    Dataset537 V6/V38 emit custom sigmoid/affinity heads despite raw labels
    being 0..4, so using the semantic exporter would silently recreate the
    failed V5 argmax path. This helper allocates the exact head count requested
    by the trainer.
    """
    import torch
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian
    from nnunetv2.utilities.helpers import dummy_context, empty_cache
    from tqdm import tqdm

    # [2026-06-05] Active path = 4-channel ABBC export only. This set previously listed ~22
    # dead V-version channel counts; the active call (run_task1_abbc_eval, output_channels=
    # TASK1_V288_OUTPUT_CHANNELS=4) passed ONLY because BICM_V6_OUTPUT_CHANNELS also == 4.
    # Pin it explicitly to the active ABBC contract so removing the dead constants cannot break it.
    # 4 = ABBC export (the only active head; the retired 9-channel affinity arm was removed 2026-06-10).
    allowed = {int(TASK1_V288_OUTPUT_CHANNELS)}
    if int(output_channels) not in allowed:
        raise ValueError(f"unsupported custom output_channels={output_channels}; allowed={sorted(allowed)}")

    def _internal_predict(data_pad: torch.Tensor, slicers: list[tuple], do_on_device: bool) -> torch.Tensor:
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = predictor.device if do_on_device else torch.device("cpu")
        try:
            empty_cache(predictor.device)
            data_work = data_pad.to(results_device)
            predicted_logits = torch.zeros(
                (int(output_channels), *data_work.shape[1:]),
                dtype=torch.half,
                device=results_device,
            )
            n_predictions = torch.zeros(data_work.shape[1:], dtype=torch.half, device=results_device)
            gaussian = compute_gaussian(
                tuple(predictor.configuration_manager.patch_size),
                sigma_scale=1.0 / 8,
                value_scaling_factor=10,
                device=results_device,
            ) if predictor.use_gaussian else 1

            for sl in tqdm(slicers, disable=not predictor.allow_tqdm):
                workon = data_work[sl][None].to(predictor.device)
                prediction = predictor._internal_maybe_mirror_and_predict(workon)[0].to(results_device)
                if int(prediction.shape[0]) != int(output_channels):
                    raise RuntimeError(
                        "Custom nnU-Net head mismatch during inference: "
                        f"expected {output_channels} channels, got {tuple(prediction.shape)}"
                    )
                if predictor.use_gaussian:
                    prediction *= gaussian
                predicted_logits[sl] += prediction
                n_predictions[sl[1:]] += gaussian
            predicted_logits /= n_predictions
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError("Encountered inf in custom-head predicted logits")
        except Exception as exc:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(predictor.device)
            empty_cache(results_device)
            raise exc
        return predicted_logits

    with torch.no_grad():
        if not torch.is_tensor(data):
            raise TypeError(f"expected torch.Tensor input, got {type(data).__name__}")
        if data.ndim != 4:
            raise ValueError(f"expected preprocessed data shape [C, Z, Y, X], got {tuple(data.shape)}")
        predictor.network = predictor.network.to(predictor.device)
        predictor.network.eval()
        empty_cache(predictor.device)
        with torch.autocast(predictor.device.type, enabled=True) if predictor.device.type == "cuda" else dummy_context():
            data_pad, slicer_revert_padding = pad_nd_image(
                data,
                predictor.configuration_manager.patch_size,
                "constant",
                {"value": 0},
                True,
                None,
            )
            slicers = predictor._internal_get_sliding_window_slicers(data_pad.shape[1:])
            if predictor.perform_everything_on_device and predictor.device.type != "cpu":
                try:
                    predicted_logits = _internal_predict(data_pad, slicers, True)
                except RuntimeError as exc:
                    print(
                        "Custom-head prediction on device failed; retrying with CPU result buffers. "
                        f"Original error: {exc}"
                    )
                    empty_cache(predictor.device)
                    predicted_logits = _internal_predict(data_pad, slicers, False)
            else:
                predicted_logits = _internal_predict(data_pad, slicers, bool(predictor.perform_everything_on_device))
            empty_cache(predictor.device)
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
    return predicted_logits


def _load_preprocessed_sample(config_dir: Path, sample: str) -> np.ndarray:
    npy_path = config_dir / f"{sample}.npy"
    npz_path = config_dir / f"{sample}.npz"
    if npy_path.exists():
        return np.load(npy_path, "r").astype(np.float32, copy=False)
    if npz_path.exists():
        return np.load(npz_path)["data"].astype(np.float32, copy=False)
    raise FileNotFoundError(f"preprocessed sample missing for {sample} under {config_dir}")






def _resize_channel_first_probabilities(probs: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Resize channel-first probability maps to the raw ROI target shape.

    [AUDIT][Risk:High][Scope:coordinate_space]
    V6 prediction runs on nnU-Net preprocessed ROI arrays, while IoU-F is
    reported in the raw cropped ROI coordinate system used by the instance GT.
    This helper is a coordinate-space restore, not threshold tuning: it applies
    fixed linear interpolation per probability channel and keeps the promotion
    threshold at 0.5.
    """
    if probs.ndim != 4:
        raise ValueError(f"expected probability maps [C,Z,Y,X], got {probs.shape}")
    target_shape = tuple(int(v) for v in target_shape)
    if tuple(probs.shape[1:]) == target_shape:
        return probs.astype(np.float32, copy=False)
    if any(int(v) <= 0 for v in probs.shape[1:]) or any(v <= 0 for v in target_shape):
        raise ValueError(f"invalid resize shape: source={probs.shape[1:]} target={target_shape}")
    factors = [1.0] + [float(t) / float(s) for s, t in zip(probs.shape[1:], target_shape)]
    resized = ndi.zoom(probs.astype(np.float32, copy=False), zoom=factors, order=1)
    if tuple(resized.shape[1:]) != target_shape:
        raise RuntimeError(f"probability resize failed: got {resized.shape}, expected C+{target_shape}")
    return np.clip(resized, 0.0, 1.0).astype(np.float32, copy=False)




















def _contact_instance_head_probs(output: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return support/core/contact/hard probabilities for legacy and V2 heads."""
    if output.shape[0] >= 16:
        morph = output[0:4].astype(np.float32, copy=False)
        morph = morph - np.max(morph, axis=0, keepdims=True)
        exp = np.exp(morph)
        probs = exp / np.maximum(np.sum(exp, axis=0, keepdims=True), 1e-8)
        support_prob = probs[1] + probs[2]
        contact_prob = probs[2]
        hard_prob = probs[3]
        core_prob = 1.0 / (1.0 + np.exp(-output[4]))
        return support_prob, core_prob, contact_prob, hard_prob
    support_prob = 1.0 / (1.0 + np.exp(-output[0]))
    core_prob = 1.0 / (1.0 + np.exp(-output[1]))
    contact_prob = 1.0 / (1.0 + np.exp(-output[2]))
    hard_prob = 1.0 / (1.0 + np.exp(-output[3]))
    return support_prob, core_prob, contact_prob, hard_prob


def decode_contact_instance_prediction(foundation_pred: np.ndarray,
                                       output: np.ndarray,
                                       profile: str = "contact_instance_v1") -> tuple[np.ndarray, dict]:
    """Decode Contact-Instance heads into pelvic fragment IDs."""
    from skimage.segmentation import watershed

    semantic = foundation_pred.astype(np.uint8, copy=False)
    pelvic = (semantic >= 1) & (semantic <= NUM_ANATOMIES)
    support_prob, core_prob, contact_prob, hard_prob = _contact_instance_head_probs(output)
    support = pelvic & (support_prob >= 0.45) & (hard_prob < 0.60)
    contact_wall = pelvic & (contact_prob >= 0.45)
    support = support & ~ndi.binary_dilation(contact_wall, structure=np.ones((3, 3, 3), dtype=bool))
    seed_mask = support & (core_prob >= 0.50)
    seed_cc, n_seed = _label_components(seed_mask)
    if n_seed == 0:
        return np.zeros_like(foundation_pred, dtype=np.uint16), {
            "decoder": profile,
            "seed_components": 0,
            "support_voxels": int(support.sum()),
            "contact_wall_voxels": int(contact_wall.sum()),
            "missing_seed_component": [{"component": 1, "voxels": int(support.sum())}] if support.any() else [],
        }
    energy = contact_prob * 10.0 + hard_prob * 4.0 + (1.0 - core_prob) * 0.25
    try:
        local = watershed(energy.astype(np.float32), seed_cc.astype(np.int32), mask=support, connectivity=1)
    except Exception:
        local = _watershed_assign_energy(seed_cc, support, energy)
    decoded = np.zeros_like(foundation_pred, dtype=np.uint16)
    next_by_anatomy = anatomy_start_ids()
    assigned = []
    removed = []
    for local_id in [int(v) for v in np.unique(local) if int(v) > 0]:
        mask = local == local_id
        if int(mask.sum()) == 0:
            continue
        anat_vals = semantic[mask]
        anat_vals = anat_vals[(anat_vals >= 1) & (anat_vals <= 3)]
        if len(anat_vals) == 0:
            removed.append({"local_id": local_id, "reason": "outside_pelvis", "voxels": int(mask.sum())})
            continue
        anatomy_id = int(np.bincount(anat_vals.astype(np.int64)).argmax())
        same = mask & (semantic == anatomy_id)
        max_core = float(core_prob[same].max()) if same.any() else 0.0
        mean_support = float(support_prob[same].mean()) if same.any() else 0.0
        if int(same.sum()) < 300 and max_core < 0.35 and mean_support < 0.45:
            removed.append({
                "local_id": local_id,
                "reason": "weak_small_island",
                "anatomy": anatomy_id,
                "voxels": int(same.sum()),
                "max_core": max_core,
                "mean_support": mean_support,
            })
            continue
        gid = next_by_anatomy[anatomy_id]
        next_by_anatomy[anatomy_id] += 1
        decoded[same] = gid
        assigned.append({
            "local_id": local_id,
            "global_id": gid,
            "anatomy": anatomy_id,
            "voxels": int(same.sum()),
            "max_core": max_core,
            "mean_contact": float(contact_prob[same].mean()) if same.any() else 0.0,
        })
    comp_cc, n_comp = _label_components(support)
    missing = []
    for comp_id in range(1, n_comp + 1):
        comp = comp_cc == comp_id
        if not (seed_cc[comp] > 0).any():
            missing.append({"component": int(comp_id), "voxels": int(comp.sum())})
    return decoded, {
        "decoder": profile,
        "decoder_profile": profile,
        "seed_components": int(n_seed),
        "support_voxels": int(support.sum()),
        "contact_wall_voxels": int(contact_wall.sum()),
        "hard_negative_voxels": int((pelvic & (hard_prob >= 0.5)).sum()),
        "missing_seed_component": missing,
        "assigned": assigned,
        "removed": removed,
    }











def write_eval_visualization(eval_json_path: Path,
                             out_text: Path | None = None,
                             out_json: Path | None = None) -> dict:
    """Write a compact text/JSON summary for a split-anatomy eval JSON."""
    data = json.loads(eval_json_path.read_text())
    summary = data.get("summary", {})
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(_json_sanitize(summary), indent=2))
    if out_text is not None:
        lines = ["# Split Anatomy Evaluation", ""]
        for ds_id, row in sorted(summary.items()):
            lines.append(
                f"- Ds{ds_id} `{row.get('dataset_name')}`: "
                f"n={row.get('n_cases')}, "
                f"Dice={row.get('mean_foreground_dice'):.4f}, "
                f"IoU={row.get('mean_foreground_iou'):.4f}"
            )
        out_text.parent.mkdir(parents=True, exist_ok=True)
        out_text.write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Active split-anatomy V2 evaluation")
    sub = parser.add_subparsers(dest="cmd", required=True)


    p_task1_abbc_eval = sub.add_parser(
        "task1-abbc-eval",
        help="clean end-to-end ABBC Stage-2 eval scored with the official-aligned v2 proxy (femur-aware)",
    )
    p_task1_abbc_eval.add_argument("--samples", nargs="+", default=None,
                                   help="explicit held-out val sample ids (PENGWIN_<case>_<anatomy>); "
                                        "default reads the fold's val list from splits_final.json")
    p_task1_abbc_eval.add_argument("--dataset-id", type=int, required=True)
    p_task1_abbc_eval.add_argument("--trainer", required=True)
    p_task1_abbc_eval.add_argument("--checkpoint", default="checkpoint_best.pth")
    p_task1_abbc_eval.add_argument("--fold", type=int, default=0)
    p_task1_abbc_eval.add_argument("--plans", default="nnUNetResEncUNetLPlans")
    p_task1_abbc_eval.add_argument("--config", default="3d_fullres")
    p_task1_abbc_eval.add_argument("--preprocessed-plans-dir", default="nnUNetPlans_3d_fullres")
    p_task1_abbc_eval.add_argument("--roi-pad-vox", type=int, default=24)
    p_task1_abbc_eval.add_argument("--background-threshold", type=float, default=0.5)
    p_task1_abbc_eval.add_argument("--core-threshold", type=float, default=0.5)
    p_task1_abbc_eval.add_argument("--min-component-voxels", type=int, default=100)
    p_task1_abbc_eval.add_argument("--anatomy-size-ratio-keep", type=float, default=0.0)
    p_task1_abbc_eval.add_argument("--decoder-support-ratio-cap", type=float, default=6.0)
    p_task1_abbc_eval.add_argument("--gt-fragment-min-mm3", type=float, default=500.0)
    p_task1_abbc_eval.add_argument("--cc-prune-mm3", type=float, default=1000.0)
    p_task1_abbc_eval.add_argument("--iou-match-threshold", type=float, default=0.10)
    p_task1_abbc_eval.add_argument("--out", default=str(RESULT_REPORT / "eval_task1_abbc_v2.json"))

    args = parser.parse_args()
    if args.cmd == "task1-abbc-eval":
        result = run_task1_abbc_eval(
            dataset_id=args.dataset_id,
            trainer=args.trainer,
            out_path=Path(args.out),
            samples=args.samples,
            checkpoint=args.checkpoint,
            fold=args.fold,
            plans=args.plans,
            config=args.config,
            preprocessed_plans_dir=args.preprocessed_plans_dir,
            roi_pad_vox=args.roi_pad_vox,
            background_threshold=args.background_threshold,
            core_threshold=args.core_threshold,
            min_component_voxels=args.min_component_voxels,
            anatomy_size_ratio_keep=args.anatomy_size_ratio_keep,
            decoder_support_ratio_cap=args.decoder_support_ratio_cap,
            gt_fragment_min_mm3=args.gt_fragment_min_mm3,
            cc_prune_mm3=args.cc_prune_mm3,
            iou_match_threshold=args.iou_match_threshold,
        )
        print(json.dumps(_json_sanitize({"cohort": result["cohort"]["overall"], "out": args.out}), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

"""Target-family router for PENGWIN Task 1 v2.0 inference.

The artifact is a small joblib payload:
    {"model": sklearn classifier, "feature_names": list[str], "labels": ["pelvic", "femur"]}

Only code is stored in git. The joblib file must be packaged into model.tar.gz,
typically at:
    /opt/ml/model/stage1_router/stage1_target_router_fold0.joblib
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi


def _orientation_code(img: sitk.Image) -> str:
    return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img.GetDirection())


def _canonicalize_sitk(img: sitk.Image, target: str = "LPS") -> sitk.Image:
    if _orientation_code(img) == target:
        return img
    return sitk.DICOMOrient(img, target)


def _robust_percentiles(
    arr: np.ndarray,
    qs: tuple[float, ...] = (0.5, 1, 5, 25, 50, 75, 95, 99, 99.5),
) -> list[float]:
    return [float(np.percentile(arr, q)) for q in qs]


def _sampled_bone_features(arr: np.ndarray, spacing_zyx: tuple[float, float, float]) -> dict[str, float]:
    shape = np.asarray(arr.shape, dtype=np.int64)
    stride = np.maximum(np.ceil(shape / 96).astype(int), 1)
    sampled = arr[:: stride[0], :: stride[1], :: stride[2]]
    bone = sampled > 200
    features: dict[str, float] = {
        "sample_stride_z": float(stride[0]),
        "sample_stride_y": float(stride[1]),
        "sample_stride_x": float(stride[2]),
        "bone_frac_sampled": float(bone.mean()),
    }
    if not bool(bone.any()):
        for name in (
            "bone_bbox_frac_z",
            "bone_bbox_frac_y",
            "bone_bbox_frac_x",
            "bone_centroid_frac_z",
            "bone_centroid_frac_y",
            "bone_centroid_frac_x",
            "bone_cc_count",
            "bone_largest_cc_frac",
            "bone_second_cc_frac",
            "bone_bbox_mm_z",
            "bone_bbox_mm_y",
            "bone_bbox_mm_x",
        ):
            features[name] = 0.0
        return features

    coords = np.argwhere(bone)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    bbox = hi - lo
    centroid = coords.mean(axis=0)
    sampled_shape = np.asarray(sampled.shape)
    labels, n_cc = ndi.label(bone, structure=np.ones((3, 3, 3), dtype=bool))
    sizes: list[float] = []
    if n_cc:
        sizes = [float(x) for x in ndi.sum(bone, labels, index=np.arange(1, n_cc + 1))]
        sizes.sort(reverse=True)
    features.update(
        {
            "bone_bbox_frac_z": float(bbox[0] / sampled_shape[0]),
            "bone_bbox_frac_y": float(bbox[1] / sampled_shape[1]),
            "bone_bbox_frac_x": float(bbox[2] / sampled_shape[2]),
            "bone_bbox_mm_z": float(bbox[0] * stride[0] * spacing_zyx[0]),
            "bone_bbox_mm_y": float(bbox[1] * stride[1] * spacing_zyx[1]),
            "bone_bbox_mm_x": float(bbox[2] * stride[2] * spacing_zyx[2]),
            "bone_centroid_frac_z": float(centroid[0] / max(sampled_shape[0] - 1, 1)),
            "bone_centroid_frac_y": float(centroid[1] / max(sampled_shape[1] - 1, 1)),
            "bone_centroid_frac_x": float(centroid[2] / max(sampled_shape[2] - 1, 1)),
            "bone_cc_count": float(n_cc),
            "bone_largest_cc_frac": float(sizes[0] / max(float(bone.sum()), 1.0)) if sizes else 0.0,
            "bone_second_cc_frac": float(sizes[1] / max(float(bone.sum()), 1.0)) if len(sizes) > 1 else 0.0,
        }
    )
    return features


def extract_image_features(image_path: str | Path) -> dict[str, float]:
    img = _canonicalize_sitk(sitk.ReadImage(str(image_path)), target="LPS")
    arr = sitk.GetArrayFromImage(img).astype(np.float32, copy=False)
    spacing_xyz = img.GetSpacing()
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    shape = np.asarray(arr.shape, dtype=np.float64)
    fov = shape * np.asarray(spacing_zyx, dtype=np.float64)
    p = _robust_percentiles(arr)
    features = {
        "shape_z": float(shape[0]),
        "shape_y": float(shape[1]),
        "shape_x": float(shape[2]),
        "spacing_z": float(spacing_zyx[0]),
        "spacing_y": float(spacing_zyx[1]),
        "spacing_x": float(spacing_zyx[2]),
        "fov_z": float(fov[0]),
        "fov_y": float(fov[1]),
        "fov_x": float(fov[2]),
        "fov_z_over_x": float(fov[0] / max(fov[2], 1e-6)),
        "fov_x_over_z": float(fov[2] / max(fov[0], 1e-6)),
        "fov_y_over_x": float(fov[1] / max(fov[2], 1e-6)),
        "hu_p00_5": p[0],
        "hu_p01": p[1],
        "hu_p05": p[2],
        "hu_p25": p[3],
        "hu_p50": p[4],
        "hu_p75": p[5],
        "hu_p95": p[6],
        "hu_p99": p[7],
        "hu_p99_5": p[8],
    }
    features.update(_sampled_bone_features(arr, spacing_zyx))
    return features


def load_router(router_path: str | Path) -> dict:
    path = Path(router_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"target-family router artifact missing: {path}. "
            "Package stage1_router/stage1_target_router_fold0.joblib in model.tar.gz "
            "or set PENGWIN_TARGET_ROUTER=0."
        )
    payload = joblib.load(path)
    for key in ("model", "feature_names", "labels"):
        if key not in payload:
            raise KeyError(f"router artifact {path} missing key {key!r}")
    return payload


def predict_family(image_path: str | Path, payload: dict) -> tuple[str, float]:
    feats = extract_image_features(image_path)
    feature_names = list(payload["feature_names"])
    missing = [name for name in feature_names if name not in feats]
    if missing:
        raise KeyError(f"router feature extractor missing features: {missing}")

    x = np.asarray([[float(feats[name]) for name in feature_names]], dtype=np.float32)
    model = payload["model"]
    labels = list(payload["labels"])
    pred = model.predict(x)[0]
    family = labels[int(pred)] if isinstance(pred, (np.integer, int)) else str(pred)
    if family not in {"pelvic", "femur"}:
        raise ValueError(f"router predicted unsupported family {family!r}")

    p_femur = float("nan")
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(x)[0], dtype=np.float64)
        classes = getattr(model, "classes_", np.arange(len(probs)))
        for i, cls in enumerate(classes):
            if str(cls) == "1" or str(cls).lower() == "femur":
                p_femur = float(probs[i])
                break
    return family, p_femur

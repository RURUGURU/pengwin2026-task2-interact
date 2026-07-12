#!/usr/bin/env python3
"""PENGWIN 2026 Task 1 result visualizer.

This module is intentionally result-artifact only:
  - reads completed GPU eval JSON files from code_task1/result
  - writes text/JSON tables and dashboards into code_task1/result
  - scans nnU-Net checkpoint files and writes a lightweight weight manifest

It does not create log files. If a command needs durable output, it should be a
JSON/text result artifact under code_task1/result, not a .log side channel.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

import sys
sys.path.insert(0, str(Path(__file__).parent))
from core import (
    ANATOMY_NAMES, ANATOMY_RANGES, DATASETS, DATA_RAW, NN_PREP,
    NN_RES, RESULT, RESULT_REPORT, RESULT_VISUALIZE, RESULT_WEIGHT, SOTA,
    CONTACT_INSTANCE_CORE_CH, CONTACT_INSTANCE_EDGE_BREAK_CHS,
    CONTACT_INSTANCE_OUTPUT_CHANNELS, configure_nnunet_env,
)
from eval import (
    SOTA_CASE_SETS, _contact_instance_head_probs, _json_sanitize, binary_metrics,
    case_dataset_id, decode_contact_instance_prediction, fragment_matching_metrics,
    run_boundary_fragment_eval, write_eval_visualization,
)
from utils import (
    ORIENTATION_CONTRACT_VERSION, canonicalize_sitk, find_case_dir, inst_to_anat,
    orientation_code,
    BoundaryFragmentParams, BFV3_LABELS, contact_barrier_distance,
    compute_boundary_fragment_target, decode_boundary_fragment, instance_iouf,
    oracle_topology_diagnostics,
)
# Registry single source: generic fragment loops use the FULL view (femur shown
# when present); the Ds537 per-anatomy ROI viz stays pelvic_only by design.
from utils import (
    MIN_INSTANCE_ID,
    MAX_INSTANCE_ID,
    anatomy_ranges_by_name,
)

configure_nnunet_env()


CHECKPOINT_NAMES = {
    "checkpoint_best.pth": "best",
    "checkpoint_final.pth": "final",
    "checkpoint_latest.pth": "latest",
    "checkpoint_swa.pth": "swa",
}


def _fmt(v, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    try:
        f = float(v)
        if np.isnan(f):
            return "n/a"
        return f"{f:.{digits}f}"
    except Exception:
        return str(v)


def _bar(value: float, width: int = 30) -> str:
    if value is None:
        return "|" + " " * width + "|"
    try:
        v = max(0.0, min(1.0, float(value)))
    except Exception:
        return "|" + " " * width + "|"
    n = int(round(v * width))
    return "|" + "#" * n + " " * (width - n) + "|"


def load_completed_eval(path: Path) -> dict | None:
    """Load one completed eval JSON, skipping legacy/non-eval JSONs."""
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    required = {"iou_f_mean", "iou_a_mean", "case_results"}
    if not required.issubset(data):
        return None
    return data


def scan_eval_jsons(result_dir: Path = RESULT) -> list[dict]:
    """Return completed eval records sorted by IoU-F descending.

    Recursive: scans `<experiment_tag>/eval_jsons/` subfolders too. Skips
    `archive_*` (frozen historic experiments).
    """
    rows = []
    paths = [p for p in result_dir.rglob("eval*.json")
             if not any(part.startswith("archive_") for part in p.parts)]
    for path in sorted(paths):
        data = load_completed_eval(path)
        if data is None:
            continue
        rows.append({
            "path": str(path),
            "file": path.name,
            "mode": data.get("mode"),
            "scope": data.get("case_scope", "custom"),
            "n_cases": data.get("n_cases"),
            "iou_a": data.get("iou_a_mean"),
            "iou_f": data.get("iou_f_mean"),
            "iou_f_highest": data.get("iou_f_highest_mean"),
            "hd95_f": data.get("hd95_f_mean"),
            "assd_f": data.get("assd_f_mean"),
            "unmatched": data.get("match_stats", {}).get("unmatched_gt_fragments"),
            "total_gt": data.get("match_stats", {}).get("total_gt_fragments"),
            "judgement": data.get("sota_gap_diagnosis", {}).get("judgement"),
            "primary_bottleneck": data.get("sota_gap_diagnosis", {}).get("primary_bottleneck"),
        })
    rows.sort(key=lambda r: float(r["iou_f"]) if r["iou_f"] is not None else -1.0,
              reverse=True)
    return rows


def scan_weight_manifest(results_root: Path = NN_RES,
                         derived_root: Path = RESULT_WEIGHT) -> list[dict]:
    """Scan checkpoint files without loading tensor payloads.

    Raw nnU-Net weights stay in nnunet/results; derived weights such as SWA
    outputs live under the active RESULT_WEIGHT directory. The manifest gives
    result-side traceability without loading tensor arrays, which would be slow
    and memory-heavy for 1GB+ checkpoints.
    """
    rows = []
    if not results_root.exists():
        raw_paths = []
    else:
        raw_paths = sorted(results_root.glob("Dataset*/*/fold_*/checkpoint*.pth"))
    for path in raw_paths:
        stat = path.stat()
        fold = path.parent.name
        run_dir = path.parent.parent
        dataset_dir = run_dir.parent
        rows.append({
            "dataset": dataset_dir.name,
            "run": run_dir.name,
            "fold": fold,
            "checkpoint": path.name,
            "kind": CHECKPOINT_NAMES.get(path.name, "other"),
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime(stat.st_mtime)),
            "path": str(path),
        })
    for path in sorted(derived_root.glob("*.pth")) if derived_root.exists() else []:
        stat = path.stat()
        rows.append({
            "dataset": "derived",
            "run": "active/weights",
            "fold": "n/a",
            "checkpoint": path.name,
            "kind": "derived",
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime(stat.st_mtime)),
            "path": str(path),
        })
    return rows


def write_weight_manifest(rows: list[dict], stamp: str,
                          result_dir: Path = RESULT) -> dict:
    """Write checkpoint/weight manifest as JSON and text."""
    result_dir.mkdir(parents=True, exist_ok=True)
    out_json = result_dir / f"weights_manifest_{stamp}.json"
    out_text = result_dir / f"weights_manifest_{stamp}.txt"

    by_dataset: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for row in rows:
        by_dataset[row["dataset"]] = by_dataset.get(row["dataset"], 0) + 1
        by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1

    payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "count": len(rows),
        "by_dataset": by_dataset,
        "by_kind": by_kind,
        "weights": rows,
    }
    out_json.write_text(json.dumps(_json_sanitize(payload), indent=2,
                                   allow_nan=False))

    lines = [
        f"# Weight Manifest {stamp}",
        "",
        f"- Checkpoints found: `{len(rows)}`.",
        "- Raw .pth files remain under `nnunet/results`; derived .pth files live under the active `RESULT_WEIGHT` directory.",
        "",
        "## By Dataset",
        "",
        "| Dataset | Checkpoints |",
        "|---|---:|",
    ]
    for dataset, count in sorted(by_dataset.items()):
        lines.append(f"| {dataset} | {count} |")
    lines.extend([
        "",
        "## Checkpoints",
        "",
        "| Dataset | Fold | Kind | Size MB | Modified UTC | Path |",
        "|---|---|---|---:|---|---|",
    ])
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['fold']} | {row['kind']} | "
            f"{row['size_mb']:.2f} | {row['modified_utc']} | `{row['path']}` |"
        )
    lines.append("")
    out_text.write_text("\n".join(lines))
    return {"json": str(out_json), "text": str(out_text), "count": len(rows)}


def write_result_summary(eval_rows: list[dict], stamp: str,
                         result_dir: Path = RESULT) -> dict:
    """Write a compact leaderboard-style summary for completed eval JSONs."""
    result_dir.mkdir(parents=True, exist_ok=True)
    out_json = result_dir / f"visual_summary_{stamp}.json"
    out_text = result_dir / f"visual_summary_{stamp}.txt"

    best = eval_rows[0] if eval_rows else None
    payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sota": SOTA,
        "best_by_iou_f": best,
        "evals": eval_rows,
    }
    out_json.write_text(json.dumps(_json_sanitize(payload), indent=2,
                                   allow_nan=False))

    lines = [
        f"# Result Summary {stamp}",
        "",
        f"- Completed eval JSONs: `{len(eval_rows)}`.",
        f"- SOTA IoU-F target: `{SOTA['iou_f']:.4f}`.",
    ]
    if best:
        lines.append(
            f"- Best current eval: `{best['file']}` IoU-F `{_fmt(best['iou_f'])}`, "
            f"gap `{_fmt(float(best['iou_f']) - SOTA['iou_f'])}`."
        )
    lines.extend([
        "",
        "## Eval Table",
        "",
        "| File | Mode | Scope | N | IoU-A | IoU-F | Bar | HD95 | ASSD | Unmatched | Judgement |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|---|",
    ])
    for row in eval_rows:
        unmatched = "n/a"
        if row.get("unmatched") is not None and row.get("total_gt") is not None:
            unmatched = f"{row['unmatched']}/{row['total_gt']}"
        lines.append(
            f"| {row['file']} | {row.get('mode')} | {row.get('scope')} | "
            f"{row.get('n_cases')} | {_fmt(row.get('iou_a'))} | {_fmt(row.get('iou_f'))} | "
            f"`{_bar(row.get('iou_f'))}` | {_fmt(row.get('hd95_f'), 2)} | "
            f"{_fmt(row.get('assd_f'), 2)} | {unmatched} | {row.get('judgement')} |"
        )
    lines.append("")
    out_text.write_text("\n".join(lines))
    return {"json": str(out_json), "text": str(out_text), "count": len(eval_rows)}


def _visual_suffix(eval_path: Path) -> str:
    """Stable dashboard suffix that cannot collide across eval variants."""
    return eval_path.stem.replace("eval_", "", 1)


def run_all(result_dir: Path = RESULT,
            artifact_dir: Path = RESULT_VISUALIZE) -> dict:
    """Generate current visual artifacts into the active V2 result folder."""
    stamp = time.strftime("%Y%m%d", time.gmtime())
    eval_rows = scan_eval_jsons(result_dir)
    summary = write_result_summary(eval_rows, stamp, artifact_dir)

    # Per-eval dashboards use eval.py's canonical SOTA visualization helper.
    dashboards = []
    for row in eval_rows:
        path = Path(row["path"])
        suffix = _visual_suffix(path)
        out_json = artifact_dir / f"visual_{suffix}_{stamp}.json"
        out_text = artifact_dir / f"visual_{suffix}_{stamp}.txt"
        write_eval_visualization(path, out_text=out_text, out_json=out_json)
        dashboards.append({"source": str(path), "json": str(out_json),
                           "text": str(out_text)})

    weights = write_weight_manifest(scan_weight_manifest(), stamp, artifact_dir)
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary,
        "dashboards": dashboards,
        "weights": weights,
    }
    manifest_path = artifact_dir / f"visual_manifest_{stamp}.json"
    manifest_path.write_text(json.dumps(_json_sanitize(manifest), indent=2,
                                        allow_nan=False))
    return manifest


# =============================================================================
# Case volume visualization: GT now, prediction later
# =============================================================================
ANATOMY_COLORS = {
    "Sacrum": (0.95, 0.55, 0.12),
    "LeftHip": (0.10, 0.55, 0.95),
    "RightHip": (0.10, 0.78, 0.35),
    "Femur": (0.95, 0.20, 0.35),
}


def _read_mha_array(path: Path, *, canonical_lps: bool = True) -> tuple[np.ndarray, object]:
    """Read an MHA volume and default visual QC to canonical LPS orientation."""
    import SimpleITK as sitk

    img = sitk.ReadImage(str(path))
    if canonical_lps:
        img = canonicalize_sitk(img)
    return sitk.GetArrayFromImage(img), img


def _sitk_geometry_matches(a: object, b: object) -> bool:
    """Return true when two SimpleITK images share the same physical grid."""
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), rtol=0, atol=1e-5)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), rtol=0, atol=1e-4)
        and np.allclose(a.GetDirection(), b.GetDirection(), rtol=0, atol=1e-5)
    )


def _resample_label_to_reference(label_img: object, reference_img: object) -> object:
    """Nearest-neighbor resample for visual overlays on the source CT grid."""
    import SimpleITK as sitk

    return sitk.Resample(
        label_img,
        reference_img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        label_img.GetPixelID(),
    )


def _ct_window(arr: np.ndarray, lo: float = -500.0, hi: float = 1500.0) -> np.ndarray:
    """Bone-friendly CT window normalized to [0, 1] for overlays."""
    x = np.clip(arr.astype(np.float32), lo, hi)
    return (x - lo) / max(1e-6, hi - lo)


def _label_rgb(label_proj: np.ndarray, mode: str) -> np.ndarray:
    """Map a projected label image to RGB colors."""
    rgb = np.zeros(label_proj.shape + (3,), dtype=np.float32)
    if mode == "anatomy":
        for idx, name in enumerate(ANATOMY_NAMES, start=1):
            rgb[label_proj == idx] = ANATOMY_COLORS[name]
        return rgb

    ids = [int(v) for v in np.unique(label_proj) if int(v) > 0]
    for label_id in ids:
        rng = np.random.default_rng(label_id * 2654435761 % (2**32))
        rgb[label_proj == label_id] = rng.uniform(0.20, 0.95, size=3)
    return rgb


def _project_label_for_mip(label: np.ndarray, axis: int, mode: str) -> np.ndarray:
    """Project labels for QC overlays without using label-id priority.

    For semantic anatomy, `label.max(axis=...)` is misleading because class 3
    would win over class 2 whenever both exist along the ray. Use the dominant
    non-background class along each projected pixel instead. Instance fragment
    MIPs keep the historical max-ID projection because the visual is a compact
    overview, not a per-pixel metric.
    """
    if mode != "anatomy":
        return label.max(axis=axis)
    class_ids = np.array(list(range(1, len(ANATOMY_NAMES) + 1)), dtype=np.int16)
    counts = np.stack([(label == int(class_id)).sum(axis=axis) for class_id in class_ids], axis=0)
    max_counts = counts.max(axis=0)
    projected = class_ids[counts.argmax(axis=0)].astype(np.int16, copy=False)
    projected[max_counts == 0] = 0
    return projected


def render_label_mips(image: np.ndarray, label: np.ndarray, out_path: Path,
                      *, title: str, mode: str = "anatomy") -> None:
    """Write 3-axis MIP overlay PNG for one GT or prediction volume.

    `mode="anatomy"` expects labels 0..4. `mode="fragment"` can receive raw
    PENGWIN instance IDs; colors are deterministic per ID so GT/pred IDs can be
    visually compared without assuming the numbers match semantically.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img_norm = _ct_window(image)
    axes_spec = [(0, "axial"), (1, "coronal"), (2, "sagittal")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=120)
    for ax, (axis, name) in zip(axes, axes_spec):
        img_mip = img_norm.max(axis=axis)
        lbl_mip = _project_label_for_mip(label, axis, mode)
        gray = np.stack([img_mip] * 3, axis=-1)
        color = _label_rgb(lbl_mip.astype(np.int32), mode)
        mask = lbl_mip > 0
        overlay = gray.copy()
        overlay[mask] = 0.48 * gray[mask] + 0.52 * color[mask]
        ax.imshow(np.clip(overlay, 0, 1))
        ax.set_title(f"{title} - {name}")
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)




def _write_obj(path: Path, verts_xyz: np.ndarray, faces: np.ndarray) -> None:
    """Write a simple Wavefront OBJ mesh for external 3D viewers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for x, y, z in verts_xyz:
            f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
        for a, b, c in faces:
            f.write(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")


def render_3d_surfaces(label_inst: np.ndarray, ref_img, out_png: Path,
                       mesh_dir: Path, *, title: str,
                       step_size: int = 2, max_faces_per_anatomy: int = 35_000,
                       labels_are_anatomy: bool = False) -> dict:
    """Render anatomy-colored 3D surfaces and export OBJ meshes.

    This keeps the 3D artifact independent from a notebook or GUI. The PNG is a
    quick QC view, while the OBJ files can be opened later in MeshLab, Blender,
    3D Slicer, or any standard mesh viewer.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage import measure

    spacing_xyz = tuple(float(v) for v in ref_img.GetSpacing())
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    anat = label_inst.astype(np.uint8, copy=False) if labels_are_anatomy else inst_to_anat(label_inst)
    fig = plt.figure(figsize=(9, 8), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    meshes = {}
    mins = []
    maxs = []

    for idx, name in enumerate(ANATOMY_NAMES, start=1):
        mask = anat == idx
        if not mask.any():
            continue
        try:
            verts_zyx, faces, _normals, _values = measure.marching_cubes(
                mask.astype(np.uint8),
                level=0.5,
                spacing=spacing_zyx,
                step_size=max(1, int(step_size)),
            )
        except ValueError:
            continue
        verts_xyz = verts_zyx[:, [2, 1, 0]]
        obj_path = mesh_dir / f"{name}.obj"
        _write_obj(obj_path, verts_xyz, faces)

        plot_faces = faces
        if len(plot_faces) > max_faces_per_anatomy:
            stride = int(np.ceil(len(plot_faces) / max_faces_per_anatomy))
            plot_faces = plot_faces[::stride]
        poly = Poly3DCollection(
            verts_xyz[plot_faces],
            alpha=0.42,
            facecolor=ANATOMY_COLORS[name],
            edgecolor="none",
        )
        ax.add_collection3d(poly)
        mins.append(verts_xyz.min(axis=0))
        maxs.append(verts_xyz.max(axis=0))
        meshes[name] = {
            "obj": str(obj_path),
            "vertices": int(len(verts_xyz)),
            "faces": int(len(faces)),
        }

    if mins:
        mn = np.min(np.vstack(mins), axis=0)
        mx = np.max(np.vstack(maxs), axis=0)
        center = (mn + mx) / 2.0
        radius = max(float((mx - mn).max()) / 2.0, 1.0)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_title(title)
    ax.set_xlabel("x mm")
    ax.set_ylabel("y mm")
    ax.set_zlabel("z mm")
    ax.view_init(elev=22, azim=-62)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return meshes


def semantic_pred_to_anatomy(pred: np.ndarray, ds_id: int) -> np.ndarray:
    """Map split semantic predictions to the shared anatomy color convention."""
    pred = pred.astype(np.uint8, copy=False)
    out = np.zeros_like(pred, dtype=np.uint8)
    if ds_id == 532:
        mask = (pred >= 1) & (pred <= 3)
        out[mask] = pred[mask]
    elif ds_id == 533:
        out[pred == 1] = 4
    else:
        raise ValueError(f"unsupported split semantic dataset id: {ds_id}")
    return out


def _is_split_semantic_prediction(pred: np.ndarray, ds_id: int) -> bool:
    """Return true when `pred` uses split semantic labels rather than instances."""
    max_label = int(pred.max()) if pred.size else 0
    if ds_id == 532:
        return max_label <= 3
    if ds_id == 533:
        return max_label <= 1
    return False


def visualize_case(case_id: str, pred_path: Path | None = None,
                   out_dir: Path | None = None,
                   step_size: int = 2) -> dict:
    """Create GT and optional prediction visual artifacts for one case."""
    import SimpleITK as sitk

    cid = str(case_id).zfill(3)
    case_dir = find_case_dir(cid)
    if case_dir is None:
        raise FileNotFoundError(f"case {cid} not found under {DATA_RAW}")
    out_dir = out_dir or (RESULT_VISUALIZE / "case_visuals" / cid)
    out_dir.mkdir(parents=True, exist_ok=True)

    image, img_ref = _read_mha_array(case_dir / "image.mha")
    gt, gt_ref = _read_mha_array(case_dir / "label.mha")
    gt_resampled = False
    if not _sitk_geometry_matches(gt_ref, img_ref):
        gt_ref = _resample_label_to_reference(gt_ref, img_ref)
        import SimpleITK as sitk
        gt = sitk.GetArrayFromImage(gt_ref)
        gt_resampled = True
    gt = gt.astype(np.int16, copy=False)

    artifacts: dict[str, object] = {
        "case": cid,
        "case_dir": str(case_dir),
        "out_dir": str(out_dir),
        "orientation_contract": ORIENTATION_CONTRACT_VERSION,
        "source_orientation": orientation_code(sitk.ReadImage(str(case_dir / "image.mha"))),
        "visual_orientation": "LPS",
        "visual_geometry": {
            "size_xyz": list(img_ref.GetSize()),
            "spacing_xyz": [float(v) for v in img_ref.GetSpacing()],
            "origin_xyz": [float(v) for v in img_ref.GetOrigin()],
            "direction": [float(v) for v in img_ref.GetDirection()],
        },
        "gt": {},
        "pred": None,
    }

    gt_anat = inst_to_anat(gt)
    gt_anat_png = out_dir / f"PENGWIN_{cid}_gt_anatomy_mip.png"
    gt_frag_png = out_dir / f"PENGWIN_{cid}_gt_fragments_mip.png"
    gt_3d_png = out_dir / f"PENGWIN_{cid}_gt_3d.png"
    render_label_mips(image, gt_anat, gt_anat_png, title=f"{cid} GT anatomy", mode="anatomy")
    render_label_mips(image, gt, gt_frag_png, title=f"{cid} GT fragments", mode="fragment")
    gt_meshes = render_3d_surfaces(
        gt, gt_ref, gt_3d_png, out_dir / "gt_meshes",
        title=f"PENGWIN {cid} GT 3D", step_size=step_size)
    artifacts["gt"] = {
        "anatomy_mip_png": str(gt_anat_png),
        "fragment_mip_png": str(gt_frag_png),
        "fragment_mip_note": None,
        "geometry_resampled_to_ct": gt_resampled,
        "surface_png": str(gt_3d_png),
        "meshes": gt_meshes,
    }

    if pred_path is not None:
        pred, pred_ref = _read_mha_array(pred_path)
        pred_resampled = False
        if not _sitk_geometry_matches(pred_ref, img_ref):
            pred_ref = _resample_label_to_reference(pred_ref, img_ref)
            pred = sitk.GetArrayFromImage(pred_ref)
            pred_resampled = True
        pred = pred.astype(np.int16, copy=False)
        pred_ds = case_dataset_id(cid)
        pred_is_semantic = _is_split_semantic_prediction(pred, pred_ds)
        pred_anat = semantic_pred_to_anatomy(pred, pred_ds) if pred_is_semantic else inst_to_anat(pred)
        pred_anat_png = out_dir / f"PENGWIN_{cid}_pred_anatomy_mip.png"
        pred_frag_png = out_dir / f"PENGWIN_{cid}_pred_fragments_mip.png"
        pred_3d_png = out_dir / f"PENGWIN_{cid}_pred_3d.png"
        render_label_mips(image, pred_anat, pred_anat_png, title=f"{cid} Pred anatomy", mode="anatomy")
        # Split anatomy datasets output semantic labels, not fragment instance IDs.
        # Rendering those semantic values with the fragment palette makes, for
        # example, class-2 LeftHip look like GT fragment ID 2. Keep the legacy
        # filename for downstream manifests, but color semantic predictions by
        # anatomy so GT/pred visual comparisons do not mix label contracts.
        pred_fragment_note = None
        if pred_is_semantic:
            gt_instance_frag_png = out_dir / f"PENGWIN_{cid}_gt_instance_fragments_mip.png"
            render_label_mips(
                image,
                gt,
                gt_instance_frag_png,
                title=f"{cid} GT instance fragments",
                mode="fragment",
            )
            render_label_mips(
                image,
                gt_anat,
                gt_frag_png,
                title=f"{cid} GT semantic anatomy",
                mode="anatomy",
            )
            artifacts["gt"]["instance_fragment_mip_png"] = str(gt_instance_frag_png)
            artifacts["gt"]["fragment_mip_note"] = (
                "Prediction is split semantic anatomy; this legacy GT fragment "
                "PNG is rendered with anatomy colors for direct comparison. "
                "Raw GT instance colors are preserved in gt_instance_fragments_mip.png."
            )
            render_label_mips(
                image,
                pred_anat,
                pred_frag_png,
                title=f"{cid} Pred semantic anatomy",
                mode="anatomy",
            )
            pred_fragment_note = (
                "Prediction is split semantic anatomy, not fragment IDs; "
                "this legacy fragment PNG is rendered with anatomy colors."
            )
        else:
            render_label_mips(image, pred, pred_frag_png, title=f"{cid} Pred fragments", mode="fragment")
        pred_meshes = render_3d_surfaces(
            pred_anat, pred_ref, pred_3d_png, out_dir / "pred_meshes",
            title=f"PENGWIN {cid} Pred 3D", step_size=step_size,
            labels_are_anatomy=True)
        artifacts["pred"] = {
            "source": str(pred_path),
            "dataset_id": pred_ds,
            "prediction_contract": "split_semantic" if pred_is_semantic else "instance",
            "source_orientation": orientation_code(sitk.ReadImage(str(pred_path))),
            "visual_orientation": "LPS",
            "orientation_contract": ORIENTATION_CONTRACT_VERSION,
            "geometry_resampled_to_ct": pred_resampled,
            "anatomy_mip_png": str(pred_anat_png),
            "fragment_mip_png": str(pred_frag_png),
            "fragment_mip_note": pred_fragment_note,
            "surface_png": str(pred_3d_png),
            "meshes": pred_meshes,
        }

    manifest_path = out_dir / f"PENGWIN_{cid}_visual_manifest.json"
    manifest_path.write_text(json.dumps(_json_sanitize(artifacts), indent=2, allow_nan=False))
    artifacts["manifest"] = str(manifest_path)
    return artifacts


def fragment_stats(label_inst: np.ndarray) -> dict:
    """Summarize GT/pred fragment IDs for visual case review."""
    rows = []
    for label_id in sorted(int(v) for v in np.unique(label_inst) if int(v) > 0):
        mask = label_inst == label_id
        coords = np.argwhere(mask)
        anatomy = "Unknown"
        for name, lo, hi in ANATOMY_RANGES:
            if lo <= label_id <= hi:
                anatomy = name
                break
        bbox = None
        if len(coords):
            mn = coords.min(axis=0)
            mx = coords.max(axis=0) + 1
            bbox = {
                "z": [int(mn[0]), int(mx[0])],
                "y": [int(mn[1]), int(mx[1])],
                "x": [int(mn[2]), int(mx[2])],
            }
        rows.append({
            "label_id": label_id,
            "anatomy": anatomy,
            "voxels": int(mask.sum()),
            "bbox_zyx": bbox,
        })
    by_anatomy = {name: 0 for name in ANATOMY_NAMES}
    for row in rows:
        if row["anatomy"] in by_anatomy:
            by_anatomy[row["anatomy"]] += 1
    return {
        "n_fragments": len(rows),
        "fragments_by_anatomy": by_anatomy,
        "fragments": rows,
    }


def export_hard_view(case_set: str = "hard10",
                     out_dir: Path | None = None,
                     step_size: int = 2) -> dict:
    """Create an analysis-ready GT hard-case folder tree."""
    if case_set not in SOTA_CASE_SETS:
        raise ValueError(f"unknown case_set={case_set!r}; choose from {sorted(SOTA_CASE_SETS)}")
    out_dir = out_dir or (RESULT_VISUALIZE / "hard_view" / case_set)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "case_set": case_set,
        "out_dir": str(out_dir),
        "cases": [],
    }
    for rank, cid in enumerate(SOTA_CASE_SETS[case_set], start=1):
        cid = str(cid).zfill(3)
        case_dir = find_case_dir(cid)
        if case_dir is None:
            raise FileNotFoundError(f"case {cid} not found under {DATA_RAW}")
        dst = out_dir / f"hard{rank:02d}_id_{cid}"
        image_dir = dst / "image"
        segment_dir = dst / "segment"
        visual_dir = dst / "visual"
        image_dir.mkdir(parents=True, exist_ok=True)
        segment_dir.mkdir(parents=True, exist_ok=True)
        visual_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(case_dir / "image.mha", image_dir / "image.mha")
        shutil.copy2(case_dir / "label.mha", segment_dir / "label.mha")

        image, _img_ref = _read_mha_array(case_dir / "image.mha")
        gt, gt_ref = _read_mha_array(case_dir / "label.mha")
        gt = gt.astype(np.int16, copy=False)
        gt_anat = inst_to_anat(gt)
        anat_png = visual_dir / "gt_anatomy_mip.png"
        frag_png = visual_dir / "gt_fragments_mip.png"
        surf_png = visual_dir / "gt_3d_surface.png"
        render_label_mips(image, gt_anat, anat_png, title=f"{cid} GT anatomy", mode="anatomy")
        render_label_mips(image, gt, frag_png, title=f"{cid} GT fragments", mode="fragment")
        meshes = render_3d_surfaces(
            gt, gt_ref, surf_png, visual_dir / "gt_meshes",
            title=f"PENGWIN {cid} GT 3D", step_size=step_size)
        stats = fragment_stats(gt)
        stats_path = visual_dir / "gt_fragment_stats.json"
        stats_path.write_text(json.dumps(_json_sanitize(stats), indent=2, allow_nan=False))
        row = {
            "rank": rank,
            "case": cid,
            "folder": str(dst),
            "image": str(image_dir / "image.mha"),
            "segment": str(segment_dir / "label.mha"),
            "visual": {
                "gt_anatomy_mip": str(anat_png),
                "gt_fragments_mip": str(frag_png),
                "gt_3d_surface": str(surf_png),
                "gt_meshes": meshes,
                "gt_fragment_stats": str(stats_path),
            },
            "stats": {
                "n_fragments": stats["n_fragments"],
                "fragments_by_anatomy": stats["fragments_by_anatomy"],
            },
        }
        manifest["cases"].append(row)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(_json_sanitize(manifest), indent=2, allow_nan=False))
    manifest["manifest"] = str(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", nargs="?", default="all",
                        choices=[
                            "all", "summary", "weights", "eval", "case", "hard-view",
                        ],
                        help="artifact group to generate")
    parser.add_argument("--eval-json", help="required for cmd=eval")
    parser.add_argument("--result-dir", default=str(RESULT),
                        help="root to scan for existing eval JSON files")
    parser.add_argument("--artifact-dir", default=str(RESULT_VISUALIZE),
                        help="active output folder for generated visual artifacts")
    parser.add_argument("--case", help="case ID for cmd=case, for example 025")
    parser.add_argument("--case-set", default="hard10",
                        choices=sorted(SOTA_CASE_SETS),
                        help="case set for cmd=hard-view")
    parser.add_argument("--pred", help="optional prediction .mha for cmd=case")
    parser.add_argument("--out-dir", help="output directory for cmd=case")
    parser.add_argument("--step-size", type=int, default=2,
                        help="marching-cubes step size for 3D surfaces; larger is faster/coarser")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    artifact_dir = Path(args.artifact_dir)
    stamp = time.strftime("%Y%m%d", time.gmtime())

    if args.cmd == "all":
        manifest = run_all(result_dir, artifact_dir)
        print(json.dumps(_json_sanitize(manifest), indent=2, allow_nan=False))
    elif args.cmd == "summary":
        print(json.dumps(write_result_summary(scan_eval_jsons(result_dir),
                                              stamp, artifact_dir),
                         indent=2, allow_nan=False))
    elif args.cmd == "weights":
        print(json.dumps(write_weight_manifest(scan_weight_manifest(),
                                               stamp, artifact_dir),
                         indent=2, allow_nan=False))
    elif args.cmd == "eval":
        if not args.eval_json:
            raise SystemExit("--eval-json is required for cmd=eval")
        eval_path = Path(args.eval_json)
        data = load_completed_eval(eval_path)
        if data is None:
            raise SystemExit(f"not a completed eval JSON: {args.eval_json}")
        suffix = _visual_suffix(eval_path)
        visual = write_eval_visualization(
            eval_path,
            out_text=artifact_dir / f"visual_{suffix}_{stamp}.txt",
            out_json=artifact_dir / f"visual_{suffix}_{stamp}.json",
        )
        print(json.dumps(_json_sanitize({
            "source": visual["source"],
            "scope": visual["scope"],
            "iou_f": visual["overall"]["iou_f"],
        }), indent=2, allow_nan=False))
    elif args.cmd == "case":
        if not args.case:
            raise SystemExit("--case is required for cmd=case")
        artifacts = visualize_case(
            args.case,
            pred_path=Path(args.pred) if args.pred else None,
            out_dir=Path(args.out_dir) if args.out_dir else None,
            step_size=args.step_size,
        )
        print(json.dumps(_json_sanitize(artifacts), indent=2, allow_nan=False))
    elif args.cmd == "hard-view":
        manifest = export_hard_view(
            case_set=args.case_set,
            out_dir=Path(args.out_dir) if args.out_dir else None,
            step_size=args.step_size,
        )
        print(json.dumps(_json_sanitize({
            "manifest": manifest["manifest"],
            "case_set": manifest["case_set"],
            "n_cases": len(manifest["cases"]),
        }), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

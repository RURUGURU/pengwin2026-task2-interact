"""PENGWIN 2026 Task 1 active utilities."""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from collections import Counter
from typing import Any, Literal
import itertools

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import distance as scipy_distance
from skimage.measure import regionprops
from skimage.morphology import ball
from skimage.segmentation import watershed

# NOTE [consolidation 2026-06-12]: `DATA_RAW` is imported lazily inside the two
# functions that use it (find_case_dir / list_cases), and the previously-unused
# `PAD` import was dropped. This keeps `utils` importable STANDALONE: the anatomy
# registry below is now defined here and re-exported by `core`, so a module-level
# `from core import ...` would create a core<->utils import cycle (utils is loaded
# while core is still partway through defining itself). Deferring core access to
# call time avoids that entirely (same pattern already used by `list_cases`).

# =============================================================================
# Anatomy <-> instance-ID registry (consolidated here from the former
# anatomy_registry.py module — 2026-06-12). This is the single source of truth
# for the PENGWIN anatomy <-> instance-ID encoding.
#
# PENGWIN encodes anatomy identity into each fragment's *global* instance ID via
# a fixed per-anatomy offset (capacity 50 IDs each)::
#
#     Sacrum 1-50,  LeftHip 51-100,  RightHip 101-150,  Femur 151-200
#
# Internally the model and loss only ever need ``(anatomy_group, local_id)``; the
# global ID is purely an *I/O encoding* at the evaluation boundary (the PENGWIN
# ``.mha`` submission format requires global IDs). All instance-ID arithmetic in
# ``loss.py`` / ``eval.py`` / ``core.py`` / ``preprocessing.py`` / ``utils.py``가
# 모두 이 정의를 사용하므로:
#
#   * adding an anatomy (or changing a range) is a one-row edit here, and
#   * "did we cover every anatomy?" is answerable by inspection rather than by
#     hunting scattered magic numbers (150, range(1,151), {1:1,2:51,3:101}, ...).
#
# DECOY WARNING -- these are *instance* IDs. They are a DIFFERENT namespace from:
#   * CASE IDs (``is_pelvic``/``is_femur`` in core.py: 1-120 / 151-200 / 251-420),
#   * HU / CT-clip thresholds (e.g. 150.0, 2000),
#   * probability thresholds (e.g. 0.50).
# The same digits (150, 200, 50) appear in all of these. Do NOT route those
# through this registry; only route true instance-ID arithmetic here.
#
# PELVIC-ONLY vs FULL -- some code paths are *deliberately* pelvic-scoped (the raw
# Dataset537 pelvic source, the 3-anatomy decode proxy). For those, pass
# ``pelvic_only=True`` so the Femur exclusion is an explicit, reviewable choice
# instead of a hidden ``<= 150``.
# =============================================================================


@dataclass(frozen=True)
class Anatomy:
    """One anatomy's global instance-ID block."""

    name: str
    index: int  # 1-indexed semantic class (background = 0)
    id_lo: int
    id_hi: int

    @property
    def capacity(self) -> int:
        return self.id_hi - self.id_lo + 1


# THE single source of truth.  Add an anatomy = add one row.
ANATOMY_REGISTRY: list[Anatomy] = [
    Anatomy("Sacrum", 1, 1, 50),
    Anatomy("LeftHip", 2, 51, 100),
    Anatomy("RightHip", 3, 101, 150),
    Anatomy("Femur", 4, 151, 200),
]

# Pelvic-only subset (Sacrum / LeftHip / RightHip). Used by code that is
# *intentionally* pelvic-scoped; the Femur exclusion there is a semantic choice,
# made explicit here rather than hidden behind a magic 150.
PELVIC_ANATOMY_INDICES: tuple[int, ...] = (1, 2, 3)

# -----------------------------------------------------------------------------
# Derived scalars (never hardcode these downstream).
# -----------------------------------------------------------------------------
NUM_ANATOMIES: int = len(ANATOMY_REGISTRY)  # 4 (max semantic anatomy class index)
MIN_INSTANCE_ID: int = min(a.id_lo for a in ANATOMY_REGISTRY)  # 1
MAX_INSTANCE_ID: int = max(a.id_hi for a in ANATOMY_REGISTRY)  # 200
PELVIC_MAX_INSTANCE_ID: int = max(
    a.id_hi for a in ANATOMY_REGISTRY if a.index in PELVIC_ANATOMY_INDICES
)  # 150
INSTANCE_CAPACITY: int = max(a.capacity for a in ANATOMY_REGISTRY)  # 50 (uniform today)

# -----------------------------------------------------------------------------
# Legacy-compatible views so existing `from core import ANATOMY_RANGES` (and the
# retired utils.py dicts) keep resolving without behavior change.
# -----------------------------------------------------------------------------
ANATOMY_RANGES: list[tuple[str, int, int]] = [
    (a.name, a.id_lo, a.id_hi) for a in ANATOMY_REGISTRY
]
ANATOMY_NAMES: list[str] = [a.name for a in ANATOMY_REGISTRY]
ANATOMY_RANGE_DICT: dict[str, tuple[int, int]] = {
    a.name: (a.id_lo, a.id_hi) for a in ANATOMY_REGISTRY
}
ANATOMY_TO_INDEX: dict[str, int] = {a.name: a.index for a in ANATOMY_REGISTRY}

_BY_INDEX: dict[int, Anatomy] = {a.index: a for a in ANATOMY_REGISTRY}
_BY_NAME: dict[str, Anatomy] = {a.name: a for a in ANATOMY_REGISTRY}

# Vectorized id -> anatomy index lookup (slot 0 = background/invalid). Capacity-
# agnostic: works even if anatomies have non-uniform widths.
_ID_TO_ANATOMY = np.zeros(MAX_INSTANCE_ID + 1, dtype=np.int16)
for _a in ANATOMY_REGISTRY:
    _ID_TO_ANATOMY[_a.id_lo : _a.id_hi + 1] = _a.index
del _a


# =============================================================================
# Anatomy registry lookups
# =============================================================================
def anatomy_by_index(anatomy_index: int) -> Anatomy | None:
    return _BY_INDEX.get(int(anatomy_index))


def anatomy_by_name(name: str) -> Anatomy | None:
    return _BY_NAME.get(name)


def anatomy_of_id(gid: int) -> int:
    """Semantic anatomy index (1..N) for a global instance id; 0 if bg/invalid."""
    g = int(gid)
    a = _BY_INDEX.get(int(_ID_TO_ANATOMY[g])) if 0 <= g <= MAX_INSTANCE_ID else None
    return a.index if a is not None else 0


def id_range(anatomy_index: int) -> tuple[int, int]:
    """(lo, hi) global-ID block for an anatomy index. Replaces V5_ANATOMY_RANGES[name]."""
    a = _BY_INDEX[int(anatomy_index)]
    return (a.id_lo, a.id_hi)


def all_anatomy_ranges(pelvic_only: bool = False) -> dict[int, tuple[int, int]]:
    """{anatomy_index: (lo, hi)}. Replaces scattered {1:(1,50),...} dicts.

    pelvic_only=True omits Femur (the deliberate 3-anatomy proxy view).
    """
    return {
        a.index: (a.id_lo, a.id_hi)
        for a in ANATOMY_REGISTRY
        if (not pelvic_only or a.index in PELVIC_ANATOMY_INDICES)
    }


def anatomy_start_ids(pelvic_only: bool = False) -> dict[int, int]:
    """{anatomy_index: lo}. Replaces next_by_anatomy = {1:1, 2:51, 3:101}[, 4:151]."""
    return {
        a.index: a.id_lo
        for a in ANATOMY_REGISTRY
        if (not pelvic_only or a.index in PELVIC_ANATOMY_INDICES)
    }


def anatomy_ranges_by_name(pelvic_only: bool = False) -> dict[str, tuple[int, int]]:
    """{name: (lo, hi)}. Drop-in for V5_ANATOMY_RANGES / V5_ANATOMY_RANGES_WITH_FEMUR."""
    return {
        a.name: (a.id_lo, a.id_hi)
        for a in ANATOMY_REGISTRY
        if (not pelvic_only or a.index in PELVIC_ANATOMY_INDICES)
    }


# =============================================================================
# Global <-> local id conversion (capacity-aware; not hardcoded to 50)
# =============================================================================
def global_to_local(gid: int) -> int:
    """Global id -> anatomy-local id (1..capacity); 0 if bg/invalid.

    Replaces ``((id - 1) % 50) + 1``.
    """
    g = int(gid)
    a = _BY_INDEX.get(anatomy_of_id(g))
    return (g - a.id_lo + 1) if a is not None else 0


def local_to_global(anatomy_index: int, local_id: int) -> int:
    """(anatomy_index, local_id 1..capacity) -> global id.

    Replaces ``id_lo + counter`` reconstruction at the decode boundary.
    """
    a = _BY_INDEX[int(anatomy_index)]
    return a.id_lo + int(local_id) - 1


# =============================================================================
# Vectorized array helpers (replace the scattered magic-number masks)
# =============================================================================
def valid_instance_mask(arr: np.ndarray, pelvic_only: bool = False) -> np.ndarray:
    """Boolean mask of voxels holding a valid global instance id.

    Replaces ``(arr >= 1) & (arr <= 150)`` (and the half-migrated ``<= 200``).
    pelvic_only=True caps at RightHip (150) for intentionally pelvic-scoped code.
    """
    a = np.asarray(arr)
    hi = PELVIC_MAX_INSTANCE_ID if pelvic_only else MAX_INSTANCE_ID
    return (a >= MIN_INSTANCE_ID) & (a <= hi)


def clip_to_valid_instances(arr: np.ndarray, pelvic_only: bool = False) -> np.ndarray:
    """Zero out voxels that are not valid instance ids; keep valid ones.

    Replaces ``np.where((inst >= 1) & (inst <= 150), inst, 0)``.
    """
    a = np.asarray(arr)
    return np.where(valid_instance_mask(a, pelvic_only=pelvic_only), a, 0)


def anatomy_index_array(arr: np.ndarray) -> np.ndarray:
    """Per-voxel semantic anatomy index (0 where bg/invalid/out-of-range)."""
    a = np.asarray(arr)
    out = np.zeros(a.shape, dtype=np.int16)
    inb = (a >= 0) & (a <= MAX_INSTANCE_ID)
    out[inb] = _ID_TO_ANATOMY[a[inb].astype(np.int64)]
    return out


def same_anatomy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise: do (foreground) global ids a, b belong to the same anatomy?

    Replaces ``((a - 1) // 50) == ((b - 1) // 50)``. Note the ``// 50`` form
    already handles all four anatomies; this helper is capacity-aware and also
    excludes background-background matches.
    """
    ai = anatomy_index_array(a)
    bi = anatomy_index_array(b)
    return (ai == bi) & (ai > 0)

try:
    import SimpleITK as sitk
except ImportError:  # allow pure numpy tests to import the module
    sitk = None


try:
    import cc3d
except ImportError:  # pragma: no cover - 환경 QA에서 의존성 유무를 별도로 확인한다.
    cc3d = None

try:
    import skfmm
except ImportError:  # pragma: no cover - scipy fallback으로 import 가능성을 유지한다.
    skfmm = None

CANONICAL_ORIENTATION = "LPS"
CT_HU_CLIP_RANGE = (-1000.0, 2000.0)
ORIENTATION_CONTRACT_VERSION = "lps_ct_v1"


def _require_sitk():
    if sitk is None:
        raise ImportError("SimpleITK is required for image orientation and .mha I/O helpers.")
    return sitk


def orientation_code(img: sitk.Image) -> str:
    """SimpleITK image direction에 대응하는 DICOM orientation code를 반환한다."""
    sitk_mod = _require_sitk()
    return sitk_mod.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img.GetDirection())


def canonicalize_sitk(img: sitk.Image,
                      orientation: str = CANONICAL_ORIENTATION) -> sitk.Image:
    """리샘플링 없이 image를 canonical orientation으로 permute/flip 한다."""
    if orientation_code(img) == orientation:
        return img
    sitk_mod = _require_sitk()
    return sitk_mod.DICOMOrient(img, orientation)


def find_case_dir(case_id: str) -> Path | None:
    """Locate case folder under data/task1_2/extracted/<part>/<case_id>/."""
    from core import DATA_RAW
    cid = str(case_id).zfill(3)
    for part in sorted(DATA_RAW.iterdir()):
        if not part.is_dir():
            continue
        cd = part / cid
        if (cd / "image.mha").exists() and (cd / "label.mha").exists():
            return cd
    return None


def list_cases(case_filter: str | None = None) -> list[Path]:
    """Return all case dirs, optionally filtered by `pelvic` or `femur`."""
    from core import DATA_RAW, is_femur, is_pelvic

    out = []
    for part in sorted(DATA_RAW.iterdir()):
        if not part.is_dir():
            continue
        for cd in sorted(part.iterdir()):
            if not (cd.is_dir() and cd.name.isdigit()):
                continue
            cid = int(cd.name)
            if case_filter == "pelvic" and not is_pelvic(cid):
                continue
            if case_filter == "femur" and not is_femur(cid):
                continue
            if (cd / "image.mha").exists() and (cd / "label.mha").exists():
                out.append(cd)
    return out


def inst_to_anat(inst: np.ndarray) -> np.ndarray:
    """Map PENGWIN fragment IDs to global anatomy labels: bg/Sa/LH/RH/Femur."""
    out = np.zeros_like(inst, dtype=np.uint8)
    for idx, (_name, lo, hi) in enumerate(ANATOMY_RANGES, start=1):
        out[(inst >= lo) & (inst <= hi)] = idx
    return out


def crop_save_mha(arr_full: np.ndarray, ref_img: sitk.Image, bbox: tuple,
                  out_path: Path, dtype=np.uint8) -> None:
    """Save a cropped array as `.mha` while preserving physical location."""
    sitk_mod = _require_sitk()
    arr = arr_full[bbox].astype(dtype)
    # [QC][Invariant:nonempty_crop]
    # SimpleITK can fail with a low-level "paste IO region" error when a bbox
    # accidentally yields an empty axis. Raise here with the actual bbox so data
    # builder bugs are traceable before a partially written dataset is produced.
    if 0 in arr.shape:
        raise ValueError(f"empty crop for {out_path}: shape={arr.shape}, bbox={bbox}")
    save_crop_array_mha(arr, ref_img, bbox, out_path, dtype=dtype)


def save_crop_array_mha(arr_crop: np.ndarray, ref_img: sitk.Image, bbox: tuple,
                        out_path: Path, dtype=np.uint8) -> None:
    """Save an already-cropped array with physical origin derived from `bbox`.

    [DATA][Scope:roi_dataset]
    V5 target generation creates ROI-sized labels directly. Re-cropping those
    labels with `bbox` would silently create empty arrays. This helper keeps the
    same spatial metadata contract as `crop_save_mha` while guarding that the
    provided crop shape matches the bbox extent.
    """
    sitk_mod = _require_sitk()
    expected_shape = tuple(int(s.stop - s.start) for s in bbox)
    if tuple(arr_crop.shape) != expected_shape:
        raise ValueError(
            f"crop shape mismatch for {out_path}: arr={tuple(arr_crop.shape)} "
            f"expected={expected_shape} bbox={bbox}"
        )
    if 0 in arr_crop.shape:
        raise ValueError(f"empty crop for {out_path}: shape={arr_crop.shape}, bbox={bbox}")
    arr = arr_crop.astype(dtype, copy=False)
    img = sitk_mod.GetImageFromArray(arr)
    spacing = ref_img.GetSpacing()
    direction = ref_img.GetDirection()
    origin = np.array(ref_img.GetOrigin(), dtype=float)
    start_zyx = np.array([s.start for s in bbox], dtype=float)
    start_xyz = start_zyx[::-1]
    direction_m = np.array(direction).reshape(3, 3)
    new_origin = origin + direction_m @ (start_xyz * np.array(spacing))
    img.SetSpacing(spacing)
    img.SetDirection(direction)
    img.SetOrigin(tuple(float(x) for x in new_origin))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk_mod.WriteImage(img, str(out_path), useCompression=True)


def save_full_mha(arr: np.ndarray, ref_img: sitk.Image, out_path: Path,
                  dtype=np.uint8) -> None:
    """Save an array as `.mha` with copied geometry."""
    sitk_mod = _require_sitk()
    img = sitk_mod.GetImageFromArray(arr.astype(dtype, copy=False))
    img.CopyInformation(ref_img)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk_mod.WriteImage(img, str(out_path), useCompression=True)


def clip_ct_hu(arr: np.ndarray,
               hu_range: tuple[float, float] = CT_HU_CLIP_RANGE) -> np.ndarray:
    """Clip CT intensities exactly once before nnU-Net normalization.

    This helper is deliberately shared by preprocessing and standalone
    inference. The model was trained on LPS-oriented CT volumes clipped to this
    bone-safe HU window; any ad-hoc inference path that skips the same clip or
    keeps source RAS voxel order breaks the left/right label contract.
    """
    lo, hi = hu_range
    return np.clip(arr, lo, hi).astype(arr.dtype, copy=False)


def _image_metadata(img: sitk.Image) -> dict:
    """Return JSON-safe SimpleITK geometry and orientation metadata."""
    return {
        "orientation": orientation_code(img),
        "size_xyz": [int(v) for v in img.GetSize()],
        "spacing_xyz": [float(v) for v in img.GetSpacing()],
        "origin_xyz": [float(v) for v in img.GetOrigin()],
        "direction": [float(v) for v in img.GetDirection()],
    }


def prepare_lps_ct_for_nnunet(source_image: str | Path | sitk.Image,
                              out_path: str | Path,
                              hu_range: tuple[float, float] = CT_HU_CLIP_RANGE) -> dict:
    """Prepare one CT for nnU-Net using the training-time orientation contract.

    Contract:
        source read -> DICOMOrient(..., "LPS") -> HU clip [-1000, 2000] -> .mha.

    `DICOMOrient` changes voxel order and direction metadata while preserving
    physical anatomy. That is exactly what we need for bilateral labels: the
    Pelvic model learned LeftHip/RightHip in canonical LPS coordinates, so
    standalone inference must not feed original RAS source bytes directly.
    """
    sitk_mod = _require_sitk()
    source_path = str(source_image) if isinstance(source_image, (str, Path)) else None
    img = sitk_mod.ReadImage(source_path) if source_path is not None else source_image
    source_meta = _image_metadata(img)

    img_lps = canonicalize_sitk(img, CANONICAL_ORIENTATION)
    clipped = clip_ct_hu(sitk_mod.GetArrayFromImage(img_lps), hu_range=hu_range)
    out_path = Path(out_path)
    save_full_mha(clipped, img_lps, out_path, dtype=clipped.dtype)

    lo, hi = hu_range
    return {
        "contract_version": ORIENTATION_CONTRACT_VERSION,
        "source_path": source_path,
        "source": source_meta,
        "output_path": str(out_path),
        "output": _image_metadata(img_lps),
        "hu_clip_range": [float(lo), float(hi)],
        "clipped_min": float(np.min(clipped)),
        "clipped_max": float(np.max(clipped)),
        "input_dtype": str(sitk_mod.GetArrayFromImage(img).dtype),
        "output_dtype": str(clipped.dtype),
    }


# =============================================================================
# 흡수된 BoundaryFragment V3 target/decoder primitives
# [AUDIT][Risk:High][Scope:module_consolidation]
# Reason: 프로젝트 루트의 Python 파일을 8개로 제한하기 위해 기존 boundary_fragment.py의
# 공용 label/target/decoder 계약을 utils.py로 흡수한다. 기존 함수명은 유지해서
# preprocessing/eval/visualize의 동작 표면을 바꾸지 않는다.
# QA: rg로 외부 boundary_fragment import가 남지 않는지 확인한다.
# =============================================================================
BFV3_LABELS = {
    "background": 0,
    "external_context": 1,
    "fracture_barrier": 2,
    "fragment_shell": 3,
    "fragment_core": 4,
}
BFV3_CLASS_NAMES = [
    "background",
    "external_context",
    "thin_contact_ridge",
    "fragment_shell",
    "fragment_core",
]
BFV3_SUPPORT_LABELS = (2, 3, 4)
BFV3_CORE_LABEL = 4
BFV3_BARRIER_LABEL = 2


@dataclass(frozen=True)
class BoundaryFragmentParams:
    """Physical-mm geometry for the V3 target."""

    target_profile: str = "v3_thin_ridge"
    contact_band_mm: float = 3.0
    contact_search_mm: float = 3.0
    contact_ridge_mm: float = 1.0
    contact_same_anatomy_only: bool = False
    external_band_mm: float = 3.0
    shell_mm: float = 8.0
    tiny_core_radius_mm: float = 1.5
    # [2026-06-10] core representation. CANONICAL = "erosion" (the MEASURED winner).
    # "erosion" = fixed shell_mm erosion (8mm). Hypothesis was it EMPTIES the core for thin
    # fragments -> point-seed fallback -> speckled core. "medial_skeleton" = medial-axis dynamic
    # boundary (skeletonize+dilate) was tried to fix that speckle. BUT the held-out eval REFUTED
    # the hypothesis: medial REGRESSED official metrics (dev IoU-F 0.772 -> 0.711, HD95 19.0 ->
    # 22.74) despite lowering split_err. The deployed/best model + the canonical
    # boundary_fragment_v3_targets sidecar are EROSION, so the default is erosion to reproduce the
    # winner on any sidecar regen. The medial_skeleton branch is retained (below) for revisiting,
    # not as the default. See analysis/ab_medial_iouf_20260609_checkpoint_best.json.
    core_mode: str = "erosion"
    core_dilate_iter: int = 2


def pelvic_instance_mask(inst: np.ndarray) -> np.ndarray:
    """Return the valid-instance mask for original PENGWIN labels.

    Despite the historical "pelvic" name, this now spans ALL anatomies incl.
    Femur (1-200) via the registry. That is a safe superset: pelvic-only data
    holds no 151-200 voxels (so the mask is identical there), while per-anatomy
    Femur ROIs are correctly included instead of silently zeroed. Callers that
    genuinely need a 3-anatomy pelvic subset must use the explicit pelvic_only
    helpers in the anatomy registry (defined above in this module), not this
    function.
    """

    return valid_instance_mask(inst)


def _same_anatomy_ids(fragment_id: int) -> tuple[int, int]:
    """(lo, hi) of the anatomy block containing ``fragment_id``; (0, 0) if none.

    Registry-backed so Femur (151-200) resolves to its block instead of falling
    through to (0, 0) and breaking same-anatomy contact for femur fragments.
    """
    idx = anatomy_of_id(int(fragment_id))
    return id_range(idx) if idx else (0, 0)


def _bbox(mask: np.ndarray, radius_vox: int = 0) -> tuple[slice, slice, slice] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    lo = np.maximum(coords.min(axis=0) - int(radius_vox), 0)
    hi = np.minimum(coords.max(axis=0) + int(radius_vox) + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def contact_barrier_distance(inst: np.ndarray,
                             spacing_zyx: tuple[float, float, float],
                             max_radius_mm: float = 3.0,
                             same_anatomy_only: bool = True) -> np.ndarray:
    """Distance from each fragment voxel to the nearest different fragment.

    The previous broad-contact target confused any bone-like surface with a
    fracture separator. The key distinction is now "near another pelvic
    fragment", optionally restricted to the same anatomy for legacy baselines.
    External cortex and femur/vertebra-like structures remain excluded because
    only valid pelvic instance IDs can become the "other" fragment.

    The returned array is finite only for pelvic fragment voxels that have a
    same-anatomy different-fragment neighbor within `max_radius_mm`. Oracle
    sweeps reuse this map for several thresholds instead of recomputing the
    expensive distance transform for every threshold.
    """

    inst = np.where(pelvic_instance_mask(inst), inst, 0).astype(np.uint16, copy=False)
    out = np.full(inst.shape, np.inf, dtype=np.float32)
    max_spacing = max(float(s) for s in spacing_zyx)
    radius_vox = max(1, int(np.ceil(float(max_radius_mm) / max_spacing)) + 1)
    fragment_ids = [int(v) for v in np.unique(inst) if MIN_INSTANCE_ID <= int(v) <= MAX_INSTANCE_ID]
    for fragment_id in fragment_ids:
        frag = inst == fragment_id
        box = _bbox(frag, radius_vox=radius_vox)
        if box is None:
            continue
        crop = inst[box]
        frag_crop = crop == fragment_id
        if same_anatomy_only:
            lo, hi = _same_anatomy_ids(fragment_id)
            other_fragment = (crop >= lo) & (crop <= hi) & (crop != fragment_id)
        else:
            other_fragment = (crop > 0) & (crop != fragment_id)
        if not other_fragment.any():
            continue
        dist_to_other = ndi.distance_transform_edt(~other_fragment, sampling=spacing_zyx)
        finite = frag_crop & (dist_to_other <= float(max_radius_mm))
        if finite.any():
            view = out[box]
            view[finite] = np.minimum(view[finite], dist_to_other[finite].astype(np.float32, copy=False))
            out[box] = view
    return out


def contact_barrier_mask(inst: np.ndarray,
                         spacing_zyx: tuple[float, float, float],
                         radius_mm: float = 3.0,
                         same_anatomy_only: bool = True) -> np.ndarray:
    """Different-fragment surface band inside GT fragments."""

    return contact_barrier_distance(
        inst,
        spacing_zyx,
        max_radius_mm=radius_mm,
        same_anatomy_only=same_anatomy_only,
    ) <= float(radius_mm)


def thin_contact_ridge_mask(inst: np.ndarray,
                            spacing_zyx: tuple[float, float, float],
                            search_mm: float = 3.0,
                            ridge_mm: float = 1.0,
                            contact_distance: np.ndarray | None = None,
                            same_anatomy_only: bool = False) -> np.ndarray:
    """Support-preserving same-anatomy contact ridge for V3.1.

    The old class-2 target consumed every voxel within the contact band. That
    erased too much shell in thin fragments. V3.1 keeps only fragment-surface
    voxels that face a nearby same-anatomy fragment; shell remains available on
    both sides of the ridge for marker assignment.
    """

    inst = np.where(pelvic_instance_mask(inst), inst, 0).astype(np.uint16, copy=False)
    contact_dist = contact_distance
    if contact_dist is None:
        contact_dist = contact_barrier_distance(
            inst,
            spacing_zyx,
            max_radius_mm=search_mm,
            same_anatomy_only=same_anatomy_only,
        )
    ridge = np.zeros(inst.shape, dtype=bool)
    fragment_ids = [int(v) for v in np.unique(inst) if MIN_INSTANCE_ID <= int(v) <= MAX_INSTANCE_ID]
    for fragment_id in fragment_ids:
        frag = inst == fragment_id
        box = _bbox(frag, radius_vox=1)
        if box is None:
            continue
        frag_crop = frag[box]
        if not frag_crop.any():
            continue
        dist_inside = ndi.distance_transform_edt(frag_crop, sampling=spacing_zyx)
        surface = frag_crop & (dist_inside <= float(ridge_mm))
        facing_other = contact_dist[box] <= float(search_mm)
        mark = surface & facing_other
        if mark.any():
            view = ridge[box]
            view[mark] = True
            ridge[box] = view
    return ridge


def _single_connected_core_seed(frag_crop: np.ndarray,
                                dist_inside: np.ndarray,
                                spacing_zyx: tuple[float, float, float],
                                radius_mm: float) -> np.ndarray:
    """Create one deterministic connected marker inside a fragment crop."""

    seed = np.zeros_like(frag_crop, dtype=bool)
    if not frag_crop.any():
        return seed
    center = np.unravel_index(int(np.argmax(dist_inside * frag_crop)), frag_crop.shape)
    seed[center] = True
    if float(radius_mm) <= 0.0:
        return seed
    zz, yy, xx = np.ogrid[
        :frag_crop.shape[0],
        :frag_crop.shape[1],
        :frag_crop.shape[2],
    ]
    dz = (zz - int(center[0])) * float(spacing_zyx[0])
    dy = (yy - int(center[1])) * float(spacing_zyx[1])
    dx = (xx - int(center[2])) * float(spacing_zyx[2])
    allowed = frag_crop & ((dz * dz + dy * dy + dx * dx) <= float(radius_mm) ** 2)
    grown = ndi.binary_propagation(seed, structure=np.ones((3, 3, 3), dtype=bool), mask=allowed)
    return grown.astype(bool, copy=False)


def compute_boundary_fragment_target(inst: np.ndarray,
                                     spacing_zyx: tuple[float, float, float],
                                     params: BoundaryFragmentParams | None = None,
                                     precomputed_barrier: np.ndarray | None = None,
                                     precomputed_contact_distance: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the 5-class V3 target from original pelvic fragment IDs.

    Invariants enforced here:
        - class 2 never overlaps class 1;
        - class 1 is outside fragment support;
        - every GT fragment receives class 4 core supervision, even tiny
          fragments without an 8 mm deep interior;
        - distances are evaluated in physical mm.
    """

    params = params or BoundaryFragmentParams()
    inst = np.where(pelvic_instance_mask(inst), inst, 0).astype(np.uint16, copy=False)
    support = inst > 0
    target = np.zeros(inst.shape, dtype=np.uint8)

    if precomputed_barrier is not None:
        barrier = precomputed_barrier.astype(bool, copy=False)
    elif params.target_profile == "v3_band":
        barrier = contact_barrier_mask(
            inst,
            spacing_zyx,
            radius_mm=params.contact_band_mm,
            same_anatomy_only=params.contact_same_anatomy_only,
        )
    elif params.target_profile == "v3_thin_ridge":
        barrier = thin_contact_ridge_mask(
            inst,
            spacing_zyx,
            search_mm=params.contact_search_mm,
            ridge_mm=params.contact_ridge_mm,
            contact_distance=precomputed_contact_distance,
            same_anatomy_only=params.contact_same_anatomy_only,
        )
    else:
        raise ValueError(f"unknown BoundaryFragment target_profile={params.target_profile!r}")
    target[barrier] = BFV3_LABELS["fracture_barrier"]

    # External context is the hard negative band immediately outside pelvic
    # fragments. It supervises "bone/cortex-looking but not a fragment split".
    if support.any() and float(params.external_band_mm) > 0.0:
        dist_to_support = ndi.distance_transform_edt(~support, sampling=spacing_zyx)
        external = (~support) & (dist_to_support <= float(params.external_band_mm))
        target[external] = BFV3_LABELS["external_context"]

    zero_core_fragments: list[int] = []
    fallback_core_fragments: list[int] = []
    fragment_ids = [int(v) for v in np.unique(inst) if MIN_INSTANCE_ID <= int(v) <= MAX_INSTANCE_ID]
    for fragment_id in fragment_ids:
        frag = inst == fragment_id
        if not frag.any():
            continue
        box = _bbox(frag, radius_vox=1)
        if box is None:
            continue
        frag_crop = frag[box]
        barrier_crop = barrier[box]
        dist_inside = ndi.distance_transform_edt(frag_crop, sampling=spacing_zyx)
        if params.core_mode == "medial_skeleton":
            # winner's medial-axis dynamic boundary: core = skeleton (connected centerline,
            # 1 CC per fragment CC) dilated by core_dilate_iter -> connected, non-empty,
            # shape-following core for thin AND thick fragments (no fixed-erosion emptying).
            from skimage.morphology import skeletonize
            skel = skeletonize(frag_crop)
            if skel.any():
                core_crop = (ndi.binary_dilation(skel, iterations=int(params.core_dilate_iter))
                             & frag_crop & ~barrier_crop)
            else:
                core_crop = np.zeros_like(frag_crop)
            if core_crop.any():
                core_cc, n_core_cc = ndi.label(core_crop)
                if n_core_cc > 1:
                    sizes = np.bincount(core_cc.ravel()); sizes[0] = 0
                    core_crop = core_cc == int(sizes.argmax())  # keep largest connected tube
        else:  # legacy "erosion": fixed shell_mm interior (empties thin fragments -> speckle)
            core_crop = frag_crop & (dist_inside > float(params.shell_mm)) & ~barrier_crop
            if core_crop.any():
                core_cc, n_core_cc = ndi.label(core_crop)
                if n_core_cc > 1:
                    keep_point = np.unravel_index(int(np.argmax(dist_inside * core_crop)), core_crop.shape)
                    keep_id = int(core_cc[keep_point])
                    core_crop = core_cc == keep_id
        core = np.zeros_like(frag)
        core[box] = core_crop
        if not core.any():
            # Tiny fragments are real GT instances. Give them a deterministic
            # central seed rather than letting the model learn "no core".
            max_dist = float(dist_inside.max())
            if max_dist <= 0:
                zero_core_fragments.append(fragment_id)
                center = np.argwhere(frag_crop)[0]
                core_crop = np.zeros_like(frag_crop)
                core_crop[tuple(center)] = True
                core = np.zeros_like(frag)
                core[box] = core_crop
            else:
                # If a small/thin fragment lies entirely inside the 3 mm
                # contact band, still reserve a central core seed. Without
                # this exception the decoder has no marker for real fragments
                # precisely where IoU-F is most fragile.
                core_crop = _single_connected_core_seed(
                    frag_crop,
                    dist_inside,
                    spacing_zyx,
                    radius_mm=params.tiny_core_radius_mm,
                )
                core = np.zeros_like(frag)
                core[box] = core_crop
            fallback_core_fragments.append(fragment_id)
        shell = frag & ~barrier & ~core
        target[shell] = BFV3_LABELS["fragment_shell"]
        target[core] = BFV3_LABELS["fragment_core"]

    counts = np.bincount(target.ravel(), minlength=5)
    core_component_count_by_fragment: dict[str, int] = {}
    multi_core_fragments: list[int] = []
    for fragment_id in fragment_ids:
        core_components = int(ndi.label((inst == fragment_id) & (target == BFV3_LABELS["fragment_core"]))[1])
        core_component_count_by_fragment[str(fragment_id)] = core_components
        if core_components != 1:
            multi_core_fragments.append(fragment_id)
    audit = {
        "label_counts": {str(i): int(counts[i]) for i in range(5)},
        "label_fractions": {str(i): float(counts[i]) / max(1, int(counts.sum())) for i in range(5)},
        "n_fragments": int(len(fragment_ids)),
        "zero_core_fragments": zero_core_fragments,
        "fallback_core_fragments": fallback_core_fragments,
        "core_component_count_by_fragment": core_component_count_by_fragment,
        "multi_core_fragments": multi_core_fragments,
        "class2_class1_overlap": int(((target == 2) & (target == 1)).sum()),
        "class1_inside_support_voxels": int(((target == 1) & support).sum()),
        "class2_outside_support_voxels": int(((target == 2) & ~support).sum()),
        "params": {
            "target_profile": params.target_profile,
            "contact_band_mm": float(params.contact_band_mm),
            "contact_search_mm": float(params.contact_search_mm),
            "contact_ridge_mm": float(params.contact_ridge_mm),
            "contact_same_anatomy_only": bool(params.contact_same_anatomy_only),
            "external_band_mm": float(params.external_band_mm),
            "shell_mm": float(params.shell_mm),
            "tiny_core_radius_mm": float(params.tiny_core_radius_mm),
        },
    }
    return target, audit


def decode_boundary_fragment_labels(labels: np.ndarray) -> np.ndarray:
    """Fixed V3 decoder from argmax semantic labels to fragment instances.

    No threshold search is used. Class 4 components are markers, class 3 is
    assignable support, class 2 blocks growth and is assigned back after the
    class 3/4 support has been split.
    """

    labels = labels.astype(np.uint8, copy=False)
    grow_support = (labels == BFV3_LABELS["fragment_shell"]) | (labels == BFV3_LABELS["fragment_core"])
    barrier = labels == BFV3_LABELS["fracture_barrier"]
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    seed = labels == BFV3_LABELS["fragment_core"]
    comps, n_comps = ndi.label(grow_support, structure=structure)
    out = np.zeros(labels.shape, dtype=np.uint16)
    next_id = 1
    for comp_id in range(1, n_comps + 1):
        comp = comps == comp_id
        comp_seed, n_seed = ndi.label(seed & comp, structure=structure)
        if n_seed <= 1:
            out[comp] = next_id
            next_id += 1
            continue
        box = _bbox(comp, radius_vox=0)
        if box is None:
            continue
        seed_crop = comp_seed[box]
        comp_crop = comp[box]
        _, indices = ndi.distance_transform_edt(seed_crop == 0, return_indices=True)
        assigned = seed_crop[tuple(indices)]
        for seed_id in range(1, n_seed + 1):
            part_crop = comp_crop & (assigned == seed_id)
            if not part_crop.any():
                continue
            view = out[box]
            view[part_crop] = next_id
            out[box] = view
            next_id += 1

    if barrier.any() and out.any():
        _, indices = ndi.distance_transform_edt(out == 0, return_indices=True)
        nearest = out[tuple(indices)]
        out[barrier] = nearest[barrier]
    return out


def decode_boundary_fragment_geodesic(labels: np.ndarray,
                                      barrier_cost: float = 12.0) -> np.ndarray:
    """Marker-controlled watershed with class 2 as a high-cost ridge.

    Unlike the initial hard-barrier decoder, class 2 remains inside the support
    mask. It does not disappear and get nearest-filled later; instead it raises
    the crossing cost between class-4 core markers. This is the intended V3
    oracle test for whether class2 acts as a separator without damaging support
    continuity.
    """

    labels = labels.astype(np.uint8, copy=False)
    support = np.isin(labels, BFV3_SUPPORT_LABELS)
    markers, n_markers = ndi.label(labels == BFV3_CORE_LABEL)
    if n_markers == 0:
        out, _ = ndi.label(support)
        return out.astype(np.uint16, copy=False)
    elevation = np.full(labels.shape, 100.0, dtype=np.float32)
    elevation[labels == BFV3_CORE_LABEL] = 0.0
    elevation[labels == BFV3_LABELS["fragment_shell"]] = 1.0
    elevation[labels == BFV3_BARRIER_LABEL] = float(barrier_cost)
    decoded = watershed(
        elevation,
        markers=markers.astype(np.int32, copy=False),
        mask=support,
        connectivity=1,
    )
    return decoded.astype(np.uint16, copy=False)


def decode_boundary_fragment_geodesic_v2(labels: np.ndarray) -> np.ndarray:
    """V3.1 primary marker assignment.

    This uses the same no-threshold watershed contract as `geodesic_marker`,
    but raises the class-2 crossing cost for the thinner ridge target. The
    profile name is explicit so reports can distinguish V3.1 from the previous
    broad-band geodesic ablation.
    """

    return decode_boundary_fragment_geodesic(labels, barrier_cost=64.0)


def decode_boundary_fragment(labels: np.ndarray,
                             profile: str = "hard_barrier") -> np.ndarray:
    """Dispatch the fixed V3 decoder profile."""

    if profile == "hard_barrier":
        return decode_boundary_fragment_labels(labels)
    if profile == "geodesic_marker":
        return decode_boundary_fragment_geodesic(labels)
    if profile == "geodesic_marker_v2":
        return decode_boundary_fragment_geodesic_v2(labels)
    raise ValueError(f"unknown BoundaryFragment decoder profile: {profile}")


def instance_iouf(pred: np.ndarray, gt: np.ndarray, min_gt_voxels: int = 1) -> dict[str, Any]:
    """Greedy IoU-F summary for oracle/root-cause gates.

    This is intentionally independent of fragment numeric IDs because decoded
    instances are arbitrary connected-component IDs.
    """

    pred = pred.astype(np.int32, copy=False)
    gt = np.where(pelvic_instance_mask(gt), gt, 0).astype(np.int32, copy=False)
    gt_counts = np.bincount(gt.ravel())
    pred_counts = np.bincount(pred.ravel())
    gt_ids = [i for i in range(1, len(gt_counts)) if int(gt_counts[i]) >= int(min_gt_voxels)]
    pred_ids = [i for i in range(1, len(pred_counts)) if int(pred_counts[i]) > 0]
    rows = []
    if gt_ids and pred_ids:
        max_gt = int(gt.max(initial=0)) + 1
        valid = (gt > 0) & (pred > 0)
        pair_codes, intersections = np.unique(
            pred[valid].astype(np.int64) * max_gt + gt[valid].astype(np.int64),
            return_counts=True,
        )
        best: dict[int, tuple[int | None, float]] = {int(gt_id): (None, 0.0) for gt_id in gt_ids}
        for code, inter in zip(pair_codes, intersections):
            pred_id = int(code // max_gt)
            gt_id = int(code % max_gt)
            if gt_id not in best:
                continue
            union = int(gt_counts[gt_id]) + int(pred_counts[pred_id]) - int(inter)
            iou = float(inter / max(1, union))
            if iou > best[gt_id][1]:
                best[gt_id] = (pred_id, iou)
        for gt_id in gt_ids:
            pred_id, iou = best[int(gt_id)]
            rows.append({
                "gt_id": int(gt_id),
                "gt_voxels": int(gt_counts[int(gt_id)]),
                "best_pred_id": pred_id,
                "best_iou": float(iou),
            })
    else:
        rows = [
            {
                "gt_id": int(gt_id),
                "gt_voxels": int(gt_counts[int(gt_id)]),
                "best_pred_id": None,
                "best_iou": 0.0,
            }
            for gt_id in gt_ids
        ]
    ious = [row["best_iou"] for row in rows]
    missing = [row["gt_id"] for row in rows if row["best_iou"] <= 0.0]
    tiny_rows = [row for row in rows if row["gt_voxels"] < 1000]
    return {
        "iou_f_mean": float(np.mean(ious)) if ious else 0.0,
        "iou_f_min": float(np.min(ious)) if ious else 0.0,
        "n_gt_fragments": int(len(gt_ids)),
        "n_pred_fragments": int(len(pred_ids)),
        "missing_gt_fragments": missing,
        "tiny_fragment_count": int(len(tiny_rows)),
        "tiny_iou_f_mean": float(np.mean([row["best_iou"] for row in tiny_rows])) if tiny_rows else None,
        "per_gt": rows,
    }


def oracle_topology_diagnostics(decoded: np.ndarray,
                                gt: np.ndarray,
                                target: np.ndarray,
                                iouf: dict[str, Any]) -> dict[str, Any]:
    """Explain why an oracle target/decoder failed.

    The rows identify whether low IoU is caused by one GT fragment being split
    into several predictions or by one prediction absorbing multiple GT
    fragments. This is deliberately computed from overlaps, not visual
    inspection, so the JSON is enough to guide the next target/decoder change.
    """

    gt = np.where(pelvic_instance_mask(gt), gt, 0).astype(np.int32, copy=False)
    decoded = decoded.astype(np.int32, copy=False)
    target = target.astype(np.uint8, copy=False)
    gt_ids = [int(v) for v in np.unique(gt) if int(v) > 0]
    pred_ids = [int(v) for v in np.unique(decoded) if int(v) > 0]
    max_gt = int(gt.max(initial=0)) + 1
    valid = (gt > 0) & (decoded > 0)
    pair_overlap: dict[tuple[int, int], int] = {}
    if valid.any():
        pair_codes, counts = np.unique(
            decoded[valid].astype(np.int64) * max_gt + gt[valid].astype(np.int64),
            return_counts=True,
        )
        for code, count in zip(pair_codes, counts):
            pair_overlap[(int(code % max_gt), int(code // max_gt))] = int(count)
    per_gt_best = {int(row["gt_id"]): row for row in iouf.get("per_gt", [])}
    pred_to_gt = {
        pred_id: [
            gt_id for gt_id in gt_ids
            if pair_overlap.get((gt_id, pred_id), 0) > 0
        ]
        for pred_id in pred_ids
    }
    rows = []
    for gt_id in gt_ids:
        gt_mask = gt == gt_id
        overlaps = [
            {
                "pred_id": int(pred_id),
                "voxels": int(pair_overlap.get((gt_id, pred_id), 0)),
                "fraction_of_gt": float(pair_overlap.get((gt_id, pred_id), 0) / max(1, int(gt_mask.sum()))),
            }
            for pred_id in pred_ids
            if pair_overlap.get((gt_id, pred_id), 0) > 0
        ]
        overlaps.sort(key=lambda item: item["voxels"], reverse=True)
        best_pred = per_gt_best.get(gt_id, {}).get("best_pred_id")
        best_pred = int(best_pred) if best_pred is not None else None
        merged_with = []
        if best_pred is not None:
            merged_with = [
                int(other) for other in pred_to_gt.get(best_pred, [])
                if int(other) != int(gt_id)
            ]
        barrier_mask = gt_mask & (target == BFV3_BARRIER_LABEL)
        core_mask = gt_mask & (target == BFV3_CORE_LABEL)
        if best_pred is None or not barrier_mask.any():
            barrier_wrong = 0
        else:
            barrier_wrong = int((barrier_mask & (decoded != best_pred)).sum())
        rows.append({
            "gt_id": int(gt_id),
            "gt_voxels": int(gt_mask.sum()),
            "best_iou": float(per_gt_best.get(gt_id, {}).get("best_iou", 0.0)),
            "best_pred_id": best_pred,
            "pred_overlap_count": int(len(overlaps)),
            "top_pred_overlaps": overlaps[:5],
            "merged_with_gt_ids": merged_with,
            "barrier_voxels": int(barrier_mask.sum()),
            "barrier_assigned_outside_best_pred_voxels": barrier_wrong,
            "core_seed_components": int(ndi.label(core_mask)[1]) if core_mask.any() else 0,
        })
    low_rows = [row for row in rows if row["best_iou"] < 0.90]
    return {
        "low_iou_fragments": low_rows,
        "n_low_iou_fragments": int(len(low_rows)),
        "n_pred_fragments": int(len(pred_ids)),
        "n_gt_fragments": int(len(gt_ids)),
        "rows": rows,
    }


def binary_surface_metrics(pred: np.ndarray,
                           target: np.ndarray,
                           spacing_zyx: tuple[float, float, float],
                           nsd_tolerance_mm: float = 2.0) -> dict[str, float | None]:
    """Dice, HD95, and normalized-surface Dice for binary masks."""

    pred = pred.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    inter = int((pred & target).sum())
    denom = int(pred.sum() + target.sum())
    dice = (2.0 * inter / denom) if denom else 1.0
    if not pred.any() or not target.any():
        return {
            "dice": float(dice),
            "hd95_mm": None,
            "nsd_2mm": None,
        }
    pred_border = pred ^ ndi.binary_erosion(pred)
    target_border = target ^ ndi.binary_erosion(target)
    dist_to_target = ndi.distance_transform_edt(~target_border, sampling=spacing_zyx)
    dist_to_pred = ndi.distance_transform_edt(~pred_border, sampling=spacing_zyx)
    pred_to_target = dist_to_target[pred_border]
    target_to_pred = dist_to_pred[target_border]
    distances = np.concatenate([pred_to_target, target_to_pred])
    hd95 = float(np.percentile(distances, 95)) if distances.size else 0.0
    tol = float(nsd_tolerance_mm)
    nsd_num = int((pred_to_target <= tol).sum() + (target_to_pred <= tol).sum())
    nsd_den = int(pred_to_target.size + target_to_pred.size)
    return {
        "dice": float(dice),
        "hd95_mm": hd95,
        "nsd_2mm": float(nsd_num / max(1, nsd_den)),
    }


# =============================================================================
# 흡수된 Dataset537 BICM V5 target/decoder primitives
# [DATA][Risk:High][Scope:module_consolidation]
# Reason: 기존 bicm_v5.py의 ROI target/decoder 계약을 8개 파일 내부에 보존한다.
# label 값과 decoder rule은 평가 재현성과 직결되므로 이름과 기본값을 그대로 유지한다.
# =============================================================================
V5_LABELS = {
    "background": 0,
    "exterior_context": 1,
    "interior_shell": 2,
    "core": 3,
    "contact_surface": 4,
}

V5_SUPPORT_LABELS = (2, 3, 4)
# [METRIC][Risk:High][Scope:pengwin_official_task1_aligned_v2]
# anatomy range는 이제 anatomy_registry 단일 소스에서 파생한다. PENGWIN 2026 official
# Task 1/2 spec은 anatomy region 4개를 쓴다: Sacrum 1-50, LeftHip 51-100, RightHip
# 101-150, Femur 151-200. legacy V5_ANATOMY_RANGES는 *의도적으로* pelvic 전용(3개)으로
# 남겨, v1 proxy evaluator와의 backwards-compat를 유지한다 — registry의 pelvic_only 뷰로
# 파생하므로 femur 누락이 매직 150 뒤에 숨지 않고 명시적이다. 새 official-aligned v2
# metric/femur 경로는 반드시 femur까지 포함한 _WITH_FEMUR(=full 뷰)를 써야 한다.
V5_ANATOMY_RANGES = anatomy_ranges_by_name(pelvic_only=True)
V5_ANATOMY_RANGES_WITH_FEMUR = anatomy_ranges_by_name()
PENGWIN_OFFICIAL_ANATOMY_RANGES = V5_ANATOMY_RANGES_WITH_FEMUR
V5_TARGET_PROFILES = (
    "v5_tiny_marker",
    "v5_core_ball",
    "v5_core_body",
    "v5_core_body_contact_band",
)


@dataclass(frozen=True)
class BICMV5Params:
    """V5 의 physical-mm target parameter 정의.

    Args:
        exterior_mm: 선택된 anatomy support 주변에 두는 background 밴드 너비.
        core_mm: class-3 core marker 중심을 고를 때 non-fragment background/contact
            로부터 확보하길 선호하는 최소 거리.
        core_fallback_radius_vox: fragment 안에 `core_mm` 보다 깊은 voxel이 없을 때
            사용하는 작은 connected ball 반경.
        target_profile: `v5_tiny_marker` 는 원래 V5 oracle을 그대로 재현한다.
            `v5_core_ball` 은 fragment 당 marker component를 하나로 유지하면서도
            marker를 compact physical ball로 확장해 학습 시 core class가 한자리수
            voxel target이 되지 않도록 한다. `v5_core_body` 는 marker를 connected
            interior body로 만들어 Dice+CE에 충분한 positive support를 주면서
            fragment 당 seed component는 여전히 하나만 허용한다.
            `v5_core_body_contact_band` 는 동일한 core body를 유지하면서 support
            안쪽에서만 contact supervision을 넓힌다.
        core_ball_radius_mm: `v5_core_ball` 의 physical 반지름.
        core_body_mm: `v5_core_body` 에서 non-fragment/contact 로부터 요구하는 최소
            physical 거리. fragment가 너무 얇으면 compact ball로 fallback 한다.
        contact_band_mm: `v5_core_body_contact_band` 에서 직접 cross-fragment 인접
            영역 주변에 적용할 physical band 반경.

    [AUDIT][Risk:High][Scope:target_contract]
    이 값들은 dataset contract의 일부이므로 반드시 `dataset.json` 에 기록되어야 한다.
    값이 바뀌면 이전 oracle/training evidence는 모두 무효가 된다.
    """

    exterior_mm: float = 3.0
    core_mm: float = 8.0
    core_fallback_radius_vox: int = 1
    target_profile: Literal[
        "v5_tiny_marker",
        "v5_core_ball",
        "v5_core_body",
        "v5_core_body_contact_band",
    ] = "v5_tiny_marker"
    core_ball_radius_mm: float = 2.5
    core_body_mm: float = 3.0
    contact_band_mm: float = 2.0


def anatomy_range(anatomy: str) -> tuple[int, int]:
    """(lo, hi) global-ID block for an anatomy NAME, incl. Femur.

    Resolves against the full 4-anatomy registry view so the femur fracture path
    works; the deliberate pelvic-only restriction lives at its call sites (e.g.
    the Dataset537 raw guard), not hidden in this lookup.
    """
    ranges = V5_ANATOMY_RANGES_WITH_FEMUR
    try:
        return ranges[str(anatomy)]
    except KeyError as exc:
        raise ValueError(f"unknown anatomy {anatomy!r}; expected {sorted(ranges)}") from exc


def anatomy_mask_from_instances(inst: np.ndarray, anatomy: str) -> np.ndarray:
    lo, hi = anatomy_range(anatomy)
    return (inst >= lo) & (inst <= hi)


def bbox_from_mask(mask: np.ndarray, pad_vox: int = 24) -> tuple[slice, slice, slice] | None:
    """anatomy ROI 하나에 대해 padding을 추가한 Z/Y/X bbox를 반환한다."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    pad = int(max(0, pad_vox))
    lo = np.maximum(coords.min(axis=0) - pad, 0)
    hi = np.minimum(coords.max(axis=0) + pad + 1, np.asarray(mask.shape))
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))  # type: ignore[return-value]


def _same_anatomy_contact(inst: np.ndarray) -> np.ndarray:
    """선택된 anatomy 하나 안에서 직접 인접한 cross-fragment voxel을 표시한다.

    [QC][Invariant:label_scope]
    V5 raw sample은 이미 anatomy 하나로 크롭된 상태라 이 crop 안의 nonzero instance
    ID 쌍은 모두 same-anatomy fragment이다. shell/core와 경쟁하는 넓은 surface
    label이 생기지 않도록 6-neighbor adjacency를 사용한다.
    """
    inst = inst.astype(np.uint16, copy=False)
    out = np.zeros(inst.shape, dtype=bool)
    for axis in range(3):
        a_sl = [slice(None), slice(None), slice(None)]
        b_sl = [slice(None), slice(None), slice(None)]
        a_sl[axis] = slice(1, None)
        b_sl[axis] = slice(None, -1)
        a = inst[tuple(a_sl)]
        b = inst[tuple(b_sl)]
        hit = (a > 0) & (b > 0) & (a != b)
        if hit.any():
            oa = out[tuple(a_sl)]
            ob = out[tuple(b_sl)]
            oa[hit] = True
            ob[hit] = True
            out[tuple(a_sl)] = oa
            out[tuple(b_sl)] = ob
    return out


def _support_contact_band(contact: np.ndarray,
                          support: np.ndarray,
                          spacing_zyx: tuple[float, float, float],
                          band_mm: float) -> np.ndarray:
    """fragment support 안쪽에서만 direct contact supervision을 넓혀준다.

    [DATA][Risk:Major][Scope:contact_target]
    V5.2 에서 connected core body는 학습 가능함을 확인했지만, class-4 direct
    contact는 Dice+CE 조건에서 0에 머물렀다. 이 helper는 오로지 contact의 training
    target 밀도만 바꾼다: probability threshold를 추가하지 않고, exterior cortex를
    contact로 표시하지도 않으며, 고정 decoder contract도 그대로 둔다. 이 profile로
    학습하기 전 oracle이 반드시 통과해야 한다.
    """
    if not contact.any() or float(band_mm) <= 0:
        return contact & support
    distance_to_contact = ndi.distance_transform_edt(~contact, sampling=spacing_zyx)
    return support & (distance_to_contact <= float(band_mm))


def _single_core_marker(mask: np.ndarray, radius_vox: int) -> np.ndarray:
    """작거나 얇은 fragment 하나에 대해 connected core marker를 하나만 만들어 반환한다.

    [QC][Invariant:one_seed_per_fragment]
    V5 decoder는 class-3 connected component 각각을 seed로 다룬다. 넓은 deep-core
    class는 여러 개의 disconnected component로 쪼개져 학습 시작 전부터 가짜
    fragment를 만들 수 있으므로, target 생성 단계에서 GT fragment 당 connected
    marker를 정확히 하나만 내보낸다.
    """
    marker = np.zeros(mask.shape, dtype=bool)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return marker
    dist = ndi.distance_transform_edt(mask)
    max_dist = float(dist.max())
    if max_dist > 0:
        candidates = np.argwhere(dist == max_dist)
        centroid = coords.mean(axis=0)
        distances = np.sum((candidates.astype(np.float64) - centroid) ** 2, axis=1)
        center = candidates[int(np.argmin(distances))]
    else:
        center = coords[len(coords) // 2]
    if radius_vox <= 0:
        marker[tuple(center)] = True
        return marker
    lo = np.maximum(center - int(radius_vox), 0)
    hi = np.minimum(center + int(radius_vox) + 1, np.asarray(mask.shape))
    local_slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    local_shape = tuple(int(b - a) for a, b in zip(lo, hi))
    zz, yy, xx = np.indices(local_shape)
    dz = zz + int(lo[0]) - int(center[0])
    dy = yy + int(lo[1]) - int(center[1])
    dx = xx + int(lo[2]) - int(center[2])
    ball = (dz * dz + dy * dy + dx * dx) <= int(radius_vox) ** 2
    marker[local_slices] = ball & mask[local_slices]
    if not marker.any():
        marker[tuple(center)] = True
    return marker


def _connected_component_containing(mask: np.ndarray, center: np.ndarray) -> np.ndarray:
    """`center` 를 포함하는 connected marker component만 남긴다.

    [QC][Invariant:one_seed_per_fragment]
    physical-radius ball이 비정형 fragment에 의해 잘리면 여러 조각으로 쪼개질 수
    있다. decoder는 class-3 component 각각을 seed로 해석하므로, target 생성에서
    marker를 다시 하나의 component로 모아주지 않으면 training 또는 oracle metric이
    오해를 부른다.
    """
    out = np.zeros(mask.shape, dtype=bool)
    if not mask.any():
        return out
    comp, n_comp = ndi.label(mask)
    center_tuple = tuple(int(v) for v in center)
    comp_id = int(comp[center_tuple]) if all(0 <= center_tuple[i] < mask.shape[i] for i in range(3)) else 0
    if comp_id <= 0:
        coords = np.argwhere(mask)
        if coords.size == 0:
            return out
        nearest = coords[int(np.argmin(np.sum((coords.astype(np.float64) - center.reshape(1, 3)) ** 2, axis=1)))]
        comp_id = int(comp[tuple(nearest)])
    if n_comp > 0 and comp_id > 0:
        out[comp == comp_id] = True
    return out


def _core_ball_marker(mask: np.ndarray,
                      spacing_zyx: tuple[float, float, float],
                      radius_mm: float) -> np.ndarray:
    """fragment 하나에 대해 compact physical-mm core marker를 하나 반환한다.

    [DATA][Risk:Major][Scope:target_sparsity]
    원래 V5 target은 작은 fragment에 대해 7-voxel fallback core를 사용했다.
    full-volume overfit 실험에서 Dice+CE는 이 marker를 무너뜨리고 sparse loss는
    오히려 과도하게 칠해버리는 문제가 관찰됐다. 이 target variant는 model output
    이나 decoder threshold를 바꾸지 않으면서 positive support만 늘려준다.
    """
    marker = np.zeros(mask.shape, dtype=bool)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return marker
    dist = ndi.distance_transform_edt(mask, sampling=spacing_zyx)
    max_dist = float(dist.max())
    if max_dist > 0:
        candidates = np.argwhere(dist == max_dist)
        centroid = coords.mean(axis=0)
        center = candidates[int(np.argmin(np.sum((candidates.astype(np.float64) - centroid) ** 2, axis=1)))]
    else:
        center = coords[len(coords) // 2]
    spacing = np.asarray(spacing_zyx, dtype=np.float64).reshape(3, 1)
    delta_mm = (coords.astype(np.float64) - center.reshape(1, 3).astype(np.float64)).T * spacing
    inside = np.sum(delta_mm * delta_mm, axis=0) <= float(radius_mm) ** 2
    marker[tuple(coords[inside].T)] = True
    marker = _connected_component_containing(marker & mask, center)
    if not marker.any():
        marker[tuple(center)] = True
    return marker


def _core_body_marker(mask: np.ndarray,
                      spacing_zyx: tuple[float, float, float],
                      body_mm: float,
                      fallback_radius_mm: float) -> np.ndarray:
    """fragment 하나에 대해 connected interior core body를 하나 반환한다.

    [DATA][Risk:Major][Scope:target_geometry]
    V5.1 에서 양극단의 실패가 함께 드러났다. tiny/ball core는 Dice+CE 에서 사라지고,
    sparse rare-class loss는 core를 수천 개 seed component로 과도하게 칠해버렸다.
    이 target은 BICM semantic contract를 그대로 유지하면서 core를 더 큰 connected
    interior body로 만든다. boundary-band specialist target과 비슷한 형태라서
    decoder는 여전히 GT fragment 당 marker component를 하나만 받게 된다.
    """
    marker = np.zeros(mask.shape, dtype=bool)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return marker
    dist = ndi.distance_transform_edt(mask, sampling=spacing_zyx)
    max_dist = float(dist.max())
    candidates = np.argwhere(dist == max_dist)
    centroid = coords.mean(axis=0)
    center = candidates[int(np.argmin(np.sum((candidates.astype(np.float64) - centroid) ** 2, axis=1)))]
    if max_dist >= float(body_mm):
        body = mask & (dist >= float(body_mm))
        marker = _connected_component_containing(body, center)
    if not marker.any():
        marker = _core_ball_marker(mask, spacing_zyx, fallback_radius_mm)
    return marker


def compute_bicm_v5_target(inst_roi: np.ndarray,
                           spacing_zyx: tuple[float, float, float],
                           params: BICMV5Params = BICMV5Params()) -> np.ndarray:
    """anatomy ROI 하나에 대한 V5 BICM target을 계산한다.

    Args:
        inst_roi: 한 anatomy에 속한 instance ID만 담은 array. non-anatomy ID는
            preprocessing 단계에서 미리 0으로 만들어 둬야 한다.
        spacing_zyx: array 순서(zyx) physical voxel spacing.
        params: 고정 V5 target geometry.

    Returns:
        `V5_LABELS` 의 class 값을 담은 uint8 label map.

    Raises:
        ValueError: spacing이 유효하지 않거나 nonzero fragment에 core marker가
            할당되지 못한 경우.
    """
    if len(spacing_zyx) != 3 or any(float(v) <= 0 for v in spacing_zyx):
        raise ValueError(f"invalid spacing_zyx: {spacing_zyx!r}")
    if params.target_profile not in V5_TARGET_PROFILES:
        raise ValueError(f"unknown V5 target_profile={params.target_profile!r}; expected {V5_TARGET_PROFILES}")
    inst = inst_roi.astype(np.uint16, copy=False)
    support = inst > 0
    target = np.zeros(inst.shape, dtype=np.uint8)
    if not support.any():
        return target

    contact = _same_anatomy_contact(inst)
    if params.target_profile == "v5_core_body_contact_band":
        contact = _support_contact_band(contact, support, spacing_zyx, params.contact_band_mm)
    exterior = ndi.distance_transform_edt(~support, sampling=spacing_zyx) <= float(params.exterior_mm)
    target[exterior & ~support] = V5_LABELS["exterior_context"]
    target[support] = V5_LABELS["interior_shell"]
    target[contact] = V5_LABELS["contact_surface"]

    for fragment_id in [int(v) for v in np.unique(inst) if int(v) > 0]:
        frag = inst == fragment_id
        if not frag.any():
            continue
        seed_mask = frag & ~contact
        dist = ndi.distance_transform_edt(seed_mask, sampling=spacing_zyx)
        if params.target_profile == "v5_core_ball":
            core = _core_ball_marker(seed_mask, spacing_zyx, float(params.core_ball_radius_mm))
            if not core.any():
                core = _core_ball_marker(frag, spacing_zyx, float(params.core_ball_radius_mm))
        elif params.target_profile in {"v5_core_body", "v5_core_body_contact_band"}:
            core = _core_body_marker(
                seed_mask,
                spacing_zyx,
                body_mm=float(params.core_body_mm),
                fallback_radius_mm=float(params.core_ball_radius_mm),
            )
            if not core.any():
                core = _core_body_marker(
                    frag,
                    spacing_zyx,
                    body_mm=float(params.core_body_mm),
                    fallback_radius_mm=float(params.core_ball_radius_mm),
                )
        elif float(dist.max()) >= float(params.core_mm):
            candidates = np.argwhere(dist == dist.max())
            center = candidates[0]
            center_mask = np.zeros(inst.shape, dtype=bool)
            center_mask[tuple(center)] = True
            # marker helper의 적용 범위를 max-distance voxel 주변의 작은 ball로
            # 제한한다. 그래야 seed가 connected 상태로 유지되고, 예전 "deep voxel은
            # 전부 core" 식의 target이 수십 개의 seed를 만들어내는 문제를 막을 수 있다.
            core = _single_core_marker(
                seed_mask & ndi.binary_dilation(center_mask, iterations=int(params.core_fallback_radius_vox)),
                int(params.core_fallback_radius_vox),
            )
        else:
            core = _single_core_marker(seed_mask, int(params.core_fallback_radius_vox))
            if not core.any():
                core = _single_core_marker(frag, int(params.core_fallback_radius_vox))
        if not core.any():
            raise ValueError(f"fragment {fragment_id} has no V5 core marker")
        target[core] = V5_LABELS["core"]
    return target


def decode_bicm_v5(labels: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """고정된 watershed로 V5 BICM label을 local instance ID로 decode 한다."""
    from skimage.segmentation import watershed

    lab = labels.astype(np.uint8, copy=False)
    support = np.isin(lab, V5_SUPPORT_LABELS)
    core = lab == V5_LABELS["core"]
    contact = lab == V5_LABELS["contact_surface"]
    seed_cc, n_seed = ndi.label(core)
    if not support.any() or n_seed == 0:
        return np.zeros(lab.shape, dtype=np.uint16), {
            "decoder": "bicm_v5_core_watershed",
            "support_voxels": int(support.sum()),
            "seed_components": int(n_seed),
            "missing_seed_component": [{"component": 1, "voxels": int(support.sum())}] if support.any() else [],
        }

    # [METRIC][Scope:fixed_decoder]
    # contact class는 ridge cost 역할이지 foreground를 지우는 용도가 아니다. 이렇게
    # 하면 실제 fragment support는 보존하면서 fracture contact surface를 가로지르는
    # marker assignment에 페널티만 줄 수 있다. 이 decoder에는 threshold search가 없다.
    energy = np.zeros(lab.shape, dtype=np.float32)
    energy[lab == V5_LABELS["interior_shell"]] = 1.0
    energy[contact] = 12.0
    energy[core] = 0.0
    local = watershed(energy, seed_cc.astype(np.int32), mask=support, connectivity=1)
    decoded = local.astype(np.uint16, copy=False)
    support_cc, n_support = ndi.label(support)
    missing = []
    for comp_id in range(1, n_support + 1):
        comp = support_cc == comp_id
        if not (seed_cc[comp] > 0).any():
            missing.append({"component": int(comp_id), "voxels": int(comp.sum())})
    return decoded, {
        "decoder": "bicm_v5_core_watershed",
        "support_voxels": int(support.sum()),
        "contact_voxels": int(contact.sum()),
        "seed_components": int(n_seed),
        "missing_seed_component": missing,
        "pred_fragment_count": int(len([v for v in np.unique(decoded) if int(v) > 0])),
    }


def decode_bicm_v5_seed_healed(labels: np.ndarray,
                               min_core_component_voxels: int = 128,
                               core_closing_iters: int = 2) -> tuple[np.ndarray, dict[str, Any]]:
    """watershed 직전에 결정론적 core seed healing을 거친 뒤 decode 한다.

    [AUDIT][Risk:High][Scope:decoder_contract]
    이 decoder는 진단용 V5.4 한정이다. GT label을 쓰지도 않고 validation data로
    threshold를 튜닝하지도 않는다. 규칙은 "true fragment 하나는 fragment 크기에
    맞는 core marker를 하나 가진다"는 oracle invariant에서 곧장 가져온 것이며,
    V5.2/V5.3 예측에서는 수백 개의 작은 core component가 만들어지는 문제가 있었다.

    [QC][Invariant:no_gt_dependency]
    입력은 예측된 semantic label뿐이다. 이걸로 fragment 개수는 좋아졌는데 contact는
    여전히 0이라면, 남은 병목은 seed postprocessing이 아니라 contact 학습 자체다.
    """
    lab = labels.astype(np.uint8, copy=False)
    support = np.isin(lab, V5_SUPPORT_LABELS)
    contact = lab == V5_LABELS["contact_surface"]
    core = lab == V5_LABELS["core"]
    if not support.any() or not core.any():
        return decode_bicm_v5(labels)

    structure = ndi.generate_binary_structure(3, 1)
    healed_core = ndi.binary_closing(
        core & support & ~contact,
        structure=structure,
        iterations=int(max(0, core_closing_iters)),
    )
    healed_core &= support & ~contact
    comp, n_comp = ndi.label(healed_core)
    keep = np.zeros(lab.shape, dtype=bool)
    component_sizes: list[int] = []
    for comp_id in range(1, n_comp + 1):
        size = int((comp == comp_id).sum())
        component_sizes.append(size)
        if size >= int(min_core_component_voxels):
            keep[comp == comp_id] = True
    if not keep.any():
        raw_comp, raw_n = ndi.label(core & support & ~contact)
        if raw_n > 0:
            sizes = [(int((raw_comp == cid).sum()), int(cid)) for cid in range(1, raw_n + 1)]
            _size, cid = max(sizes)
            keep[raw_comp == cid] = True

    healed_labels = lab.copy()
    raw_core = healed_labels == V5_LABELS["core"]
    healed_labels[raw_core] = V5_LABELS["interior_shell"]
    healed_labels[keep] = V5_LABELS["core"]
    decoded, trace = decode_bicm_v5(healed_labels)
    trace = dict(trace)
    trace.update({
        "decoder": "bicm_v5_seed_healed",
        "raw_core_components": int(ndi.label(core)[1]),
        "healed_core_components": int(ndi.label(keep)[1]),
        "min_core_component_voxels": int(min_core_component_voxels),
        "core_closing_iters": int(core_closing_iters),
        "removed_or_merged_core_components": int(max(0, int(ndi.label(core)[1]) - int(ndi.label(keep)[1]))),
    })
    return decoded, trace


def label_distribution(labels: np.ndarray) -> dict[str, int]:
    """audit report용 JSON-safe class voxel count를 반환한다."""
    counts = {name: int((labels == value).sum()) for name, value in V5_LABELS.items()}
    counts["support"] = int(np.isin(labels, V5_SUPPORT_LABELS).sum())
    return counts


# =============================================================================
# 흡수된 official ABBC target/decoder primitives
# [AUDIT][Risk:High][Scope:module_consolidation]
# Reason: 기존 abbc_official.py의 official target/decoder helper를 utils.py로 이동한다.
# 이 경로는 현재 V247 실험의 주 경로는 아니지만 preprocessing/eval legacy command가
# import 실패 없이 재현되도록 필요한 계약을 보존한다.
# =============================================================================
ABBC_OFFICIAL_LABELS = {
    "background": 0,
    "boundary": 1,
    "core": 2,
    "contact_surface": 3,
}
ABBC_BOUNDARY_LABEL = 1
ABBC_CORE_LABEL = 2
ABBC_BORDER_LABEL = 3
ABBC_CONTACT_LABEL = ABBC_BORDER_LABEL
ABBC_DIVERGENCE_THRESHOLD = 0.11
ABBC_DISTANCE_THRESHOLD_VOX = 6.0
ABBC_FRACTURE_DISK_RADIUS = 6
ABBC_CONTACT_RADIUS_VOX = 3
ABBC_CORE_CLEARANCE_RADIUS_DELTA = 2

# Registry-derived decode tables. The "PELVIC" prefix is historical: these now
# span ALL anatomies incl. Femur so the watershed->global-ID reconstruction does
# not silently drop femur. Safe superset — a pelvic-only model never predicts the
# femur semantic class, so its femur entry stays unused.
PELVIC_ANATOMY_RANGES = {a.index: (a.name, a.id_lo, a.id_hi) for a in ANATOMY_REGISTRY}
PELVIC_GLOBAL_START = anatomy_start_ids()


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    if cc3d is not None:
        labeled = cc3d.connected_components(mask > 0).astype(np.uint16, copy=False)
        return labeled, int(labeled.max())
    return ndi.label(mask > 0, structure=np.ones((3, 3, 3), dtype=np.uint8))


def _label_components_by_value(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected components without merging adjacent different label values."""
    out = np.zeros(values.shape, dtype=np.uint16)
    next_id = 1
    for value in [int(v) for v in np.unique(values) if int(v) > 0]:
        cc, n_cc = _label_components(values == value)
        for cc_id in range(1, n_cc + 1):
            out[cc == cc_id] = next_id
            next_id += 1
    return out, next_id - 1


def _mask_bbox(mask: np.ndarray, pad: int = 0) -> tuple[slice, slice, slice] | None:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    mins = np.maximum(coords.min(axis=0) - int(pad), 0)
    maxs = np.minimum(coords.max(axis=0) + int(pad) + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(mins, maxs))


def _bbox_from_slices(slices: tuple[slice, ...]) -> list[list[int]]:
    return [[int(s.start), int(s.stop)] for s in slices]


def _slices_from_bbox(bbox: list[list[int]] | tuple[tuple[int, int], ...]) -> tuple[slice, ...]:
    return tuple(slice(int(lo), int(hi)) for lo, hi in bbox)


def _distance_assign(seed_cc: np.ndarray,
                     support: np.ndarray,
                     use_fmm: bool = True) -> np.ndarray:
    """Assign support voxels to the nearest seed, using FMM when available."""
    out = np.zeros(seed_cc.shape, dtype=np.uint16)
    seed_ids = [int(v) for v in np.unique(seed_cc) if int(v) > 0]
    if not seed_ids or not support.any():
        return out
    if use_fmm and skfmm is not None:
        min_dist = np.full(seed_cc.shape, np.inf, dtype=np.float32)
        mask = ~support
        for seed_id in seed_ids:
            border = np.ones(seed_cc.shape, dtype=bool)
            border[seed_cc == seed_id] = False
            try:
                dist = skfmm.distance(np.ma.MaskedArray(border, mask))
            except Exception:
                return _distance_assign(seed_cc, support, use_fmm=False)
            dist_data = np.asarray(dist.data, dtype=np.float32)
            if isinstance(dist, np.ma.MaskedArray):
                dist_data[dist.mask] = np.inf
            update = support & (dist_data < min_dist)
            out[update] = seed_id
            min_dist = np.minimum(min_dist, dist_data)
        return out

    nearest = ndi.distance_transform_edt(
        seed_cc == 0,
        return_distances=False,
        return_indices=True,
    )
    assigned = seed_cc[tuple(nearest)]
    out[support] = assigned[support]
    return out


def get_fracture_regions(instances: np.ndarray,
                         disk_radius: int = ABBC_FRACTURE_DISK_RADIUS,
                         groups: dict[int, list[int]] | None = None) -> np.ndarray:
    """Return voxels where same-anatomy fragment shells overlap."""
    if groups is None:
        groups = {
            anatomy_id: [
                int(v) for v in np.unique(instances)
                if lo <= int(v) <= hi
            ]
            for anatomy_id, (_name, lo, hi) in PELVIC_ANATOMY_RANGES.items()
        }
    footprint = ball(max(1, int(disk_radius) // 2))
    out = np.zeros(instances.shape, dtype=np.uint8)
    for ids in groups.values():
        ids = [int(v) for v in ids if int(v) > 0]
        if len(ids) <= 1:
            continue
        overlap = np.zeros(instances.shape, dtype=np.uint8)
        for fragment_id in ids:
            bbox = _mask_bbox(instances == fragment_id, pad=int(disk_radius))
            if bbox is None:
                continue
            frag = instances[bbox] == fragment_id
            if not frag.any():
                continue
            outer = ndi.binary_dilation(frag, structure=footprint)
            inner = ndi.binary_erosion(frag, structure=footprint, border_value=0)
            overlap_crop = overlap[bbox]
            overlap_crop[(outer ^ inner)] += 1
            overlap[bbox] = overlap_crop
        out[overlap > 1] = 1
    return out


def get_contact_surface_regions(instances: np.ndarray,
                                contact_radius: int = ABBC_CONTACT_RADIUS_VOX,
                                surface_erosion_radius: int = 1,
                                groups: dict[int, list[int]] | None = None) -> np.ndarray:
    """Return narrow same-anatomy contact surfaces inside fragment voxels.

    The previous broad border target marked overlapping dilated outlines, which
    produced high recall but too many false positive separator voxels. Contact V2
    marks only fragment surface voxels that are close to another fragment from
    the same anatomy. That keeps class 3 usable as a decoder energy ridge.
    """
    if groups is None:
        groups = {
            anatomy_id: [
                int(v) for v in np.unique(instances)
                if lo <= int(v) <= hi
            ]
            for anatomy_id, (_name, lo, hi) in PELVIC_ANATOMY_RANGES.items()
        }
    contact_radius = max(1, int(contact_radius))
    surface_erosion_radius = max(1, int(surface_erosion_radius))
    near_footprint = ball(contact_radius)
    erode_footprint = ball(surface_erosion_radius)
    out = np.zeros(instances.shape, dtype=np.uint8)
    for ids in groups.values():
        ids = [int(v) for v in ids if int(v) > 0]
        if len(ids) <= 1:
            continue
        same_anatomy = np.isin(instances, ids)
        for fragment_id in ids:
            bbox = _mask_bbox(instances == fragment_id, pad=contact_radius + 2)
            if bbox is None:
                continue
            frag = instances[bbox] == fragment_id
            if not frag.any():
                continue
            others = same_anatomy[bbox] & ~frag
            if not others.any():
                continue
            surface = frag & ~ndi.binary_erosion(frag, structure=erode_footprint, border_value=0)
            near_other = ndi.binary_dilation(others, structure=near_footprint)
            out_crop = out[bbox]
            out_crop[surface & near_other] = 1
            out[bbox] = out_crop
    return out


def compute_abbc_official_target(instances: np.ndarray,
                                 divergence_threshold: float = ABBC_DIVERGENCE_THRESHOLD,
                                 distance_threshold: float = ABBC_DISTANCE_THRESHOLD_VOX,
                                 fracture_disk_radius: int = ABBC_FRACTURE_DISK_RADIUS,
                                 enforce_one_core_cc: bool = True) -> np.ndarray:
    """Convert original pelvic fragment IDs to official ABBC classes.

    The Contact V2 target keeps the official IDs but narrows class 3 to
    same-anatomy contact/fracture surface voxels. Class 1 remains assignable
    shell/body support and class 2 provides marker seeds. `enforce_one_core_cc`
    guards tiny fragments before decoder-side small-core filtering is evaluated.
    """
    pelvic = np.zeros_like(instances, dtype=np.uint16)
    for _anatomy_id, (_name, lo, hi) in PELVIC_ANATOMY_RANGES.items():
        mask = (instances >= lo) & (instances <= hi)
        pelvic[mask] = instances[mask].astype(np.uint16, copy=False)

    target = np.zeros(instances.shape, dtype=np.uint8)
    fragment_ids = [int(v) for v in np.unique(pelvic) if int(v) > 0]
    for fragment_id in fragment_ids:
        bbox = _mask_bbox(pelvic == fragment_id, pad=20)
        if bbox is None:
            continue
        frag = pelvic[bbox] == fragment_id
        dist = ndi.distance_transform_edt(frag)
        dist = ndi.gaussian_filter(dist, sigma=2)
        gradients = [
            np.gradient(np.gradient(dist, axis=axis), axis=axis)
            for axis in range(1, instances.ndim)
        ]
        div = np.sum(np.stack(gradients, axis=0), axis=0) if gradients else np.zeros_like(dist)
        core = (-div > float(divergence_threshold)) & frag
        core[dist > float(distance_threshold)] = True

        if enforce_one_core_cc:
            core = core & frag
            if not core.any() and frag.any():
                core[np.unravel_index(int(np.argmax(dist)), dist.shape)] = True
            core_cc, n_core = _label_components(core)
            if n_core > 1:
                sizes = ndi.sum(core, core_cc, index=np.arange(1, n_core + 1))
                keep = int(np.argmax(sizes)) + 1
                core = core_cc == keep

        crop = target[bbox]
        crop[frag] = ABBC_BOUNDARY_LABEL
        crop[core & frag] = ABBC_CORE_LABEL
        target[bbox] = crop

    fracture_clearance = get_contact_surface_regions(
        pelvic,
        contact_radius=max(1, int(fracture_disk_radius) // 2) + ABBC_CORE_CLEARANCE_RADIUS_DELTA,
    )
    target[(fracture_clearance > 0) & (target == ABBC_CORE_LABEL)] = ABBC_BOUNDARY_LABEL
    fractures = get_contact_surface_regions(
        pelvic,
        contact_radius=max(1, int(fracture_disk_radius) // 2),
    )
    target[fractures > 0] = ABBC_BORDER_LABEL
    target[pelvic == 0] = 0
    if enforce_one_core_cc:
        for fragment_id in fragment_ids:
            frag = pelvic == fragment_id
            core = frag & (target == ABBC_CORE_LABEL)
            bbox = _mask_bbox(frag, pad=1)
            if bbox is None:
                continue
            core_cc, n_core = _label_components(core[bbox])
            target_crop = target[bbox]
            if n_core == 1:
                target[bbox] = target_crop
                continue
            target_crop[core[bbox]] = ABBC_BOUNDARY_LABEL
            if n_core > 1:
                sizes = ndi.sum(core[bbox], core_cc, index=np.arange(1, n_core + 1))
                keep = int(np.argmax(sizes)) + 1
                target_crop[core_cc == keep] = ABBC_CORE_LABEL
            else:
                candidate = frag[bbox] & (target_crop != ABBC_BORDER_LABEL)
                if not candidate.any():
                    candidate = frag[bbox]
                dist = ndi.distance_transform_edt(candidate)
                target_crop[np.unravel_index(int(np.argmax(dist)), dist.shape)] = ABBC_CORE_LABEL
            target[bbox] = target_crop
    return target


def compute_abbc_official_target_dynamic(
    instances: np.ndarray,
    divergence_threshold: float = ABBC_DIVERGENCE_THRESHOLD,
    distance_threshold: float = ABBC_DISTANCE_THRESHOLD_VOX,
    min_disk_radius: int = 2,
    max_disk_radius: int = 12,
    target_boundary_thickness_mm: float = 6.0,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    enforce_one_core_cc: bool = True,
) -> tuple[np.ndarray, dict]:
    """V0.x [PHASE 1] Dynamic boundary thickness ABBC target (MIC-DKFZ winner 방식).

    PENGWIN 2024 1위 (MIC-DKFZ) 의 핵심 차별화 기법 — 모든 fragment 에 fixed disk_radius=6 을
    적용하면 thin fragment 가 거의 전부 boundary 로 흡수되어 core 가 사라지고, 반대로 thick
    fragment 는 0.03% sparse contact 로 학습 불가. 본 함수는 fragment 별 medial axis transform
    으로 thickness 를 측정한 뒤 dynamic 하게 disk_radius 를 조정한다.

    인수:
        instances: GT instance ID 볼륨 (uint16, 0=bg).
        min_disk_radius / max_disk_radius: dynamic radius 의 하한/상한.
        target_boundary_thickness_mm: 모든 fragment 가 도달하기 원하는 boundary 두께 (mm).
        spacing_zyx: voxel spacing (mm). medial axis 거리 mm 단위 변환에 사용.

    반환:
        target: 4-class ABBC (background/boundary/core/contact_surface).
        audit: per-fragment dynamic_radius + thickness 기록 dict.

    구현 노트:
        - fragment 당 medial axis percentile 95 를 thickness 의 mm 추정값으로 사용.
        - dynamic_radius = clamp(target_boundary_thickness_mm / thickness_mm, min, max)
        - thickness 가 작을수록 (얇은 조각) radius ↑ → boundary 가 sparse 되지 않음.
        - thickness 가 클수록 (두꺼운 조각) radius ↓ → core 가 보존됨.
    """
    pelvic = np.zeros_like(instances, dtype=np.uint16)
    for _anatomy_id, (_name, lo, hi) in PELVIC_ANATOMY_RANGES.items():
        mask = (instances >= lo) & (instances <= hi)
        pelvic[mask] = instances[mask].astype(np.uint16, copy=False)

    fragment_ids = [int(v) for v in np.unique(pelvic) if int(v) > 0]
    audit_per_fragment = []
    target = np.zeros(instances.shape, dtype=np.uint8)

    # Step 1: per-fragment dynamic radius 계산 (medial axis 기반).
    spacing_arr = np.array(spacing_zyx, dtype=np.float32)
    avg_spacing_mm = float(np.mean(spacing_arr))
    dynamic_radius_per_fragment: dict[int, int] = {}
    for fragment_id in fragment_ids:
        bbox = _mask_bbox(pelvic == fragment_id, pad=20)
        if bbox is None:
            continue
        frag = pelvic[bbox] == fragment_id
        # Distance transform 으로 medial axis thickness 추정.
        # sampling 인자를 spacing 으로 주면 mm 단위 거리.
        dist_mm = ndi.distance_transform_edt(frag, sampling=spacing_arr)
        if dist_mm[frag].size == 0:
            thickness_mm = 1.0
        else:
            thickness_mm = float(np.percentile(dist_mm[frag], 95))
        thickness_mm = max(thickness_mm, 0.5)

        # Dynamic radius: target thickness / measured thickness.
        # thin fragment (작은 thickness) → 큰 radius.
        # thick fragment (큰 thickness) → 작은 radius (default 6 근처).
        ratio = target_boundary_thickness_mm / thickness_mm
        dynamic_r_vox = int(round(ratio * (6.0 / avg_spacing_mm)))
        dynamic_r_vox = max(min_disk_radius, min(max_disk_radius, dynamic_r_vox))
        dynamic_radius_per_fragment[fragment_id] = dynamic_r_vox
        audit_per_fragment.append({
            "fragment_id": fragment_id,
            "thickness_mm_p95": round(thickness_mm, 2),
            "dynamic_radius_vox": dynamic_r_vox,
            "fragment_size": int(frag.sum()),
        })

    # Step 2: per-fragment boundary/core 생성 (기존 로직 유지).
    for fragment_id in fragment_ids:
        bbox = _mask_bbox(pelvic == fragment_id, pad=20)
        if bbox is None:
            continue
        frag = pelvic[bbox] == fragment_id
        dist = ndi.distance_transform_edt(frag)
        dist = ndi.gaussian_filter(dist, sigma=2)
        gradients = [
            np.gradient(np.gradient(dist, axis=axis), axis=axis)
            for axis in range(1, instances.ndim)
        ]
        div = np.sum(np.stack(gradients, axis=0), axis=0) if gradients else np.zeros_like(dist)
        core = (-div > float(divergence_threshold)) & frag
        core[dist > float(distance_threshold)] = True

        if enforce_one_core_cc:
            core = core & frag
            if not core.any() and frag.any():
                core[np.unravel_index(int(np.argmax(dist)), dist.shape)] = True
            core_cc, n_core = _label_components(core)
            if n_core > 1:
                sizes = ndi.sum(core, core_cc, index=np.arange(1, n_core + 1))
                keep = int(np.argmax(sizes)) + 1
                core = core_cc == keep

        crop = target[bbox]
        crop[frag] = ABBC_BOUNDARY_LABEL
        crop[core & frag] = ABBC_CORE_LABEL
        target[bbox] = crop

    # Step 3: contact surface — fragment 별 dynamic radius 사용.
    # 기존 fixed 와 달리, 인접 fragment 쌍 별 max(radius) 적용.
    for fragment_id in fragment_ids:
        r = dynamic_radius_per_fragment.get(fragment_id, 6)
        # contact_radius 는 cross-fragment 접촉면 영역. dynamic.
        contact_r = max(1, r // 2)
        clear_r = contact_r + ABBC_CORE_CLEARANCE_RADIUS_DELTA
        # 단일 fragment 만 isolate 해서 contact 추출.
        single = np.where(pelvic == fragment_id, pelvic, 0)
        # 인접 fragments 와의 contact 만 추출하려면 전체 pelvic 필요.
        # 따라서 본 함수는 simple per-fragment 가 아닌 global pass 로 처리.

    # [V0.x][PHASE 3-B][2026-06-01] Contact radius 확장.
    # 이전: contact_radius = max_r // 2  (max_r=12 시 contact_r=6, 0.05% sparse)
    # 현재: contact_radius = max_r * 0.8 (max_r=12 시 contact_r=9-10, ~0.15% 예상)
    # 동기: within-anatomy contact 의 0.034% sparse class collapse 직접 해결.
    # diag6 결과 cross-anatomy 는 2.54% 로 무의미하나, within-anatomy 의 sparse class
    # 학습 어려움이 V0~V72 의 지속된 bottleneck. radius 1.6x 확장으로 contact 영역
    # 단면적 2.5x 증가 → V300 BADB 의 boundary refinement 와 시너지.
    if dynamic_radius_per_fragment:
        max_r = max(dynamic_radius_per_fragment.values())
    else:
        max_r = 6
    expanded_contact_r = max(2, int(round(max_r * 0.8)))  # Phase 3-B 핵심 변경
    fracture_clearance = get_contact_surface_regions(
        pelvic, contact_radius=expanded_contact_r + ABBC_CORE_CLEARANCE_RADIUS_DELTA,
    )
    target[(fracture_clearance > 0) & (target == ABBC_CORE_LABEL)] = ABBC_BOUNDARY_LABEL
    fractures = get_contact_surface_regions(
        pelvic, contact_radius=expanded_contact_r,
    )
    target[fractures > 0] = ABBC_BORDER_LABEL
    target[pelvic == 0] = 0

    if enforce_one_core_cc:
        for fragment_id in fragment_ids:
            frag = pelvic == fragment_id
            core = frag & (target == ABBC_CORE_LABEL)
            bbox = _mask_bbox(frag, pad=1)
            if bbox is None:
                continue
            core_cc, n_core = _label_components(core[bbox])
            target_crop = target[bbox]
            if n_core == 1:
                target[bbox] = target_crop
                continue
            target_crop[core[bbox]] = ABBC_BOUNDARY_LABEL
            if n_core > 1:
                sizes = ndi.sum(core[bbox], core_cc, index=np.arange(1, n_core + 1))
                keep = int(np.argmax(sizes)) + 1
                target_crop[core_cc == keep] = ABBC_CORE_LABEL
            else:
                candidate = frag[bbox] & (target_crop != ABBC_BORDER_LABEL)
                if not candidate.any():
                    candidate = frag[bbox]
                dist = ndi.distance_transform_edt(candidate)
                target_crop[np.unravel_index(int(np.argmax(dist)), dist.shape)] = ABBC_CORE_LABEL
            target[bbox] = target_crop

    audit = {
        "n_fragments": len(fragment_ids),
        "max_dynamic_radius": max_r if dynamic_radius_per_fragment else 0,
        "min_dynamic_radius": min(dynamic_radius_per_fragment.values()) if dynamic_radius_per_fragment else 0,
        "mean_dynamic_radius": float(np.mean(list(dynamic_radius_per_fragment.values()))) if dynamic_radius_per_fragment else 0,
        "per_fragment": audit_per_fragment,
    }
    return target, audit


def abbc_component_to_instances(patch: np.ndarray,
                                min_core_size: int = 50) -> np.ndarray:
    """Convert one connected ABBC component to local instances."""
    original = patch.copy()
    support_no_border = (patch > 0) & (patch != ABBC_BORDER_LABEL)
    core_cc, n_core = _label_components(patch == ABBC_CORE_LABEL)
    if n_core == 0:
        return np.zeros_like(patch, dtype=np.uint16)

    remove_ids = []
    for seed_id in range(1, n_core + 1):
        if int((core_cc == seed_id).sum()) < int(min_core_size):
            remove_ids.append(seed_id)
    if remove_ids:
        remove_mask = np.isin(core_cc, remove_ids)
        patch = patch.copy()
        patch[remove_mask] = ABBC_BOUNDARY_LABEL
        core_cc, n_core = _label_components(patch == ABBC_CORE_LABEL)
        support_no_border = (patch > 0) & (patch != ABBC_BORDER_LABEL)
    if n_core == 0:
        return np.zeros_like(patch, dtype=np.uint16)

    local = _distance_assign(core_cc, support_no_border)
    if n_core > 1:
        local = _distance_assign(local, original > 0)
    return local.astype(np.uint16, copy=False)


def abbc_to_instance(abbc: np.ndarray,
                     min_core_size: int = 50) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert official ABBC prediction to unlabeled local instances."""
    comp_cc, n_components = _label_components(abbc > 0)
    instances = np.zeros(abbc.shape, dtype=np.uint16)
    next_id = 1
    rows = []
    for comp_id in range(1, n_components + 1):
        bbox = _mask_bbox(comp_cc == comp_id, pad=0)
        if bbox is None:
            continue
        patch_mask = comp_cc[bbox] == comp_id
        patch = abbc[bbox].copy()
        patch[~patch_mask] = 0
        local = abbc_component_to_instances(patch, min_core_size=min_core_size)
        local_ids = [int(v) for v in np.unique(local) if int(v) > 0]
        for local_id in local_ids:
            mask = local == local_id
            instances_crop = instances[bbox]
            instances_crop[mask] = next_id
            instances[bbox] = instances_crop
            next_id += 1
        rows.append({
            "component": int(comp_id),
            "bbox": _bbox_from_slices(bbox),
            "input_voxels": int(patch_mask.sum()),
            "core_components": int(_label_components(patch == ABBC_CORE_LABEL)[1]),
            "assigned_instances": int(len(local_ids)),
        })
    return instances, {
        "component_count": int(n_components),
        "assigned_instances": int(next_id - 1),
        "components": rows,
    }


def _anatomy_ids_for_instances(semantic: np.ndarray,
                               instances: np.ndarray) -> dict[int, list[int]]:
    out = {1: [], 2: [], 3: []}
    for instance_id in [int(v) for v in np.unique(instances) if int(v) > 0]:
        vox = semantic[instances == instance_id]
        vox = vox[(vox >= 1) & (vox <= 3)]
        if len(vox) == 0:
            continue
        anatomy_id = int(np.bincount(vox.astype(np.int64)).argmax())
        out.setdefault(anatomy_id, []).append(instance_id)
    return out


def remap_instance_labels(semantic: np.ndarray,
                          instances: np.ndarray,
                          min_size: int = 1000) -> tuple[np.ndarray, dict[str, Any]]:
    """Split by semantic anatomy, merge tiny same-anatomy fragments, assign IDs."""
    combined = (semantic.astype(np.uint32) * 100000 + instances.astype(np.uint32))
    combined[(semantic == 0) | (instances == 0)] = 0
    split_cc, _ = _label_components_by_value(combined)
    split_cc = split_cc.astype(np.uint16, copy=False)
    anatomy_ids = _anatomy_ids_for_instances(semantic, split_cc)

    small_merges = []
    footprint_small = ball(3)
    footprint_big = ball(10)
    labels, counts = np.unique(split_cc, return_counts=True)
    for label, count in zip(labels, counts):
        label = int(label)
        if label == 0 or int(count) > int(min_size):
            continue
        vox = semantic[split_cc == label]
        vox = vox[(vox >= 1) & (vox <= 3)]
        if len(vox) == 0:
            split_cc[split_cc == label] = 0
            continue
        anatomy_id = int(np.bincount(vox.astype(np.int64)).argmax())
        bbox = _mask_bbox(split_cc == label, pad=15)
        if bbox is None:
            continue
        sub = split_cc[bbox]
        touching_label = None
        for footprint in (footprint_small, footprint_big):
            dilated = ndi.binary_dilation(sub == label, structure=footprint)
            candidates, candidate_counts = np.unique(sub[dilated & (sub != label) & (sub > 0)], return_counts=True)
            same = [
                (int(c), int(n)) for c, n in zip(candidates, candidate_counts)
                if int(c) in anatomy_ids.get(anatomy_id, [])
            ]
            if same:
                touching_label = max(same, key=lambda item: item[1])[0]
                break
        if touching_label is None:
            continue
        split_cc[split_cc == label] = touching_label
        small_merges.append({
            "source": int(label),
            "target": int(touching_label),
            "anatomy": int(anatomy_id),
            "voxels": int(count),
        })

    remapped = np.zeros_like(split_cc, dtype=np.uint16)
    next_by_anatomy = dict(PELVIC_GLOBAL_START)
    assigned = []
    for anatomy_id, _row in PELVIC_ANATOMY_RANGES.items():
        instance_ids = [
            int(v) for v in np.unique(split_cc[(semantic == anatomy_id) & (split_cc > 0)])
            if int(v) > 0
        ]
        instance_ids.sort(key=lambda v: int((split_cc == v).sum()), reverse=True)
        for instance_id in instance_ids:
            global_id = next_by_anatomy[anatomy_id]
            remapped[(split_cc == instance_id) & (semantic == anatomy_id)] = global_id
            assigned.append({
                "local_instance": int(instance_id),
                "global_id": int(global_id),
                "anatomy": int(anatomy_id),
                "voxels": int(((split_cc == instance_id) & (semantic == anatomy_id)).sum()),
            })
            next_by_anatomy[anatomy_id] += 1
    return remapped, {
        "small_merges": small_merges,
        "assigned": assigned,
        "assigned_count": len(assigned),
    }


def heal_by_splitting_quick(instances: np.ndarray,
                            abbc: np.ndarray,
                            fracture_disk_radius: int = ABBC_FRACTURE_DISK_RADIUS) -> tuple[np.ndarray, dict[str, Any]]:
    """Split large instances when predicted border indicates an internal break."""
    out = instances.copy()
    split_rows = []
    next_label = int(out.max()) + 1
    for region in regionprops(out):
        label_id = int(region.label)
        bbox = tuple(slice(int(a), int(b)) for a, b in zip(region.bbox[:3], region.bbox[3:]))
        inst_patch = out[bbox]
        object_mask = inst_patch == label_id
        if int(object_mask.sum()) < 400:
            continue
        pred_border = (abbc[bbox] == ABBC_BORDER_LABEL) & object_mask
        expected_border = get_fracture_regions(inst_patch * object_mask, fracture_disk_radius).astype(bool)
        error = pred_border & ~expected_border
        eroded = ndi.binary_erosion(error, structure=ball(1), border_value=0)
        n_error = int(eroded.sum())
        n_pred = int(pred_border.sum())
        relative_error = float(n_error / max(1, n_pred))
        if not (relative_error > 0.1 or n_error > 100):
            continue

        cut = pred_border.copy()
        work = object_mask.copy()
        split_cc = None
        split_count = 0
        for _ in range(10):
            cut = ndi.binary_dilation(cut, structure=ball(3))
            work[cut] = False
            split_cc, split_count = _label_components(work)
            if split_count > 1:
                sizes = sorted([int((split_cc == i).sum()) for i in range(1, split_count + 1)], reverse=True)
                if len(sizes) > 1 and sizes[1] > 400:
                    break
        else:
            continue
        if split_cc is None or split_count <= 1:
            continue
        keep_ids = [i for i in range(1, split_count + 1) if int((split_cc == i).sum()) > 100]
        if len(keep_ids) <= 1:
            continue
        seed_cc = np.zeros_like(split_cc, dtype=np.uint16)
        for rank, seed_id in enumerate(keep_ids, start=1):
            seed_cc[split_cc == seed_id] = rank
        assigned = _distance_assign(seed_cc, object_mask)
        crop = out[bbox]
        for rank in range(1, len(keep_ids) + 1):
            new_id = next_label
            next_label += 1
            crop[assigned == rank] = new_id
        out[bbox] = crop
        split_rows.append({
            "source": label_id,
            "new_instances": len(keep_ids),
            "border_error_voxels": n_error,
            "relative_error": relative_error,
        })
    out = _compact_local_labels(out)
    return out, {"splits": split_rows, "split_count": len(split_rows)}


def heal_by_merging(instances: np.ndarray,
                    abbc: np.ndarray,
                    semantic: np.ndarray,
                    fracture_disk_radius: int = ABBC_FRACTURE_DISK_RADIUS) -> tuple[np.ndarray, dict[str, Any]]:
    """Merge same-anatomy pairs when merged fracture score is clearly better."""
    out = instances.copy()
    anatomy_ids = _anatomy_ids_for_instances(semantic, out)
    merge_rows = []
    labels, counts = np.unique(out, return_counts=True)
    candidates = [int(v) for v, n in zip(labels, counts) if int(v) > 0 and int(n) > 100]
    anatomy_by_label = {
        instance_id: anatomy_id
        for anatomy_id, ids in anatomy_ids.items()
        for instance_id in ids
    }
    for a, b in itertools.combinations(candidates, 2):
        if anatomy_by_label.get(a) != anatomy_by_label.get(b):
            continue
        bbox_a = _mask_bbox(out == a, pad=20)
        bbox_b = _mask_bbox(out == b, pad=20)
        if bbox_a is None or bbox_b is None:
            continue
        bbox = [
            [max(sa.start, sb.start), min(sa.stop, sb.stop)]
            for sa, sb in zip(bbox_a, bbox_b)
        ]
        if any(hi <= lo for lo, hi in bbox):
            continue
        slicer = _slices_from_bbox(bbox)
        sub = out[slicer].copy()
        pair = (sub == a) | (sub == b)
        if not pair.any():
            continue
        touch = ndi.binary_dilation(sub == a, structure=ball(2)) & ndi.binary_dilation(sub == b, structure=ball(2))
        if not touch.any():
            continue
        split_border = get_fracture_regions(np.where(pair, sub, 0), fracture_disk_radius).astype(bool)
        merged = sub.copy()
        merged[merged == b] = a
        merged_border = get_fracture_regions(np.where(pair, merged, 0), fracture_disk_radius).astype(bool)
        pred_border = abbc[slicer] == ABBC_BORDER_LABEL
        split_score = scipy_distance.dice(split_border.ravel(), pred_border.ravel())
        merged_score = scipy_distance.dice(merged_border.ravel(), pred_border.ravel())
        split_score = 0.0 if np.isnan(split_score) else float(split_score)
        merged_score = 0.0 if np.isnan(merged_score) else float(merged_score)
        if merged_score < split_score - 0.05 or (merged_score == 1.0 and split_score == 1.0):
            out[out == b] = a
            merge_rows.append({
                "source": int(b),
                "target": int(a),
                "anatomy": int(anatomy_by_label[a]),
                "split_score": split_score,
                "merged_score": merged_score,
            })
    out = _compact_local_labels(out)
    return out, {"merges": merge_rows, "merge_count": len(merge_rows)}


def _compact_local_labels(instances: np.ndarray) -> np.ndarray:
    out = np.zeros_like(instances, dtype=np.uint16)
    for new_id, old_id in enumerate([int(v) for v in np.unique(instances) if int(v) > 0], start=1):
        out[instances == old_id] = new_id
    return out


def refine_border(mask: np.ndarray,
                  instances: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Fill semantic foreground not assigned to an instance by nearest instance."""
    out = instances.copy()
    fill = mask & (out == 0)
    if not fill.any() or not (out > 0).any():
        return out, {"filled_voxels": 0}
    nearest = ndi.distance_transform_edt(out == 0, return_distances=False, return_indices=True)
    nearest_label = out[tuple(nearest)]
    out[fill] = nearest_label[fill]
    return out, {"filled_voxels": int(fill.sum())}


def remove_small_global_instances(instances: np.ndarray,
                                  min_size: int = 1000) -> tuple[np.ndarray, dict[str, Any]]:
    out = instances.copy()
    removed = []
    labels, counts = np.unique(out, return_counts=True)
    for label, count in zip(labels, counts):
        label = int(label)
        if label == 0 or int(count) >= int(min_size):
            continue
        out[out == label] = 0
        removed.append({"global_id": label, "voxels": int(count)})
    return out, {"removed": removed, "removed_count": len(removed)}


def decode_official_abbc(semantic: np.ndarray,
                         abbc: np.ndarray,
                         min_core_size: int = 50,
                         min_instance_size: int = 1000,
                         fracture_disk_radius: int = ABBC_FRACTURE_DISK_RADIUS) -> tuple[np.ndarray, dict[str, Any]]:
    """Full official-style ABBC postprocess for pelvic instance prediction."""
    semantic = semantic.astype(np.uint8, copy=False)
    abbc = abbc.astype(np.uint8, copy=False).copy()
    pelvic_mask = (semantic >= 1) & (semantic <= 3)
    abbc[~pelvic_mask] = 0

    local_instances, conversion_trace = abbc_to_instance(abbc, min_core_size=min_core_size)
    remapped, remap_trace = remap_instance_labels(semantic, local_instances, min_size=min_instance_size)

    healed_local = _compact_local_labels(remapped)
    split_local, split_trace = heal_by_splitting_quick(
        healed_local, abbc, fracture_disk_radius=fracture_disk_radius,
    )
    merged_local, merge_trace = heal_by_merging(
        split_local, abbc, semantic, fracture_disk_radius=fracture_disk_radius,
    )
    remapped2, remap2_trace = remap_instance_labels(semantic, merged_local, min_size=min_instance_size)
    removed, remove_trace = remove_small_global_instances(remapped2, min_size=min_instance_size)
    # Refinement should only fill voxels supported by the ABBC foreground. Using
    # the full semantic pelvic mask here would resurrect Dataset532 false
    # positives even when the ABBC model predicted background there.
    refined, refine_trace = refine_border(pelvic_mask & (abbc > 0), removed)

    values, counts = np.unique(refined, return_counts=True)
    trace = {
        "decoder": "official_v1",
        "label_contract": {"0": "background", "1": "boundary_shell", "2": "core", "3": "contact_surface"},
        "min_core_size": int(min_core_size),
        "min_instance_size": int(min_instance_size),
        "fracture_disk_radius": int(fracture_disk_radius),
        "conversion": conversion_trace,
        "semantic_remap_initial": remap_trace,
        "split_healing": split_trace,
        "merge_healing": merge_trace,
        "semantic_remap_final": remap2_trace,
        "small_instance_removal": remove_trace,
        "border_refine": refine_trace,
        "prediction_voxels": {str(int(v)): int(c) for v, c in zip(values, counts) if int(v) > 0},
        "missing_seed_component": [
            row for row in conversion_trace.get("components", [])
            if row.get("input_voxels", 0) > 0 and row.get("assigned_instances", 0) == 0
        ],
    }
    return refined.astype(np.uint16, copy=False), trace


def audit_official_target(instances: np.ndarray,
                          target: np.ndarray) -> dict[str, Any]:
    """Return target invariants for one official ABBC label volume."""
    values, counts = np.unique(target, return_counts=True)
    count_map = Counter({int(v): int(c) for v, c in zip(values, counts)})
    zero_core = []
    multi_core = []
    core_leak = int(((target == ABBC_CORE_LABEL) & ~valid_instance_mask(instances)).sum())
    core_contact_overlap = int(((target == ABBC_CORE_LABEL) & (target == ABBC_CONTACT_LABEL)).sum())
    n_fragments = 0
    tiny_fragments = 0
    tiny_with_core = 0
    fragment_core_coverage = []
    core_cc_hist = Counter()
    for _anatomy_id, (_name, lo, hi) in PELVIC_ANATOMY_RANGES.items():
        selected = (instances >= lo) & (instances <= hi)
        for fragment_id in [int(v) for v in np.unique(instances[selected]) if lo <= int(v) <= hi]:
            n_fragments += 1
            frag = instances == fragment_id
            frag_voxels = int(frag.sum())
            core = (target == ABBC_CORE_LABEL) & frag
            if frag_voxels < 1000:
                tiny_fragments += 1
                if core.any():
                    tiny_with_core += 1
            fragment_core_coverage.append({
                "fragment_id": int(fragment_id),
                "fragment_voxels": frag_voxels,
                "core_voxels": int(core.sum()),
                "contact_voxels": int(((target == ABBC_CONTACT_LABEL) & frag).sum()),
                "core_fraction": float(core.sum() / max(1, frag_voxels)),
            })
            bbox = _mask_bbox(core, pad=1)
            n_cc = 0
            if bbox is not None:
                n_cc = _label_components(core[bbox])[1]
            core_cc_hist[int(n_cc)] += 1
            if n_cc == 0:
                zero_core.append(fragment_id)
            elif n_cc > 1:
                multi_core.append({"fragment_id": fragment_id, "core_cc": int(n_cc)})
    total = int(target.size)
    contact_cc, n_contact_cc = _label_components(target == ABBC_CONTACT_LABEL)
    contact_sizes = []
    if n_contact_cc:
        sizes = ndi.sum(target == ABBC_CONTACT_LABEL, contact_cc, index=np.arange(1, n_contact_cc + 1))
        contact_sizes = [int(v) for v in sizes]
    return {
        "n_fragments": int(n_fragments),
        "target_label_voxels": {str(i): int(count_map.get(i, 0)) for i in range(4)},
        "target_label_fractions": {str(i): float(count_map.get(i, 0)) / max(1, total) for i in range(4)},
        "contact_component_count": int(n_contact_cc),
        "contact_component_sizes_top10": sorted(contact_sizes, reverse=True)[:10],
        "core_contact_overlap_voxels": int(core_contact_overlap),
        "tiny_fragment_count_lt1000": int(tiny_fragments),
        "tiny_fragment_with_core_count": int(tiny_with_core),
        "fragment_core_coverage_worst10": sorted(
            fragment_core_coverage,
            key=lambda row: (row["core_voxels"] > 0, row["core_fraction"]),
        )[:10],
        "core_cc_hist": {str(k): int(v) for k, v in sorted(core_cc_hist.items())},
        "zero_core": zero_core,
        "multi_core": multi_core,
        "core_leak_voxels": int(core_leak),
        "ok": not zero_core and not multi_core and core_leak == 0 and core_contact_overlap == 0,
    }

def smoke_test() -> int:
    """Run lightweight checks for the active split anatomy + V68 fragment path."""
    print("\n=== PENGWIN 2026 code_task1 smoke tests (active: Ds539 anatomy + Ds538 ABBC, STU-Net-B) ===")
    passed = 0

    def _t(name, fn):
        nonlocal passed
        try:
            fn()
            print(f"  OK {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            raise

    # [cleanup 2026-06-07] The previous smoke test asserted a RETIRED contract — Dataset537
    # active with a phantom "PengwinTrainerBICMInstanceTopologyV73" (no such class), and
    # DATASETS == [532,533,537]. The active deployed pipeline is Ds539 (anatomy, STU-Net-B)
    # + Ds538 (fracture ABBC, STU-Net-B). These tests now assert that current reality so a
    # reader is not misled and `python utils.py test` passes.
    DS539_TRAINER = "PengwinTrainerSTUNetBaseAnatomyV301"
    DS538_TRAINER = "PengwinTrainerSTUNetBaseABBCPhase1V302"

    def t01_imports():
        import core, preprocessing, model, train  # noqa: F401

    def t02_core_active_registry():
        from core import DATASETS
        assert set(DATASETS) == {538, 539}, sorted(DATASETS)  # retired 532/533/537 removed 2026-06-12
        # active deployed datasets + their REAL trainers:
        assert DATASETS[539]["name"] == "Dataset539_PelvicFemurAnatomyV3"
        assert DATASETS[539]["kind"] == "anatomy_semantic"
        assert DATASETS[539]["trainer"] == DS539_TRAINER
        assert DATASETS[538]["name"] == "Dataset538_PelvicFemurBICMFragmentV5"
        assert DATASETS[538]["kind"] == "bicm_v5"
        assert DATASETS[538]["trainer"] == DS538_TRAINER

    def t03_label_map():
        arr = np.array([0, 1, 50, 51, 100, 101, 150, 151, 200, 250])
        assert inst_to_anat(arr).tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 4, 0]

    def t04_trainer_attrs():
        import core
        # the ACTIVE deployed trainers exist as REAL classes (this used to import a phantom V73).
        assert hasattr(core, DS539_TRAINER), DS539_TRAINER
        assert hasattr(core, DS538_TRAINER), DS538_TRAINER
        # x-mirror is disabled for the L/R-asymmetric anatomy datasets (532 retired, 539 active)
        assert "Dataset539_PelvicFemurAnatomyV3" in core.PengwinTrainer.DISABLE_X_MIRROR_DATASETS

    def t05_train_model_contract():
        import train
        # TRAINERS is now derived dynamically from core (no hand-maintained phantom list);
        # the active trainers must therefore be present.
        assert DS539_TRAINER in train.TRAINERS
        assert DS538_TRAINER in train.TRAINERS

    tests = [
        ("imports", t01_imports),
        ("core active registry", t02_core_active_registry),
        ("instance-to-anatomy map", t03_label_map),
        ("trainer active attrs", t04_trainer_attrs),
        ("train/model active contract", t05_train_model_contract),
    ]
    for name, fn in tests:
        _t(name, fn)
    print(f"\n{passed}/{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        raise SystemExit(smoke_test())
    print("Use: python utils.py test")

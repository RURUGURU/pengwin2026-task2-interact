"""PENGWIN 2026 Task 1 — Core constants and dataset registry.

Single source of truth for:
    - PENGWIN label scheme (1-200 instance IDs)
    - Anatomy classes + ranges
    - Pelvic / Femur split (case ID ranges)
    - Active split Pelvic/Femur anatomy datasets
    - Active pelvic per-anatomy BICM V5 fragment dataset
"""
from __future__ import annotations
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import sys
import time
from typing import Literal

# =============================================================================
# Paths
# =============================================================================
# PENGWIN_ROOT wins when set (the GC container's inference.py exports it = /opt/ml/model
# BEFORE importing any code_task1 module). When unset (host/dev), auto-resolve the repo root
# from this file's location (core.py lives at <root>/code_task1/core.py), so the tree is
# portable across machines without hardcoding /workspace.
ROOT = Path(os.environ.get("PENGWIN_ROOT") or Path(__file__).resolve().parents[1])
DATA_RAW = ROOT / "data/task1_2/extracted"
# nnUNet 출력 root 를 code_task1/result/ 아래로 통합 (2026-05-31 정리).
# 이전: ROOT/nnunet/{raw,preprocessed,results} — workspace 루트에 분산.
# 현재: ROOT/code_task1/result/{raw,preprocessed,results} — code_task1 내부에서 자급자족.
# 변경 이유: code_task1 이 독립 작업 디렉토리가 되도록 — 모든 산출물이 한 곳에 모이게.
NN_RAW = ROOT / "code_task1/result/raw"
NN_PREP = ROOT / "code_task1/result/preprocessed"
NN_RES = ROOT / "code_task1/result/results"
RESULT = ROOT / "code_task1/result"
# Result artifacts 의 평면화 (2026-05-31 정리).
# 이전: task1_active/<UTC YYYYMMDD>/{reports,weights,visualize} — date partition.
# 현재: reports/ 단일 폴더 — 시간 정보는 파일명의 RESULT_DATE 접미사로 표현.
# 변경 이유: code_task1/ 내부 중복 prefix (task1_active) 제거, day-partition 으로 인한
# 디스크 분산 및 GC 부담 회피.
RESULT_DATE = os.environ.get("PENGWIN_RESULT_DATE", time.strftime("%Y%m%d", time.gmtime()))
RESULT_REPORT = RESULT / "reports"
RESULT_WEIGHT = RESULT / "weights"
RESULT_VISUALIZE = RESULT / "visualize"


def configure_nnunet_env(force: bool = False) -> dict[str, str]:
    """Set nnU-Net path environment variables from this repo layout.

    nnU-Net still reads these paths from environment variables at import and
    inference time. Keeping the setup here makes every Python entry point
    self-contained and prevents the noisy "nnUNet_raw is not defined" warnings
    during smoke tests.
    """
    values = {
        "nnUNet_raw": str(NN_RAW),
        "nnUNet_preprocessed": str(NN_PREP),
        "nnUNet_results": str(NN_RES),
        # nnU-Net v1-style compatibility variable used by a few helpers.
        "nnUNet_raw_data_base": str(NN_RAW),
    }
    for key, value in values.items():
        if force or not os.environ.get(key):
            os.environ[key] = value
    return values


def get_logger(name: str) -> logging.Logger:
    """Return a module logger without requiring a separate local logging module."""
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
    return log


configure_nnunet_env()


# =============================================================================
# PENGWIN label scheme (대회 공식) — single source of truth in anatomy_registry
# =============================================================================
# Instance IDs encode anatomy via fixed 50-wide blocks (Sacrum 1-50, LeftHip
# 51-100, RightHip 101-150, Femur 151-200). The canonical registry AND every
# instance-ID arithmetic helper now live in `anatomy_registry`, re-exported here
# so existing `from core import ANATOMY_RANGES` / `core.<x>` references keep
# resolving unchanged. Add or modify an anatomy THERE (one row), never here.
#
# DECOY WARNING: the case-ID helpers below (is_pelvic/is_femur, 1-120/151-200/
# 251-420) and HU/probability thresholds elsewhere are a DIFFERENT namespace that
# happens to share the digits 150/200/50 — they must NOT route through the registry.
#
# NOTE [consolidation 2026-06-12]: the anatomy registry now lives in `utils.py`
# (formerly the standalone `anatomy_registry.py`). core re-exports those symbols so
# existing `from core import ANATOMY_RANGES` / `core.<x>` references keep resolving
# unchanged. Because `utils` imports `DATA_RAW`/`PAD` from `core`, the re-export
# import is deferred to AFTER those constants are defined (see below, post-`PAD`),
# avoiding a core<->utils circular import.


# =============================================================================
# Pelvic / Femur split (active source case ID ranges)
# =============================================================================
def is_pelvic(cid: int) -> bool:
    """True if case ID belongs to Pelvic train set (001-120, 151-200)."""
    return (1 <= cid <= 120) or (151 <= cid <= 200)


def is_femur(cid: int) -> bool:
    """True if case ID belongs to Femur train set (251-420)."""
    return 251 <= cid <= 420


def case_subject_type(cid: int) -> str:
    """Return 'Pelvic' / 'Femur' / 'Unknown' based on case ID range."""
    if is_pelvic(cid):
        return "Pelvic"
    if is_femur(cid):
        return "Femur"
    return "Unknown"


# =============================================================================
# Dataset registry — active anatomy V2 + fragment V5 datasets only
# =============================================================================
ABBC_BOUNDARY_LABEL = 1
ABBC_CORE_LABEL = 2
ABBC_BORDER_LABEL = 3
ABBC_CONTACT_LABEL = ABBC_BORDER_LABEL
ABBC_HARD_NEGATIVE_LABEL = 4

# [AUDIT][Risk:Medium][Scope:legacy_constant]
# The constants below remain only so old JSON/probe readers can import this
# module while we delete heavy V3/V4 artifacts. Active Dataset537 V5 does not
# use these heads or sidecars; registry and CLI must point to `bicm_v5` only.
FACTOR_INSTANCE_OUTPUT_CHANNELS = 15
FACTOR_INSTANCE_CHANNEL_NAMES = [
    "support_logit",
    "contact_energy_logit",
    "core_seed_logit",
    "hard_negative_logit",
    "offset_z",
    "offset_y",
    "offset_x",
    "embedding_0",
    "embedding_1",
    "embedding_2",
    "embedding_3",
    "embedding_4",
    "embedding_5",
    "embedding_6",
    "embedding_7",
]
FACTOR_INSTANCE_SIDECAR_DIR = "factorized_instance_targets"
BOUNDARY_FRAGMENT_V3_TARGET_SIDECAR_DIR = "boundary_fragment_v3_targets"

# Retained for old JSON/probe readers only. This 19-channel morphology/edge
# contract is not the active Dataset537 training contract.
CONTACT_INSTANCE_OUTPUT_CHANNELS = 19
CONTACT_INSTANCE_MORPH_SUPPORT_CH = 1
CONTACT_INSTANCE_MORPH_CONTACT_CH = 2
CONTACT_INSTANCE_MORPH_HARD_NEGATIVE_CH = 3
CONTACT_INSTANCE_CORE_CH = 4
CONTACT_INSTANCE_EDGE_BREAK_CHS = (16, 17, 18)
CONTACT_INSTANCE_CHANNEL_NAMES = [
    "morph_background_logit",
    "morph_support_noncontact_logit",
    "morph_contact_surface_logit",
    "morph_hard_negative_logit",
    "core_center_logit",
    "offset_z",
    "offset_y",
    "offset_x",
    "embedding_0",
    "embedding_1",
    "embedding_2",
    "embedding_3",
    "embedding_4",
    "embedding_5",
    "embedding_6",
    "embedding_7",
        "edge_break_z_logit",
        "edge_break_y_logit",
        "edge_break_x_logit",
    ]

BICM_V38_OUTPUT_CHANNELS = 8

BICM_V68_OUTPUT_CHANNELS = 10

BFV3_BINARY_BARRIER_OUTPUT_CHANNELS = 6
BFV3_BINARY_BARRIER_SEED_OUTPUT_CHANNELS = 7
BFV3_XYZ_AFFINITY_OUTPUT_CHANNELS = 8
BFV3_AFFINITY13_SEED_OUTPUT_CHANNELS = 18
BFV3_MUTEX13_SEED_OUTPUT_CHANNELS = 31
BFV3_NO_CONTACT_PAIRWISE_V273_OUTPUT_CHANNELS = 28
BFV3_NO_CONTACT_PAIRWISE_V273_CHANNEL_NAMES = [
    "support_logit",
    "seed_body_logit",
    "join_0_0_1_logit",
    "join_0_1_-1_logit",
    "join_0_1_0_logit",
    "join_0_1_1_logit",
    "join_1_-1_-1_logit",
    "join_1_-1_0_logit",
    "join_1_-1_1_logit",
    "join_1_0_-1_logit",
    "join_1_0_0_logit",
    "join_1_0_1_logit",
    "join_1_1_-1_logit",
    "join_1_1_0_logit",
    "join_1_1_1_logit",
    "cut_0_0_1_logit",
    "cut_0_1_-1_logit",
    "cut_0_1_0_logit",
    "cut_0_1_1_logit",
    "cut_1_-1_-1_logit",
    "cut_1_-1_0_logit",
    "cut_1_-1_1_logit",
    "cut_1_0_-1_logit",
    "cut_1_0_0_logit",
    "cut_1_0_1_logit",
    "cut_1_1_-1_logit",
    "cut_1_1_0_logit",
    "cut_1_1_1_logit",
]
BFV3_FRAGMENT_POSITION_V275_OUTPUT_CHANNELS = 51
BFV3_SEPARATOR_GAP_V277_OUTPUT_CHANNELS = 2
BFV3_SEPARATOR_GAP_V277_CHANNEL_NAMES = [
    "support_logit",
    "separator_gap_logit",
]
BFV3_SEPARATOR_ENERGY_V278_OUTPUT_CHANNELS = 2
BFV3_SEPARATOR_SOFTMAX_V287_OUTPUT_CHANNELS = 3
BFV3_SEPARATOR_SOFTMAX_V287_CHANNEL_NAMES = [
    "background_logit",
    "support_body_logit",
    "separator_gap_logit",
]
BFV3_ABBC_V288_OUTPUT_CHANNELS = 4
BFV3_ABBC_V288_CHANNEL_NAMES = [
    "background_logit",
    "border_logit",
    "boundary_logit",
    "core_logit",
]
BFV3_ABBC_SDF_V289_OUTPUT_CHANNELS = 4
BFV3_ABBC_SDF_FDM_V290_OUTPUT_CHANNELS = 4
BFV3_ABBC_BWEIGHT_V291_OUTPUT_CHANNELS = 4
BFV3_ABBC_BWEIGHT_V291_CHANNEL_NAMES = BFV3_ABBC_V288_CHANNEL_NAMES
BFV3_CENTER_FLOW_OUTPUT_CHANNELS = 8
BFV3_NO_CONTACT_CENTER_FLOW_OUTPUT_CHANNELS = 5
BFV3_SPATIAL_EMBEDDING_OUTPUT_CHANNELS = 4
BFV3_QUERY_MASK_V280_OUTPUT_CHANNELS = 1
BFV3_QUERY_MASK_PN_V281_OUTPUT_CHANNELS = 1
BFV3_FREE_EMBEDDING_V282_OUTPUT_CHANNELS = 5
BFV3_GLOBAL_COORD_FREE_EMBEDDING_V283_OUTPUT_CHANNELS = 5


@dataclass(frozen=True)
class DatasetCfg:
    """Single dataset definition. Heterogeneous fields by `kind`.

    Active training uses split semantic anatomy targets plus one pelvic
    per-anatomy fragment target:
    - Dataset532_PelvicAnatomyV2: background/Sacrum/LeftHip/RightHip.
    - Dataset533_FemurAnatomyV2: background/Femur.
    - Dataset537_PelvicBICMFragmentV5: CT-LUT per-anatomy ROI samples with V5
      target geometry. Active training uses V68 hybrid semantic/topology heads
      because V67 support succeeded while contact/core topology failed.

    """
    name: str
    kind: Literal["anatomy_semantic", "bicm_v5"]
    filter: Literal["all", "pelvic", "femur"]
    n_classes: int
    trainer: Literal[
        "PengwinTrainer",
        "PengwinTrainerBICMFactorizedV6",
        "PengwinTrainerBICMContactV8",
        "PengwinTrainerBICMContactV9",
        "PengwinTrainerBICMContactV10",
        "PengwinTrainerBICMContactV11",
        "PengwinTrainerBICMContactV12",
        "PengwinTrainerBICMContactV13",
        "PengwinTrainerBICMContactV14",
        "PengwinTrainerBICMContactV15",
        "PengwinTrainerBICMContactV16",
        "PengwinTrainerBICMContactV17",
        "PengwinTrainerBICMContactV18",
        "PengwinTrainerBICMContactV19",
        "PengwinTrainerBICMContactV20",
        "PengwinTrainerBICMContactV22",
        "PengwinTrainerBICMContactV23",
        "PengwinTrainerBICMContactV24",
        "PengwinTrainerBICMContactV25",
        "PengwinTrainerBICMContactV26",
        "PengwinTrainerBICMContactV27",
        "PengwinTrainerBICMContactV28",
        "PengwinTrainerBICMContactV29",
        "PengwinTrainerBICMContactV30",
        "PengwinTrainerBICMContactV31",
        "PengwinTrainerBICMContactV32",
        "PengwinTrainerBICMContactV34",
        "PengwinTrainerBICMContactV35",
        "PengwinTrainerBICMContactV36",
        "PengwinTrainerBICMContactV37",
        "PengwinTrainerBICMEdgeAffinityV38",
        "PengwinTrainerBICMEdgePrimaryV39",
        "PengwinTrainerBICMEdgeCoreV40",
        "PengwinTrainerBICMInstanceCoreV41",
        "PengwinTrainerBICMInstanceCoreV42",
        "PengwinTrainerBICMEdgeContactV43",
        "PengwinTrainerBICMEdgeLocalRankV44",
        "PengwinTrainerBICMEdgeCandidateV45",
        "PengwinTrainerBICMEdgeCurriculumV46",
        "PengwinTrainerBICMSeparatedContactV47",
        "PengwinTrainerBICMSupportAwareContactV48",
        "PengwinTrainerBICMDecoderFeatureContactV49",
        "PengwinTrainerBICMDecoderFeaturePhaseV50",
        "PengwinTrainerBICMDenseEdgeCostV51",
        "PengwinTrainerBICMDenseEdgeCoreV52",
        "PengwinTrainerBICMFragmentMarkerCoreV53",
        "PengwinTrainerBICMSparseHeadBalancedV54",
        "PengwinTrainerBICMCalibratedSparseHeadV55",
        "PengwinTrainerBICMPhasedMarkerContactV56",
        "PengwinTrainerBICMDecoderFeatureMarkerPhaseV57",
        "PengwinTrainerBICMHeatmapMarkerContactV58",
        "PengwinTrainerBICMCorePreservingContactV59",
        "PengwinTrainerBICMStrictCoreTopologyV60",
        "PengwinTrainerBICMPeakSeedV61",
        "PengwinTrainerBICMSemanticTopologyV68",
        "PengwinTrainerBICMTopologyCalibratedV69",
        "PengwinTrainerBICMTopologyConsistencyV70",
        "PengwinTrainerBICMEdgeCutPrimaryV71",
        "PengwinTrainerBICMLogitCalibratedV72",
        "PengwinTrainerBICMInstanceTopologyV73",
        "PengwinTrainerBICMAdaptiveInstanceTopologyV74",
        "PengwinTrainerBICMEdgePrecisionSeedTopologyV75",
        "PengwinTrainerBICMGentleEdgePrecisionV76",
        "PengwinTrainerBICMEdgeCurriculumV77",
        "PengwinTrainerBICMCoreAnchoredEdgeCurriculumV78",
        "PengwinTrainerBICMDuplicateSeedEdgeV79",
        "PengwinTrainerBICMTopologyStateAdaptiveV80",
        "PengwinTrainerBICMCoreStableEdgePrecisionV81",
        "PengwinTrainerBICMSeedRecallEdgeSeparationV82",
        "PengwinTrainerBICMSeedSafeEdgeCalibrationV83",
        "PengwinTrainerBICMDualHeadContactCalibrationV84",
        "PengwinTrainerBICMSemanticGateContactV85",
        "PengwinTrainerBICMEvalBandEdgePrimaryV86",
        "PengwinTrainerBICMEvalBandSemanticGateV87",
        "PengwinTrainerBICMCorePreservingBandPrecisionV88",
        "PengwinTrainerBICMBandFalseOnlyPrecisionV89",
        "PengwinTrainerBICMSemanticBandProductV90",
        "PengwinTrainerBICMTopologyAwareEdgeBalanceV91",
        "PengwinTrainerBICMEvalAlignedSupportContactV93",
        "PengwinTrainerBICMFinalRowCalibratedV94",
        "PengwinTrainerBICMDecoderFeatureContactV95",
        "PengwinTrainerBICMDenseBandGateV96",
        "PengwinTrainerBICMPositiveDenseBandGateV97",
        "PengwinTrainerBICMTeacherDistilledDenseGateV98",
        "PengwinTrainerBICMAdaptiveDualMarginDenseGateV99",
        "PengwinTrainerBICMNegativeBalancedDenseGateV100",
        "PengwinTrainerBICMDualFieldProductGateV101",
        "PengwinTrainerBICMDistanceRankDenseGateV102",
        "PengwinTrainerBICMSemanticDistanceContactV103",
        "PengwinTrainerBICMTeacherSemanticContactV104",
        "PengwinTrainerBICMOffsetAssignmentV105",
        "PengwinTrainerBICMOffsetAttractorV106",
        "PengwinTrainerBICMRadialSupportOffsetV107",
        "PengwinTrainerBICMWatershedBarrierV108",
        "PengwinTrainerBICMPrecisionLockedRecallV109",
        "PengwinTrainerBICMSemanticGeometryBridgeV110",
    ]
    anatomies: list[str] = field(default_factory=list)
    anatomy: str | None = None
    global_label_range: tuple[int, int] | None = None
    foundation_dataset: int | None = None

    def __getitem__(self, key: str):
        """Dict-style access for existing utility code."""
        return getattr(self, key)


DATASETS: dict[int, DatasetCfg] = {
    # [V0.x][FIX:B2][2026-05-31] Dataset538 — pelvic+femur 4-anatomy BICM V5.
    # filter="all" 로 170 pelvic + 170 femur (총 340) 케이스 모두 포함.
    # global_label_range=(1, 200) 로 femur fragment ID (151-200) 까지 커버.
    # anatomies=[..., "Femur"] 로 빌드 시점에 4 ROI 생성.
    # foundation_dataset=539 — Stage E inference 시 Dataset539 의 5-class
    # anatomy probability 를 입력 채널로 사용.
    538: DatasetCfg(
        "Dataset538_PelvicFemurBICMFragmentV5",
        "bicm_v5",
        "all",
        5,
        # [cleanup 2026-06-07] was "PengwinTrainerBICMCoreStableEdgePrecisionV81" — a phantom
        # (no such class). The ACTIVE deployed Ds538 fracture trainer is the STU-Net-B ABBC one.
        "PengwinTrainerSTUNetBaseABBCPhase1V302",
        anatomies=["Sacrum", "LeftHip", "RightHip", "Femur"],
        anatomy=None,
        global_label_range=(1, 200),
        foundation_dataset=539,
    ),
    # [V0.x][FIX:B1][2026-05-31] Dataset539 — Dataset532 의 5-class 버전.
    # 베이스라인 Dataset001 과 호환되는 5-class semantic anatomy (Femur 포함).
    # filter="all" 로 340 케이스 모두 학습 — 170 femur-only 케이스 구제.
    539: DatasetCfg(
        "Dataset539_PelvicFemurAnatomyV3",
        "anatomy_semantic",
        "all",
        5,
        # [cleanup 2026-06-07] was the base "PengwinTrainer" — the ACTIVE deployed Ds539
        # anatomy trainer is the STU-Net-B one.
        "PengwinTrainerSTUNetBaseAnatomyV301",
        anatomies=["Sacrum", "LeftHip", "RightHip", "Femur"],
    ),
}


# =============================================================================
# SOTA reference (PENGWIN 2024 official, MIC-DKFZ 1st)
# =============================================================================
SOTA = {
    "iou_a": 0.9810,
    "iou_f": 0.9296,
    "hd95_f": 5.866,
    "assd_f": 1.843,
}


# =============================================================================
# Geometry defaults used by preprocessing diagnostics
# =============================================================================
PAD = 15              # v8: 10→15. bbox crop 시 small fragment (~100 vox) 절단 방지.
                      #     PENGWIN spec drops <500mm³ ≈ 1000 vox at 0.8mm³, but our
                      #     GT contains down to 1 vox fragments — 5 voxel 추가 margin.


# =============================================================================
# PENGWIN anatomy <-> instance-ID registry (single source of truth in utils.py).
# Re-exported here so existing `from core import ANATOMY_RANGES` / `core.<x>`
# references resolve unchanged. `utils` is a pure leaf w.r.t. core at import time
# (it accesses DATA_RAW lazily inside functions), so this import does not create a
# core<->utils cycle and `utils` stays importable standalone.
# =============================================================================
from utils import (  # noqa: E402  (utils anatomy registry, re-exported from core)
    Anatomy,
    ANATOMY_REGISTRY,
    ANATOMY_RANGES,
    ANATOMY_NAMES,
    ANATOMY_RANGE_DICT,
    ANATOMY_TO_INDEX,
    PELVIC_ANATOMY_INDICES,
    NUM_ANATOMIES,
    MIN_INSTANCE_ID,
    MAX_INSTANCE_ID,
    PELVIC_MAX_INSTANCE_ID,
    INSTANCE_CAPACITY,
    anatomy_by_index,
    anatomy_by_name,
    anatomy_of_id,
    id_range,
    all_anatomy_ranges,
    anatomy_start_ids,
    anatomy_ranges_by_name,
    global_to_local,
    local_to_global,
    valid_instance_mask,
    clip_to_valid_instances,
    anatomy_index_array,
    same_anatomy,
)


# =============================================================================
# Active nnU-Net trainers
# =============================================================================
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from torch import distributed as dist
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from typing import Tuple

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import dummy_context
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

# [V0.x][WARN-FIX:#3][2026-06-02] nnUNet 2.5.2 가 deprecated `torch.cuda.amp.GradScaler()`
# 를 nnUNetTrainer.initialize() 의 GradScaler() 호출(line 164)에서 생성하며 FutureWarning
# 을 띄운다. 경고는 *생성 시점*에 발생하므로 사후 교체로는 못 막는다. nnUNet 트레이너 모듈
# 네임스페이스의 GradScaler 심볼을 신 API(torch.amp.GradScaler('cuda')) 팩토리로 교체해
# 생성 자체를 modern API 로 바꾼다(억제 아님, 근본 수정). 모든 trainer 에 적용.
try:
    import torch.amp as _torch_amp
    import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as _nnunet_trainer_mod

    def _modern_grad_scaler(*_a, **_k):
        return _torch_amp.GradScaler("cuda", *_a, **_k)

    _nnunet_trainer_mod.GradScaler = _modern_grad_scaler
except Exception:  # pragma: no cover
    pass

# Make helper modules importable when nnU-Net imports the native PengwinTrainer
# module from its supported `nnUNetTrainer/variants` package path.
_ROOT = Path(os.environ.get("PENGWIN_ROOT") or Path(__file__).resolve().parents[1])
_CODE_TASK1 = _ROOT / "code_task1"
if str(_CODE_TASK1) not in sys.path:
    sys.path.insert(0, str(_CODE_TASK1))
# [V0.x][2026-06-01] STU-Net 백본 (vendored, Apache-2.0) — TotalSegmentator 뼈 사전학습 전이용.
try:
    from model import STUNet, STUNET_VARIANTS
    _STUNET_AVAILABLE = True
except Exception:  # pragma: no cover
    STUNet = None
    STUNET_VARIANTS = {}
    _STUNET_AVAILABLE = False
try:
    from loss import (
        LeakFreeInstanceABBCLoss,
        DC_CE_BD_loss,
        DC_CE_TV_BD_loss,
        compute_median_frequency_class_weights,
    )
    _BOUNDARY_LOSS_AVAILABLE = True
except ImportError:
    _BOUNDARY_LOSS_AVAILABLE = False

try:
    from fuseformer import ContactLegacyFuseUNet
    _FUSEFORMER_AVAILABLE = True
except ImportError:
    _FUSEFORMER_AVAILABLE = False


def _pelvic_same_anatomy_contact_mask(inst: np.ndarray) -> np.ndarray:
    """Narrow contact target: 6-neighbor different fragments in the same bone.

    [QC][Performance][Scope:dataloader_target]
    This runs for every sampled crop. The previous implementation dilated every
    fragment separately and made V4.2 CPU-bound before epoch 0 finished. The
    adjacency formulation is the same local topology contract but scales with
    three axis comparisons rather than number_of_fragments * crop_volume.
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
        same_anatomy = (
            (a > 0) & (b > 0) & (a != b)
            & (((a.astype(np.int32) - 1) // 50) == ((b.astype(np.int32) - 1) // 50))
        )
        if same_anatomy.any():
            out_a = out[tuple(a_sl)]
            out_b = out[tuple(b_sl)]
            out_a[same_anatomy] = True
            out_b[same_anatomy] = True
            out[tuple(a_sl)] = out_a
            out[tuple(b_sl)] = out_b
    return out


def _instance_edge_break_target(inst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return 3-axis same-anatomy break-edge targets for fragment topology.

    `edge_break[a, z, y, x]` describes the edge from voxel `(z,y,x)` to the
    next voxel in axis `a`. Positive edges are only same-anatomy,
    different-fragment pelvic adjacencies. All in-volume adjacencies are valid
    negatives otherwise, including same-fragment interior edges, pelvis-to-
    background edges, external cortical surface, femur/vertebra-like hard
    negatives, and pure background. Keeping the background/external edges in
    the loss is important: when valid edges were limited to pelvic support, the
    edge head was unconstrained outside the support and lit up most of the CT.
    Raw adjacency positives were only about 7e-5 of all in-volume edges in the
    overfit set, so training uses a small support-limited dilation controlled
    by `PENGWIN_EDGE_BREAK_DILATION` (default: 1 voxel).
    """
    inst = np.where((inst >= 1) & (inst <= MAX_INSTANCE_ID), inst, 0).astype(np.int16, copy=False)
    edge_break = np.zeros((3, *inst.shape), dtype=np.float32)
    edge_valid = np.zeros((3, *inst.shape), dtype=np.float32)
    support_touch = np.zeros((3, *inst.shape), dtype=bool)
    for axis in range(3):
        a_sl = [slice(None)] * 3
        b_sl = [slice(None)] * 3
        a_sl[axis] = slice(0, -1)
        b_sl[axis] = slice(1, None)
        a = inst[tuple(a_sl)]
        b = inst[tuple(b_sl)]
        in_volume = np.ones(a.shape, dtype=bool)
        a_fg = a > 0
        b_fg = b > 0
        same_anatomy = a_fg & b_fg & (
            ((a.astype(np.int32) - 1) // 50) == ((b.astype(np.int32) - 1) // 50)
        )
        edge_valid[(axis, *a_sl)] = in_volume.astype(np.float32)
        support_touch[(axis, *a_sl)] = a_fg | b_fg
        edge_break[(axis, *a_sl)] = (same_anatomy & (a != b)).astype(np.float32)
    radius = int(os.environ.get("PENGWIN_EDGE_BREAK_DILATION", "1"))
    if radius > 0 and edge_break.any():
        from scipy.ndimage import binary_dilation

        structure = np.ones((2 * radius + 1, 2 * radius + 1, 2 * radius + 1), dtype=bool)
        for axis in range(3):
            edge_break[axis] = (
                binary_dilation(edge_break[axis] > 0.5, structure=structure)
                & support_touch[axis]
            ).astype(np.float32)
    return edge_break, edge_valid


class PengwinBICMV5DataLoader3D(nnUNetDataLoader3D):
    """V5 dataloader that centers crops on sparse decoder-critical labels.

    Dataset537 V5 keeps a simple 5-class semantic target. The first 003 overfit
    showed that default nnU-Net foreground crops learn shell/support but never
    see enough core/contact voxels to make a marker-based decoder work. This
    dataloader changes only the forced-foreground class choice; transforms,
    padding, batching, and validation remain native nnU-Net behavior.
    """

    SPARSE_CLASS_DISTRIBUTION = (
        (3, 0.55),  # core marker: required for every decoded fragment seed
        (4, 0.30),  # contact surface: required to split same-anatomy fragments
        (2, 0.10),  # shell: high-volume support near-negative
        (1, 0.05),  # exterior context: suppress support leakage
    )
    SUPPORT_MIXED_CLASS_DISTRIBUTION = (
        # [DATA][Risk:High][Scope:case003_support_fp]
        # V6.2 removed contact supervision and still produced support masks
        # roughly 2-3x larger than GT. That isolates the next question to
        # sampling: the sparse profile always forces a positive foreground crop
        # and mostly centers core/contact, so the support head rarely sees
        # exterior/context-dominated negatives. This profile keeps marker/core
        # exposure but deliberately spends much more forced-foreground budget on
        # class 1 and class 2 boundaries; the remaining non-forced batches come
        # from oversample_foreground_percent < 1.0 in the trainer.
        (1, 0.40),  # exterior context: support negative adjacent to bone
        (2, 0.30),  # shell/support: positive support geometry
        (3, 0.25),  # core marker: preserve seed coverage
        (4, 0.05),  # contact: kept only as rare support-positive context
    )
    CONTACT_MIXED_CLASS_DISTRIBUTION = (
        # [DATA][Risk:High][Scope:contact_recall]
        # V7 anatomy-context support passed full-volume HD95, but contact recall
        # stayed low when only 5% of forced crops were class-4 centered. This
        # profile changes only crop centers for the next ablation: contact gets
        # enough positive patches while exterior/shell/core remain represented so
        # broad support/contact FP does not become invisible to the loss.
        (4, 0.35),  # contact: decoder split ridge
        (1, 0.25),  # exterior context: contact/support negative
        (2, 0.25),  # shell/support: support non-contact near-negative
        (3, 0.15),  # core marker: seed stability
    )
    CONTACT15_MIXED_CLASS_DISTRIBUTION = (
        # [DATA][Risk:High][Scope:single_variable_sampling]
        # `bicm_v7_contact_mixed` at 35% contact centers destabilized support
        # and collapsed contact in patch validation. This middle profile tests
        # whether the original 5% contact exposure was simply too sparse while
        # keeping most crop budget on exterior/shell/core context.
        (1, 0.35),  # exterior context: suppress broad support/contact
        (2, 0.25),  # shell/support: support non-contact near-negative
        (3, 0.25),  # core marker: seed stability
        (4, 0.15),  # contact: moderate positive exposure
    )
    CONTACT_ROI_CLASS_DISTRIBUTION = (
        # [DATA][Risk:High][Scope:contact_positive_exposure]
        # V10 full-volume case003 had good support/core but predicted zero
        # contact even in LeftHip, the only ROI with GT contact. V11 changes
        # exposure, not target or decoder: when a contact-positive ROI is
        # sampled, most forced foreground crops are centered on class 4. The
        # trainer-level ROI weighting below still keeps no-contact ROIs in the
        # stream so absent-contact negatives remain visible.
        (4, 0.60),  # contact: make the positive ridge visible to the head
        (2, 0.20),  # shell/support: support-local non-contact near-negative
        (3, 0.15),  # core marker: seed stability
        (1, 0.05),  # exterior context: suppress support/contact leakage
    )

    def __init__(self, *args, forced_class_distribution=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.forced_class_distribution = tuple(forced_class_distribution or self.SPARSE_CLASS_DISTRIBUTION)

    @staticmethod
    def _class_key(class_locations: dict, label: int):
        if not class_locations:
            return None
        for key, value in class_locations.items():
            try:
                if int(key) == int(label) and len(value) > 0:
                    return key
            except (TypeError, ValueError):
                continue
        return None

    def _pick_forced_class(self, class_locations: dict):
        available = [
            (label, prob)
            for label, prob in self.forced_class_distribution
            if self._class_key(class_locations, label) is not None
        ]
        if not available:
            return None
        labels = [label for label, _ in available]
        probs = np.asarray([prob for _, prob in available], dtype=np.float64)
        probs = probs / probs.sum()
        return self._class_key(class_locations, int(np.random.choice(labels, p=probs)))

    def get_bbox(self, data_shape: np.ndarray, force_fg: bool,
                 class_locations: dict | None,
                 overwrite_class=None, verbose: bool = False):
        # [DATA][Risk:Major][Scope:sampling]
        # Core/contact targets can be single-digit voxels per ROI. When a crop
        # is already forced to foreground, choosing the rare class explicitly is
        # the least invasive way to test whether collapse was caused by sample
        # density rather than target topology or model architecture.
        if force_fg and overwrite_class is None and class_locations:
            overwrite_class = self._pick_forced_class(class_locations)
        return super().get_bbox(
            data_shape,
            force_fg,
            class_locations,
            overwrite_class=overwrite_class,
            verbose=verbose,
        )


BICM_V6_OUTPUT_CHANNELS = 4
BICM_V8_OUTPUT_CHANNELS = 5


class _GroupedSplitMixin:
    """[2026-06-05] Source-case grouped K-fold split — single source of truth.

    Stock nnUNet do_split runs random KFold over per-ROI identifiers, so a source
    CT with several anatomy ROIs (Dataset538) leaks across train/val. This mixin
    regenerates splits_final.json grouped by source case (generate_crossval_split
    over group IDs, seed 12345) before stock do_split reads it. Single-ROI datasets
    (Ds539) are delegated unchanged (grouped == ungrouped). Both the anatomy trainer
    (via PengwinTrainer) and the fracture chain (PengwinTrainerBoundaryFragmentV3)
    inherit this so Stage A / Stage B share ONE patient-grouped split (same seed
    12345 over the same 340 source cases → identical fold partition in both).
    """

    @staticmethod
    def _source_case_from_identifier(identifier: str) -> str:
        """nnU-Net 샘플 identifier에서 zero-padding된 source case ID를 반환한다."""
        stem = str(identifier).replace("PENGWIN_", "")
        return stem.split("_", 1)[0].zfill(3)

    @staticmethod
    def _splits_group_consistent(splits: list, source_of) -> bool:
        """splits_final.json 의 각 fold 에서 같은 source case 가 train/val 에 동시에
        들어가지 않는지(=leakage 없음) 검사한다."""
        for fold in splits:
            tr_src = {source_of(k) for k in fold.get("train", [])}
            val_src = {source_of(k) for k in fold.get("val", [])}
            if tr_src & val_src:
                return False
        return True

    def _ensure_grouped_splits_file(self):
        """[FIX:C1] source-case 단위 grouped K-fold split 강제 (위 클래스 docstring 참조).

        stock 이 splits_final.json 을 읽기 전에 source case 단위로 묶은 grouped split 을
        미리 생성한다. 한 source 당 ROI 하나뿐인 데이터셋(Ds539)은 grouped==ungrouped 이라
        stock 에 위임한다. 기존 leaky split 은 `.leaky.bak` 로 백업 후 재생성한다.
        """
        if self.fold == "all":
            return
        from batchgenerators.utilities.file_and_folder_operations import (
            join, isfile, load_json, save_json,
        )
        from nnunetv2.training.dataloading.utils import get_case_identifiers
        from nnunetv2.utilities.crossval_split import generate_crossval_split

        splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
        keys = sorted(get_case_identifiers(self.preprocessed_dataset_folder))
        groups: dict[str, list[str]] = {}
        for k in keys:
            groups.setdefault(self._source_case_from_identifier(k), []).append(k)
        multi_roi = any(len(v) > 1 for v in groups.values())

        if isfile(splits_file):
            existing = load_json(splits_file)
            if self._splits_group_consistent(existing, self._source_case_from_identifier):
                return  # 이미 group-consistent → 그대로 사용
            if not multi_roi:
                return  # 단일 ROI 데이터셋은 leakage 불가 → 그대로 사용
            backup = splits_file + ".leaky.bak"
            if not isfile(backup):
                os.replace(splits_file, backup)
            self.print_to_log_file(
                f"[FIX:C1] 기존 splits_final.json 이 source-case leakage 를 포함 → "
                f"백업({backup}) 후 grouped split 재생성"
            )
        elif not multi_roi:
            return  # 단일 ROI 데이터셋 → stock 의 ungrouped split 과 동일하므로 위임

        group_ids = sorted(groups.keys())
        # n_splits is env-configurable so a final model can train on 90/10 (n_splits=10, fold0)
        # over ALL source cases instead of the default 80/20 5-fold. Leak-free grouping is
        # preserved either way (split is over source-case group IDs, seed fixed).
        _n_splits = int(os.environ.get("PENGWIN_N_SPLITS", "5"))
        group_splits = generate_crossval_split(group_ids, seed=12345, n_splits=_n_splits)
        splits = []
        for gs in group_splits:
            tr = sorted(k for g in gs["train"] for k in groups[g])
            val = sorted(k for g in gs["val"] for k in groups[g])
            splits.append({"train": tr, "val": val})
        save_json(splits, splits_file)
        self.print_to_log_file(
            f"[FIX:C1] source-case grouped {len(splits)}-fold split 생성: "
            f"{len(group_ids)} source cases → {len(keys)} ROI samples → {splits_file}"
        )

    def do_split(self):
        """Grouped split, then stock read. PengwinTrainer overrides to add case-lock."""
        self._ensure_grouped_splits_file()
        return super().do_split()


class PengwinTrainer(_GroupedSplitMixin, nnUNetTrainer):
    """현재 사용 중인 split-anatomy trainer — DC+CE baseline, SGD+Cosine, ES patience 50.

    V2 계약 사항:
        - Loss는 기본적으로 DC+CE. Boundary/Tversky 항은 명시적이고 환경 변수로
          선택되는 ablation이므로, baseline 실험은 비교 가능한 상태로 유지된다.
        - CE class weight는 명시적으로 주거나 `PENGWIN_CE_CLASS_WEIGHTS=auto`로 설정한다.
        - 단순 mirror가 해부학적 라벨을 오염시키는 Pelvic 데이터셋에서는 좌우 mirror를 끈다.
    """

    NUM_EPOCHS_DEFAULT = 1500
    ES_PATIENCE = 50
    ES_MIN_DELTA = 5e-3   # v6: 1e-3 → 5e-3 (무한 climb 방지)
    ES_MIN_EPOCHS = 100
    WARMUP_EPOCHS = 30

    # Dataset532는 작은 Sacrum/LH/RH 영역을 포함하기 때문에 foreground patch를 더 공격적으로
    # oversample한다. Contact-LegacyFuse는 sparse한 core/contact-surface class를 가지고 있어서,
    # class-3 / class-4를 명시적으로 추가 sampling한다.
    # [cleanup 2026-06-12] Dataset532 (retired) removed; no active dataset
    # currently oversamples at 0.50 via this set, so it is empty. The `elif
    # ds_name in self.PELVIC_DATASETS_OVERSAMPLE_50` branch simply never fires
    # (falls through to the 0.33 default). Re-add a row here to re-enable.
    PELVIC_DATASETS_OVERSAMPLE_50: set[str] = set()
    ABBC_OFFICIAL_DATASETS: set[str] = set()

    # BoundaryDoU는 기본 비활성. PENGWIN 2024 상위 CT 레시피는 DC+CE를 사용하면서
    # boundary 부담을 representation/postprocess가 떠안는 구조였다. 그래서 BD는
    # 조용히 켜지는 기본값이 아니라 "측정된 ablation"으로 다룬다.
    USE_BOUNDARY_LOSS = False
    BD_WEIGHT = 0.0
    USE_TVERSKY_LOSS = False
    TV_WEIGHT = 0.0
    TV_ALPHA = 0.3
    TV_BETA = 0.7
    TV_ACTIVE_CLASSES: tuple[int, ...] | None = None
    TV_CLASS_WEIGHTS: tuple[float, ...] | None = None
    # CE class weight는 기본 비활성이다. "auto"로 두면 labelsTr를 스캔한다.
    USE_CLASS_WEIGHTS = False
    CE_CLASS_WEIGHTS: dict | str | None = None
    CLASS_WEIGHT_CLIP = (0.5, 2.0)
    DEFAULT_LOSS_PROFILE = "dc_ce"
    DEFAULT_CE_CLASS_WEIGHTS = "off"
    DEFAULT_OVERSAMPLE_PROFILE = "default"

    # 좌우 비대칭이 중요한 데이터셋(LH 라벨과 RH 라벨이 따로 있는 경우) — x-mirror를 비활성화한다.
    # Dataset532는 LeftHip과 RightHip을 모두 포함하므로 단순 mirror를 하면 해부학적으로 좌우가
    # 뒤바뀐 이미지가 만들어지는데 라벨은 바뀌지 않은 채로 남는다.
    DISABLE_X_MIRROR_DATASETS = {
        # [cleanup 2026-06-12] Dataset532 (retired) removed. The active Ds539
        # entry below is the live laterality fix and MUST stay (utils QA asserts
        # 539 ∈ this set).
        # [V0.x][FIX:LR][2026-06-02] Ds539 — diag_hip_precision 조사 결과 hip 저조의
        # 87.6%가 좌우 hip 스왑(laterality 혼동)이고, axis-2(L/R) mirror augmentation 이
        # L↔R 교환을 학습시키는 직접 원인. axis-2 mirror 비활성화.
        "Dataset539_PelvicFemurAnatomyV3",
    }

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=unpack_dataset, device=device)
        self.num_epochs = int(os.environ.get("PENGWIN_NUM_EPOCHS", self.NUM_EPOCHS_DEFAULT))
        # [QA][Test:runtime_smoke]
        # 이 override들은 의도적으로 일반적인 형태로 두었다. 그래야 stock trainer로도
        # V5의 1-epoch / 40-epoch case-locked 테스트를 돌릴 수 있다. opt-in 방식이라
        # 환경 변수가 명시되지 않으면 평소 학습에 영향을 주지 않는다.
        if os.environ.get("PENGWIN_TRAIN_ITERS"):
            self.num_iterations_per_epoch = int(os.environ["PENGWIN_TRAIN_ITERS"])
        if os.environ.get("PENGWIN_VAL_ITERS"):
            self.num_val_iterations_per_epoch = int(os.environ["PENGWIN_VAL_ITERS"])
        # [REPRO][Risk:Major][Scope:optimizer_ablation]
        # 실험에서 PENGWIN_INITIAL_LR을 명시하지 않는 한 nnU-Net 기본값인 1e-2 SGD LR을 유지한다.
        # V6 case003 진단에서 warmup이 LR을 올리면서 support가 붕괴하는 것을 봤기 때문에,
        # 이 opt-in override로 target/model/loss/decoder를 건드리지 않고 optimizer 안정성만
        # 단독으로 테스트할 수 있게 했다.
        self.initial_lr = float(os.environ.get("PENGWIN_INITIAL_LR", "1e-2"))
        self.weight_decay = 3e-5
        ds_name = self.plans_manager.dataset_name
        oversample_profile = os.environ.get(
            "PENGWIN_OVERSAMPLE_PROFILE", self.DEFAULT_OVERSAMPLE_PROFILE
        ).strip().lower()
        if ds_name in self.ABBC_OFFICIAL_DATASETS:
            if oversample_profile in {"abbc_contact_energy_v1", "contact_fuse_v1"}:
                self.oversample_foreground_percent = 1.00
            elif oversample_profile == "weak0123":
                self.oversample_foreground_percent = 0.66
            else:
                self.oversample_foreground_percent = 0.50
        elif ds_name in self.PELVIC_DATASETS_OVERSAMPLE_50:
            if oversample_profile == "weak0123":
                self.oversample_foreground_percent = 0.66
            else:
                self.oversample_foreground_percent = 0.50
        else:
            self.oversample_foreground_percent = 0.33
        self._pengwin_oversample_profile = oversample_profile
        self._configure_loss_ablation_from_env()
        # Early stop 상태 변수
        self._es_best_dice = -1.0
        self._es_no_improve = 0
        self._es_triggered = False

    def _pengwin_case_lock(self, train_keys: list[str], val_keys: list[str]) -> tuple[list[str], list[str]]:
        """V5 root-cause 테스트를 위해 source-case 단위 overfit split을 강제하는 옵션 메서드.

        [DATA][Leakage:case_lock]
        Dataset537 V5에서는 source CT 하나당 여러 ROI sample이 나온다. case-locked overfit을
        제대로 하려면 같은 source case의 ROI identifier들을 train과 val 양쪽에 모두 포함시켜야
        한다. 그래야 patch validation이 다른 source case를 health check에 끼워넣지 못한다.
        이 동작은 `PENGWIN_OVERFIT_CASES` 또는 `PENGWIN_BICM_V5_CASES`가 명시적으로 설정된
        경우에만 켜진다.
        """
        raw = (
            os.environ.get("PENGWIN_OVERFIT_CASES", "").strip()
            or os.environ.get("PENGWIN_BICM_V5_CASES", "").strip()
        )
        if not raw:
            return train_keys, val_keys
        wanted = {part.strip().zfill(3) for part in raw.replace(",", " ").split() if part.strip()}
        all_keys = sorted(set(train_keys) | set(val_keys))
        selected = [
            key for key in all_keys
            if self._source_case_from_identifier(key) in wanted
        ]
        if not selected:
            raise RuntimeError(
                f"PENGWIN_OVERFIT_CASES={sorted(wanted)} matched no identifiers in "
                f"{self.plans_manager.dataset_name}"
            )
        self.print_to_log_file(
            "[PengwinTrainer] case-lock overfit split active: "
            f"cases={sorted(wanted)} identifiers={selected}"
        )
        return selected, selected

    def do_split(self):
        # grouped split (via _GroupedSplitMixin) → stock read → optional case-lock overfit.
        self._ensure_grouped_splits_file()
        train_keys, val_keys = nnUNetTrainer.do_split(self)
        return self._pengwin_case_lock(list(train_keys), list(val_keys))

    def _build_loss(self):
        """DC+CE를 기본으로 하고, 명시적이고 감사(audit) 가능한 loss ablation을 얹어서 구성한다.

        알고리즘:
            1. 부모 클래스와 동일하게 base DC_and_CE_loss를 만든다
               (batch_dice / smooth / DDP smoothing / ignore_label 의미를 그대로 보존).
               CE_CLASS_WEIGHTS가 설정돼 있으면 CE 부분에 `weight` 인자를 넘긴다.
            2. boundary/Tversky 프로파일이 명시적으로 선택돼 있으면 해당 compound loss로 감싼다.
            3. nnU-Net의 torch.compile 경로는 loss.dc가 Dice 모듈일 거라고 기대하기 때문에,
               우리 wrapper는 속성 통과(attribute pass-through)로 이를 노출한다.
            4. 마지막으로 부모처럼 DeepSupervisionWrapper로 감싼다 (multi-resolution).

        MIC-DKFZ의 loss 코드 패스를 서브클래싱하지 않는 이유:
            DC_and_CE_loss는 ignore_label과 DDP edge case들을 내부에서 다루는데,
            이걸 다시 구현하면 미묘한 버그가 생길 위험이 있다. 그래서 대체가 아니라 구성(compose)한다.

        Fallback:
            `loss.py` import가 실패하거나, USE_BOUNDARY_LOSS=False이거나, label manager가
            region 기반(multi-label, 예: 중첩된 해부 구조)이면 부모의 DC+CE로 폴백한다.
            절대 조용히 학습을 망가뜨리지 않는다.
        """
        # Region 기반 라벨 (multi-label)이면 → 부모 구현으로 위임 (boundary loss 없이).
        if self.label_manager.has_regions:
            return super()._build_loss()

        ce_kwargs = {}
        cw_tensor = None
        n_classes = self.label_manager.num_segmentation_heads
        if self.USE_CLASS_WEIGHTS and self.CE_CLASS_WEIGHTS is not None:
            cw = self._resolve_class_weights(n_classes)
            if cw is not None:
                cw_tensor = torch.as_tensor(cw, dtype=torch.float32, device=self.device)
                ce_kwargs["weight"] = cw_tensor
                self.print_to_log_file(
                    f"[PengwinTrainer v7] CE class weights: "
                    f"{[f'{w:.2f}' for w in cw]}"
                )

        loss = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            ce_kwargs,
            weight_ce=1, weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        if (
            self.USE_TVERSKY_LOSS
            and _BOUNDARY_LOSS_AVAILABLE
            and self.TV_WEIGHT > 0
        ):
            loss = DC_CE_TV_BD_loss(
                dc_ce_loss=loss,
                n_classes=n_classes,
                weight_dc_ce=1.0,
                weight_tversky=self.TV_WEIGHT,
                tversky_alpha=self.TV_ALPHA,
                tversky_beta=self.TV_BETA,
                weight_bd=self.BD_WEIGHT if self.USE_BOUNDARY_LOSS else 0.0,
                tversky_active_classes=self.TV_ACTIVE_CLASSES,
                tversky_class_weights=self.TV_CLASS_WEIGHTS,
            )
            self.print_to_log_file(
                f"[PengwinTrainer] Loss = DC+CE + {self.TV_WEIGHT}·Tversky"
                f"(alpha={self.TV_ALPHA}, beta={self.TV_BETA})"
                f" active={self.TV_ACTIVE_CLASSES or 'fg'}"
                f" class_weights={self.TV_CLASS_WEIGHTS or 'uniform'}"
                f" + {self.BD_WEIGHT if self.USE_BOUNDARY_LOSS else 0.0}·BoundaryDoU3D"
                f"  (n_classes={n_classes})"
            )
        elif self.USE_BOUNDARY_LOSS and _BOUNDARY_LOSS_AVAILABLE and self.BD_WEIGHT > 0:
            loss = DC_CE_BD_loss(
                dc_ce_loss=loss,
                n_classes=n_classes,
                weight_dc_ce=1.0,
                weight_bd=self.BD_WEIGHT,
            )
            self.print_to_log_file(
                f"[PengwinTrainer] Loss = DC+CE + {self.BD_WEIGHT}·BoundaryDoU3D"
                f"  (n_classes={n_classes})"
            )
        elif (self.USE_BOUNDARY_LOSS or self.USE_TVERSKY_LOSS) and not _BOUNDARY_LOSS_AVAILABLE:
            self.print_to_log_file(
                "[PengwinTrainer] WARN: custom loss requested but "
                "code_task1/loss.py import failed — falling back to DC+CE."
            )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            ds_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(ds_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def _configure_loss_ablation_from_env(self) -> None:
        """환경 변수로 선택된 loss ablation을 적용한다.

        지원하는 값:
            PENGWIN_LOSS_PROFILE=
                dc_ce|bd_dou_005|bd_dou_01|bd_dou_03|
                tversky_07|tversky_08|combo_tversky_bd005|
                abbc_contact_energy_v1|contact_fuse_v1|factorized_instance_v4
            PENGWIN_CE_CLASS_WEIGHTS=off|auto

        기본 학습은 그대로 비교 가능한 상태로 두면서, 코드 수정 없이도 재현 가능한
        loss 실험을 돌릴 수 있게 해준다.
        """
        profile = os.environ.get("PENGWIN_LOSS_PROFILE", self.DEFAULT_LOSS_PROFILE).strip().lower()
        self.USE_BOUNDARY_LOSS = False
        self.BD_WEIGHT = 0.0
        self.USE_TVERSKY_LOSS = False
        self.TV_WEIGHT = 0.0
        self.USE_CONTACT_FUSE_OBJECTIVE = False
        self.USE_CONTACT_ENERGY_OBJECTIVE = False
        self.USE_CONTACT_INSTANCE_OBJECTIVE = False
        self.USE_BICM_V5_SPARSE_OBJECTIVE = False
        self.USE_BICM_V5_SUPPORT_GEOMETRY_OBJECTIVE = False
        self.USE_BICM_V62_SEMANTIC_BOUNDARY_OBJECTIVE = False
        self.USE_BICM_V64_TOPOLOGY_PRECISION_OBJECTIVE = False
        self.USE_BICM_V65_BALANCED_CONTACT_CORE_OBJECTIVE = False
        self.USE_BICM_V67_SEMANTIC_EDGE_PAIR_OBJECTIVE = False
        self.TV_ALPHA = 0.3
        self.TV_BETA = 0.7
        self.TV_ACTIVE_CLASSES = None
        self.TV_CLASS_WEIGHTS = None
        if profile in ("", "dc_ce", "baseline"):
            pass
        elif profile == "bd_dou_005":
            self.USE_BOUNDARY_LOSS = True
            self.BD_WEIGHT = 0.05
        elif profile == "bd_dou_01":
            self.USE_BOUNDARY_LOSS = True
            self.BD_WEIGHT = 0.1
        elif profile == "bd_dou_03":
            self.USE_BOUNDARY_LOSS = True
            self.BD_WEIGHT = 0.3
        elif profile == "tversky_07":
            self.USE_TVERSKY_LOSS = True
            self.TV_WEIGHT = 0.5
            self.TV_ALPHA = 0.3
            self.TV_BETA = 0.7
        elif profile == "tversky_08":
            self.USE_TVERSKY_LOSS = True
            self.TV_WEIGHT = 0.5
            self.TV_ALPHA = 0.2
            self.TV_BETA = 0.8
        elif profile == "combo_tversky_bd005":
            self.USE_TVERSKY_LOSS = True
            self.TV_WEIGHT = 0.35
            self.TV_ALPHA = 0.3
            self.TV_BETA = 0.7
            self.USE_BOUNDARY_LOSS = True
            self.BD_WEIGHT = 0.05
        elif profile == "abbc_contact_energy_v1":
            # Contact-energy V2: 더 이상 class 3을 폭넓은 recall 대상으로 보지 않는다.
            # 후처리에서 energy ridge로 쓰는 좁은 contact/fracture surface이기 때문에,
            # 약간의 false negative보다 false positive가 훨씬 더 해롭다.
            # marker seed의 안정성을 위해 core는 그대로 활성 상태로 둔다.
            self.USE_TVERSKY_LOSS = True
            self.TV_WEIGHT = 0.55
            self.TV_ALPHA = 0.70
            self.TV_BETA = 0.30
            self.TV_ACTIVE_CLASSES = (2, 3)
            self.TV_CLASS_WEIGHTS = (0.0, 0.0, 1.75, 2.25)
        elif profile == "contact_fuse_v1":
            self.USE_CONTACT_FUSE_OBJECTIVE = True
            self.USE_CONTACT_ENERGY_OBJECTIVE = False
        elif profile == "bicm_v5_sparse":
            self.USE_BICM_V5_SPARSE_OBJECTIVE = True
        elif profile == "bicm_v5_support_geometry":
            self.USE_BICM_V5_SUPPORT_GEOMETRY_OBJECTIVE = True
        elif profile == "bicm_v62_semantic_boundary":
            self.USE_BICM_V62_SEMANTIC_BOUNDARY_OBJECTIVE = True
        elif profile == "bicm_v64_topology_precision":
            self.USE_BICM_V64_TOPOLOGY_PRECISION_OBJECTIVE = True
        elif profile == "bicm_v65_balanced_contact_core":
            self.USE_BICM_V65_BALANCED_CONTACT_CORE_OBJECTIVE = True
        elif profile == "bicm_v67_semantic_edge_pair":
            self.USE_BICM_V67_SEMANTIC_EDGE_PAIR_OBJECTIVE = True
        elif profile in {
            "bicm_v6_factorized",
            "bicm_v250_oracle_aligned_direct",
            "bicm_v6_precision",
            "bicm_v6_support_core",
            "bicm_v6_support_precision",
            "bicm_v6_support_surface",
            "bicm_v7_contact_balanced",
            "bicm_v7_contact_ranked",
            "bicm_v7_contact_presence",
            "bicm_v7_contact_ratio",
            "bicm_v7_contact_dense",
            "bicm_v8_contact_contour",
            "bicm_v9_contact_energy_pair",
            "bicm_v9_contact_energy_precision",
            "bicm_v10_adaptive_topology",
            "bicm_v12_contact_persistent",
            "bicm_v13_contact_precision",
            "bicm_v14_contact_curriculum",
            "bicm_v15_contact_presence_gate",
            "bicm_v16_roi_presence_classifier",
            "bicm_v19_roi_calibrated_presence",
            "bicm_v22_core_marker_separation",
            "bicm_v23_staged_core_marker",
            "bicm_v24_contact_preserved_core_marker",
            "bicm_v25_coupled_contact_core_marker",
            "bicm_v26_contact_tolerant_precision",
            "bicm_v28_isolated_contact_precision",
            "bicm_v29_gentle_isolated_contact_precision",
            "bicm_v30_soft_contact_ridge",
            "bicm_v31_local_contrastive_contact",
            "bicm_v32_memory_efficient_local_contrast",
            "bicm_v34_compact_core_marker",
            "bicm_v35_contact_precision_compact_core",
            "bicm_v36_staged_contact_precision_compact_core",
            "bicm_v37_asymmetric_contact_compact_core",
            "bicm_v38_edge_affinity",
            "bicm_v39_edge_primary",
            "bicm_v40_edge_core_separation",
            "bicm_v41_instance_core_edge",
            "bicm_v42_instance_core_edge_primary",
            "bicm_v43_edge_contact_viability",
            "bicm_v44_edge_local_rank",
            "bicm_v45_edge_candidate_save",
            "bicm_v46_staged_edge_contact",
            "bicm_v47_separated_contact_head",
            "bicm_v48_support_aware_contact_branch",
            "bicm_v49_decoder_feature_contact_branch",
            "bicm_v50_decoder_feature_contact_phase",
            "bicm_v51_dense_edge_cost",
            "bicm_v52_dense_edge_core_topology",
            "bicm_v53_fragment_marker_core",
            "bicm_v54_sparse_head_balanced",
            "bicm_v55_calibrated_sparse_head",
            "bicm_v56_phased_marker_contact",
            "bicm_v57_decoder_feature_marker_phase",
            "bicm_v58_heatmap_marker_contact",
            "bicm_v59_core_preserving_contact",
            "bicm_v60_strict_core_topology",
            "bicm_v61_peak_seed",
            "bicm_v68_semantic_topology",
            "bicm_v69_topology_calibrated",
            "bicm_v70_topology_consistency",
            "bicm_v71_edge_cut_primary",
            "bicm_v72_logit_calibrated",
            "bicm_v73_instance_topology",
            "bicm_v74_adaptive_instance_topology",
            "bicm_v75_edge_precision_seed_topology",
            "bicm_v76_gentle_edge_precision",
            "bicm_v77_edge_recall_precision_curriculum",
            "bicm_v78_core_anchored_edge_curriculum",
            "bicm_v79_duplicate_seed_edge",
            "bicm_v80_topology_state_adaptive",
            "bicm_v81_core_stable_edge_precision",
            "bicm_v82_seed_recall_edge_separation",
            "bicm_v83_seed_safe_edge_calibration",
            "bicm_v84_dual_head_contact_calibration",
            "bicm_v85_semantic_gate_contact",
            "bicm_v86_eval_band_edge_primary",
            "bicm_v87_eval_band_semantic_gate",
            "bicm_v88_core_preserving_band_precision",
            "bicm_v89_band_false_only_precision",
            "bicm_v90_semantic_band_product",
            "bicm_v91_topology_aware_edge_balance",
            "bicm_v93_eval_aligned_support_contact",
            "bicm_v94_final_row_support_contact",
            "bicm_v95_decoder_feature_contact",
            "bicm_v96_dense_band_gate",
            "bicm_v97_positive_dense_band_gate",
            "bicm_v98_teacher_distilled_dense_gate",
            "bicm_v99_adaptive_dual_margin_dense_gate",
            "bicm_v100_negative_balanced_dense_gate",
            "bicm_v101_dual_field_product_gate",
            "bicm_v102_distance_rank_dense_gate",
            "bicm_v103_semantic_distance_contact",
            "bicm_v104_teacher_semantic_contact",
            "bicm_v105_offset_assignment",
            "bicm_v106_offset_attractor",
            "bicm_v107_radial_support_offset",
            "bicm_v108_watershed_barrier",
            "bicm_v109_precision_locked_recall",
            "bicm_v110_semantic_geometry_bridge",
            "bicm_v111_joint_support_product",
            "bicm_v112_edge_graph_assignment",
            "bicm_v113_adaptive_boundary_product",
            "bicm_v114_encoder_adapter_boundary_product",
            "bicm_v115_semantic_oracle_adapter",
            "bicm_v116_graph_cost_separator",
            "bicm_v117_support_gate_semantic_contact",
            "bicm_v118_warm_support_gate_semantic_contact",
            "bicm_v119_all_network_adaptive_boundary",
            "bicm_v120_same_fragment_affinity",
            "bicm_v121_warm_same_fragment_affinity",
            "bicm_v122_warm_edge_product_precision",
            "bicm_v123_high_recall_gate_cleanup",
            "bicm_v124_support_conditioned_edge_precision",
            "bicm_v125_saturated_edge_dense_gate",
            "bicm_v126_local_adjacency_product",
            "bicm_v128_support_bridge_suppression",
            "bicm_v129_affinity_sharpening",
            "bicm_v130_support_topology_repair",
            "bicm_v131_all_network_support_topology_repair",
            "bicm_v132_support_veto_gate",
            "bicm_v133_support_veto_semantic",
            "bicm_v134_support_veto_gate_highlr",
            "bicm_v135_support_veto_gate_ultralr",
            "bicm_v136_support_topology_highlr",
            "bicm_v137_support_topology_ultralr",
            "bicm_v138_support_topology_midlr",
            "bicm_v139_support_topology_uppermidlr",
            "bicm_v140_support_topology_lowbracket",
            "bicm_v141_support_topology_highbracket",
            "bicm_v142_support_topology_coreseed_lowlr",
            "bicm_v143_support_topology_coreseed_midlr",
            "bicm_v144_strong_coreseed_lowlr",
            "bicm_v145_strong_coreseed_midlr",
            "bicm_v146_fragment_seed_presence_lowlr",
            "bicm_v147_fragment_seed_presence_midlr",
            "bicm_v148_core_heatmap_seed_lowlr",
            "bicm_v149_core_heatmap_seed_midlr",
            "bicm_v150_core_only_heatmap_seed_midlr",
            "bicm_v151_core_only_heatmap_seed_highlr",
            "bicm_v152_core_only_center_seed_midlr",
            "bicm_v153_core_heatmap_center_seed_midlr",
            "bicm_v154_core_only_center_seed_sampler",
            "bicm_v155_core_heatmap_center_seed_sampler",
            "bicm_v158_core_only_center_seed_affinity_sampler",
            "bicm_v159_core_heatmap_center_seed_affinity_sampler",
        }:
            # [QC][Invariant:custom_trainer_loss_dispatch]
            # V6 이후 BICM trainer들은 `_build_loss`를 override해서 custom sigmoid 목적함수를 쓴다.
            # base trainer는 단지 이 프로파일 이름들을 알아만 보면 된다 — 그래야 audit log에
            # 서브클래스가 진짜 loss를 만들기 전에 "DC+CE로 fallback"이라는 잘못된 메시지가 안 찍힌다.
            pass
        elif profile in {
            "contact_instance_v1",
            "contact_instance_exclusive_v1",
            "contact_instance_sigmoid_v1",
            "factorized_instance_v4",
            "factorized_instance_v4_seedfit",
            "factorized_instance_v4_staged",
            "factorized_instance_v4_v42",
            "factorized_instance_v4_heatseed",
            "factorized_instance_v4_v43",
        }:
            self.USE_CONTACT_INSTANCE_OBJECTIVE = True
            self.USE_CONTACT_FUSE_OBJECTIVE = False
            self.USE_CONTACT_ENERGY_OBJECTIVE = False
        else:
            self.print_to_log_file(
                f"[PengwinTrainer] unknown PENGWIN_LOSS_PROFILE={profile!r}; using DC+CE"
            )

        cw = os.environ.get("PENGWIN_CE_CLASS_WEIGHTS", self.DEFAULT_CE_CLASS_WEIGHTS).strip().lower()
        if cw == "auto":
            self.USE_CLASS_WEIGHTS = True
            self.CE_CLASS_WEIGHTS = "auto"
        elif cw in ("", "off", "none", "false", "0"):
            self.USE_CLASS_WEIGHTS = False
            self.CE_CLASS_WEIGHTS = None
        self.print_to_log_file(
            f"[PengwinTrainer] loss_profile={profile or 'dc_ce'} "
            f"boundary={self.BD_WEIGHT} "
            f"tversky={self.TV_WEIGHT}@({self.TV_ALPHA},{self.TV_BETA}) "
            f"ce_weights={self.CE_CLASS_WEIGHTS or 'off'} "
            f"oversample_profile={self._pengwin_oversample_profile} "
            f"oversample_fg={self.oversample_foreground_percent:.2f}"
        )

    def _resolve_class_weights(self, n_classes: int) -> np.ndarray | None:
        """CE_CLASS_WEIGHTS 속성을 (n_classes,) shape의 numpy 배열로 변환한다.

        받을 수 있는 형식:
            - None: None을 반환한다 (호출자가 weight 적용을 건너뛴다)
            - dict {class_idx: weight}: dense array로 변환하며, 누락된 키는 1.0으로 채운다
            - "auto": 생성된 nnU-Net raw label을 스캔해서 median-frequency 기반의
              clip된 weight를 계산한다.
            - np.ndarray / list: 복사 후 길이를 검증한다

        반환:
            (n_classes,) shape의 np.ndarray 또는 None.
        """
        cw = self.CE_CLASS_WEIGHTS
        if cw is None:
            return None
        if isinstance(cw, str):
            if cw == "auto":
                return self._compute_auto_class_weights(n_classes)
            return None
        if isinstance(cw, dict):
            arr = np.ones(n_classes, dtype=np.float32)
            for k, v in cw.items():
                if 0 <= int(k) < n_classes:
                    arr[int(k)] = float(v)
            return arr
        # list/array의 경우 — 길이를 검증하고 복사한다
        arr = np.asarray(cw, dtype=np.float32)
        if arr.shape != (n_classes,):
            self.print_to_log_file(
                f"[PengwinTrainer v7] CE_CLASS_WEIGHTS length {arr.shape} != "
                f"n_classes {n_classes} — ignoring."
            )
            return None
        return arr

    def _compute_auto_class_weights(self, n_classes: int) -> np.ndarray | None:
        """생성된 raw label로부터 class weight를 계산하고 audit JSON을 함께 기록한다."""
        try:
            import SimpleITK as sitk
        except ImportError:
            self.print_to_log_file("[PengwinTrainer] SimpleITK missing; CE auto weights disabled")
            return None
        dataset_name = self.plans_manager.dataset_name
        raw_root = Path(os.environ.get("nnUNet_raw", str(_ROOT / "nnunet/raw")))
        label_dir = raw_root / dataset_name / "labelsTr"
        label_paths = sorted(label_dir.glob("*.mha"))
        if not label_paths:
            self.print_to_log_file(f"[PengwinTrainer] no labelsTr found at {label_dir}; CE weights disabled")
            return None
        counts = np.zeros(n_classes, dtype=np.float64)
        for p in label_paths:
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
            bc = np.bincount(arr.ravel(), minlength=n_classes).astype(np.float64)
            counts += bc[:n_classes]
        mn, mx = self.CLASS_WEIGHT_CLIP
        counts_smooth = counts + 1.0
        freq = counts_smooth / counts_smooth.sum()
        present = counts > 0
        median = np.median(freq[present]) if present.any() else np.median(freq)
        weights = np.clip(median / freq, mn, mx).astype(np.float32)
        audit_dir = RESULT_WEIGHT
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_path = audit_dir / f"class_weights_{dataset_name}.json"
        audit_path.write_text(json.dumps({
            "dataset": dataset_name,
            "n_classes": n_classes,
            "n_label_files": len(label_paths),
            "method": "median_frequency",
            "clip": [mn, mx],
            "counts": [int(x) for x in counts.tolist()],
            "weights": [float(x) for x in weights.tolist()],
        }, indent=2))
        self.print_to_log_file(f"[PengwinTrainer] CE auto weights audit: {audit_path}")
        return weights

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """Pelvic 데이터셋에서 LH↔RH 라벨 오염을 막기 위해 axis-2 mirror(L/R)를 비활성화한다.

        transpose_forward [1,0,2]를 거치면 patch 공간의 axis 2는 원본의 x축, 즉 L/R 방향이다.
        이걸 mirror하면 LH 모양 voxel이 오른쪽에 생기는데 라벨은 여전히 LH라서 노이즈가 된다.
        """
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        ds_name = self.plans_manager.dataset_name
        if ds_name in self.DISABLE_X_MIRROR_DATASETS:
            # mirror_axes에서 axis 2를 제거한다
            new_axes = tuple(a for a in mirror_axes if a != 2)
            self.print_to_log_file(
                f"[PengwinTrainer] {ds_name} → disable axis-2 mirror (L/R asymmetry). "
                f"mirror_axes: {mirror_axes} → {new_axes}"
            )
            self.inference_allowed_mirroring_axes = new_axes
            mirror_axes = new_axes
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes


    def get_dataloaders(self):
        """부모 nnU-Net의 기본 dataloader를 그대로 사용한다.

        활성 anatomy(Dataset539) 및 baseline 경로는 모두 stock nnU-Net dataloader를 쓴다.
        """
        return super().get_dataloaders()

    def configure_optimizers(self):
        """SGD + Cosine annealing + warmup (PENGWIN 2024 표준)."""
        optimizer = SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                        momentum=0.99, nesterov=True)
        # Warmup 후 cosine 스케줄 적용
        warmup = self.WARMUP_EPOCHS

        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            # 남은 epoch 동안 peak에서 0까지 cosine 감소
            import math
            progress = (epoch - warmup) / max(1, self.num_epochs - warmup)
            return 0.5 * (1 + math.cos(math.pi * progress))

        lr_scheduler = LambdaLR(optimizer, lr_lambda)
        return optimizer, lr_scheduler

    def run_training(self):
        """학습 중간에 ES(Early Stop)가 num_epochs를 줄일 수 있도록 while-loop로 오버라이드한다.

        부모 클래스의 `for epoch in range(start, num_epochs)`는 루프 생성 시점에 num_epochs를
        한 번만 평가하기 때문에, 이후 self.num_epochs를 바꿔도 반영되지 않는다.
        그래서 매 반복마다 self.num_epochs를 다시 확인하는 while-loop를 사용한다.
        """
        self.on_train_start()
        while self.current_epoch < self.num_epochs:
            self.on_epoch_start()
            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)
            self.on_epoch_end()
        self.on_train_end()

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)

    def on_validation_epoch_end(self, val_outputs):
        super().on_validation_epoch_end(val_outputs)
        # EMA pseudo Dice 기준 Early stop 판단
        completed = self.current_epoch + 1
        if completed < self.ES_MIN_EPOCHS:
            return
        ema = float(self.logger.my_fantastic_logging["ema_fg_dice"][-1])
        if ema > self._es_best_dice + self.ES_MIN_DELTA:
            self._es_best_dice = ema
            self._es_no_improve = 0
        else:
            self._es_no_improve += 1
            if self._es_no_improve >= self.ES_PATIENCE and not self._es_triggered:
                self._es_triggered = True
                self.print_to_log_file(
                    f"Early stopping: no EMA-Dice improvement for "
                    f"{self.ES_PATIENCE} epochs. Best EMA Dice = {self._es_best_dice:.4f} "
                    f"at epoch {completed - self.ES_PATIENCE}. "
                    f"Stopping at epoch {completed}."
                )
                self.num_epochs = completed  # 메인 루프에 종료 신호를 보낸다


# =============================================================================
# [V0.x][PHASE 2A][2026-05-31] BADB: Boundary Attention Decoder Branch
# =============================================================================
#
# V0~V72 ablation 의 종합 교훈: contact precision/recall 트레이드오프는 BICM 5-class
# softmax + factorized head 의 fundamental 한계. V0.x 의 진단 #4 에서 ABBC 의 contact
# 비율이 0.03% 수준으로 sparse class collapse 가 확인됨.
#
# PENGWIN 2024 1위 (MIC-DKFZ) 의 차별점: ABBC + medial axis 기반 dynamic boundary
# thickness. 그러나 architecture 측 inductive bias 없이는 (= V291 처럼 단순 loss
# weight 변경만으로는) 0.03% sparse contact 학습이 어려움.
#
# Phase 1 (Dynamic boundary, utils.compute_abbc_official_target_dynamic): target 측
#   contact ratio 를 0.03% → 1-3% 로 자연 확장.
#
# Phase 2A (본 V300, BADB): network 측 boundary-aware inductive bias 도입. 기존
#   V291 의 4-class ABBC softmax 출력을 base 로 하되, 입력 CT 와 함께 refinement
#   conv block 을 추가하여 boundary 채널을 명시적으로 학습시킨다. base + delta
#   residual 구조이므로 학습 초기에는 V291 와 동일하게 시작, 학습 진행에 따라
#   boundary refinement 가 활성화됨.
# =============================================================================


# =============================================================================
# (위는 V300 BADB Phase 2A skeleton 종료)
# =============================================================================


# =============================================================================
# [V0.x][BACKBONE][2026-06-01] STU-Net-B 백본 trainer (anatomy + ABBC fracture)
# =============================================================================
def _build_stunet_from_plan(variant: str, arch_init_kwargs: dict,
                            num_input_channels: int, num_output_channels: int,
                            enable_deep_supervision: bool = True):
    """nnU-Net 2.5.2 plan 의 arch_init_kwargs 에서 strides 를 뽑아 STUNet 을 구성한다.

    STUNet 원본은 configuration_manager.pool_op_kernel_sizes[1:] 를 strides 로 썼다.
    2.5.2 에서는 동일 값이 arch_init_kwargs['strides'] 에 들어있다(첫 entry=[1,1,1]=stage0
    no-downsample). [1:] 로 다운샘플 strides 5개를 취하고, 6-stage(dims 6개)에 맞춰
    cap(5)/pad([1,1,1]) 한다. conv 가중치는 stride 무관 shape 이므로 사전학습 가중치가
    그대로 로드된다.
    """
    if not _STUNET_AVAILABLE:
        raise RuntimeError("STUNet 백본 import 실패 — model.py 의 STUNet 정의 확인 필요")
    strides = [list(s) for s in arch_init_kwargs["strides"][1:]]
    if len(strides) > 5:
        strides = strides[:5]
    while len(strides) < 5:
        strides.append([1, 1, 1])
    kernel_sizes = [[3, 3, 3]] * 6
    v = STUNET_VARIANTS[variant]
    return STUNet(num_input_channels, num_output_channels,
                  depth=v["depth"], dims=v["dims"],
                  pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes,
                  enable_deep_supervision=enable_deep_supervision)


def _maybe_apply_stunet_warmstart(trainer):
    """[V0.x][FIX:DDP][2026-06-01] env `PENGWIN_STUNET_PRETRAINED` 의 STU-Net 사전학습
    가중치를 trainer.network 에 warm-start 한다.

    DDP-safe: nnUNet 의 `-pretrained_weights` 경로는 단일 GPU(부모 프로세스)에서만
    monkey-patch 가능하다 — `num_gpus>1` 이면 `mp.spawn(run_ddp)` 로 뜨는 자식 프로세스가
    run_training 을 fresh import 하므로 부모의 patch 가 적용되지 않아 STUNet 기본 로더가
    깨진다. 본 훅은 각 프로세스의 initialize() 끝에서 직접 적용하므로 DDP child 에서도
    동작한다. 따라서 STU-Net warm-start 는 `-pretrained_weights` 대신
    `PENGWIN_STUNET_PRETRAINED` 환경변수를 사용한다(둘 다 주면 이중 적용되니 금지).

    모든 rank 가 동일 파일을 로드 → 가중치 일관(DDP 시작 불변식 유지). 로더는 DDP/compile
    래퍼를 자동 언랩한다. continue(-c) 체크포인트는 initialize() 이후 별도로 로드되어
    warm-start 를 덮어쓰므로(재개 시 올바름) 상호 안전하다.
    """
    import os as _os
    path = _os.environ.get("PENGWIN_STUNET_PRETRAINED", "").strip()
    if not path or getattr(trainer, "_stunet_warmstart_done", False):
        return
    if not _STUNET_AVAILABLE:
        trainer.print_to_log_file("[STU-Net warm-start] stunet 모듈 없음 — 스킵")
        return
    from model import load_stunet_pretrained_weights
    inflate = _os.environ.get("PENGWIN_STUNET_INFLATE", "ct0")
    stats = load_stunet_pretrained_weights(trainer.network, path, inflate=inflate)
    trainer._stunet_warmstart_done = True
    trainer.print_to_log_file(f"[STU-Net warm-start] {path}: {stats}")


# =============================================================================
# [V0.x][PARTIAL-LABEL][2026-06-02] Marginal Dice+CE loss (Shi et al., MedIA 2021)
# =============================================================================
class MarginalDiceCELoss(nn.Module):
    """Partial-label marginal Dice+CE — 부분 라벨 충돌 supervision 을 loss 레벨에서 근본 해결.

    조사 B 결론: Ds539 는 pelvic(=sacrum/LHip/RHip 만 라벨) 과 femur(=femur 만 라벨) 의 disjoint
    부분 라벨이라, 보이는데 미라벨된 뼈가 'background' 로 학습돼 충돌(같은 뼈가 한 케이스에선
    sacrum, 다른 케이스에선 background). Marginal loss 는 케이스별 '미라벨 foreground 클래스' 를
    **background marginal 로 접어** 페널티를 제거한다.

    - labeled_mask[B, C] (bool): 케이스별 라벨된 클래스. c=0(bg) 은 항상 labeled 취급.
      트레이너가 매 배치 batch['keys'] → case 라벨셋으로 설정한다(None 이면 전부 labeled = 표준).
    - CE: 미라벨 fg 클래스 logit 을 {bg}∪{미라벨} logsumexp 로 묶어 marginal bg log-prob 계산
      → bg 영역(미라벨 뼈 포함)에서 미라벨 클래스 예측에 페널티 0. labeled 클래스는 표준 log-softmax.
    - Dice(do_bg=False, batch_dice=False): labeled fg 클래스만 per-sample soft dice, 미라벨 제외.
    - softmax 유지 → STU-Net TotalSeg warm-start 보존. nnUNet DC_and_CE 관례(weight 1/1, smooth 1e-5).

    DeepSupervisionWrapper 와 호환: labeled_mask 는 클래스 단위(해상도 무관)이므로 모든 DS scale 에
    동일 적용된다.
    """

    def __init__(self, num_classes: int, batch_dice: bool = False, smooth: float = 1e-5,
                 weight_ce: float = 1.0, weight_dice: float = 1.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.batch_dice = bool(batch_dice)
        self.smooth = float(smooth)
        self.weight_ce = float(weight_ce)
        self.weight_dice = float(weight_dice)
        self.labeled_mask = None  # [B, C] bool — 트레이너가 배치마다 설정

    def forward(self, logits, target):
        B, C = logits.shape[:2]
        if target.shape[1] != 1:
            target = target[:, :1]
        tgt = target.long()
        dev = logits.device
        mask = self.labeled_mask
        if mask is None:
            mask = torch.ones(B, C, dtype=torch.bool, device=dev)
        else:
            mask = mask.to(dev).bool()
            if mask.shape[0] != B:  # DDP/grad accum 등으로 B 불일치 시 안전 폴백
                mask = torch.ones(B, C, dtype=torch.bool, device=dev)
        mask = mask.clone()
        mask[:, 0] = True  # bg 항상 labeled

        # bg-supergroup = {0} ∪ {미라벨 fg}
        ch = torch.arange(C, device=dev)[None, :]              # [1, C]
        fg_unlabeled = (~mask) & (ch != 0)                     # [B, C]
        bg_group = fg_unlabeled.clone()
        bg_group[:, 0] = True                                  # [B, C]

        # ---- marginal CE ----
        logp = torch.log_softmax(logits, dim=1)               # [B, C, ...]
        lse_all = torch.logsumexp(logits, dim=1, keepdim=True) # [B, 1, ...]
        neg_inf = torch.finfo(logits.dtype).min
        bgmask = bg_group.view(B, C, *([1] * (logits.ndim - 2)))
        masked = logits.masked_fill(~bgmask, neg_inf)
        bg_lse = torch.logsumexp(masked, dim=1, keepdim=True)  # [B, 1, ...]
        logq_bg = bg_lse - lse_all                             # [B, 1, ...] log marginal bg
        gathered = torch.gather(logp, 1, tgt.clamp(0, C - 1))  # [B, 1, ...]
        target_logq = torch.where(tgt == 0, logq_bg, gathered)
        ce = -(target_logq).mean()

        # ---- marginal Dice (do_bg=False, labeled fg only) ----
        p = torch.exp(logp)
        spatial = tuple(range(2, logits.ndim))
        dice_vals = []
        for c in range(1, C):
            sel = mask[:, c]                                   # [B]
            if not bool(sel.any()):
                continue
            pc = p[:, c]                                       # [B, ...]
            gc = (tgt[:, 0] == c).to(pc.dtype)                 # [B, ...]
            sdims = tuple(range(1, pc.ndim))
            inter = (pc * gc).sum(sdims)
            denom = pc.sum(sdims) + gc.sum(sdims)
            dpc = (2 * inter + self.smooth) / (denom + self.smooth)  # [B]
            dice_vals.append(dpc[sel])
        if dice_vals:
            mean_dice = torch.cat(dice_vals).mean()
        else:
            mean_dice = logits.sum() * 0.0
        return self.weight_ce * ce - self.weight_dice * mean_dice


# =============================================================================
# [V0.x][WARN-FIX][2026-06-02] nnUNet 2.5.2 deprecated-API 경고 근본 수정 mixin
# =============================================================================
def _register_numpy_safe_globals():
    """nnUNet 체크포인트(.pth) 의 numpy 메타데이터를 weights_only=True 로 안전 로드하기 위한
    allowlist (임의코드 실행과 무관한 numpy 재구성 global 만)."""
    try:
        import numpy as _np
        import torch.serialization as _ts
        g = [_np.ndarray, _np.dtype, _np.core.multiarray.scalar, _np.core.multiarray._reconstruct]
        try:
            import numpy.dtypes as _nd
            g += [getattr(_nd, _n) for _n in dir(_nd) if _n.endswith("DType")]
        except Exception:
            pass
        _ts.add_safe_globals(g)
    except Exception:
        pass


class _PengwinPolyLR:
    """torch `_LRScheduler` 를 상속하지 않는 PolyLR — nnUNet PolyLRScheduler 와 수식 동일하나
    torch 의 step-순서/epoch-인자 deprecation 경고를 발생시키지 않는다.

    nnUNet 의 lr_scheduler contract 는 (a) step(current_epoch) 호출(on_train_epoch_start),
    (b) param_groups['lr'] 갱신, (c) checkpoint 에 저장 안 함(current_epoch 로 재계산) 뿐이라
    plain 객체로 충분하다."""

    def __init__(self, optimizer, initial_lr, max_steps, exponent=0.9):
        self.optimizer = optimizer
        self.initial_lr = float(initial_lr)
        self.max_steps = int(max_steps)
        self.exponent = float(exponent)
        self.ctr = 0
        self.step(0)

    def step(self, current_step=None):
        if current_step is None:
            current_step = self.ctr
            self.ctr += 1
        frac = max(0.0, 1.0 - current_step / max(1, self.max_steps))
        new_lr = self.initial_lr * (frac ** self.exponent)
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr


class _StunetCleanTrainerMixin:
    """STU-Net trainer 공용 mixin. nnUNet 2.5.2 의 deprecated API 사용으로 뜨는 경고를
    modern API 로 **근본 수정**한다(억제 아님). 모든 STU-Net trainer 가 이 mixin 을 가장 먼저
    상속하여 아래 오버라이드가 MRO 상 우선 적용된다.

    수정 대상:
      #4 torch.compile+batch1+deep-supervision → `_do_i_compile()=False` 로 compile 비활성.
         validation 시 DS 토글에 compile 이 재컴파일 안 되어 예측이 손상되던 문제까지 근본 해결.
      #3 torch.cuda.amp.GradScaler deprecated → `torch.amp.GradScaler('cuda')` 로 교체.
      #1/#2/#5 lr_scheduler.step(epoch)/step-order deprecation → `_LRScheduler` 비상속
         `_PengwinPolyLR` 로 교체(on_train_epoch_start 의 step 호출이 우리 객체로 감).
      #6 torch.load(weights_only=False) FutureWarning → weights_only=True(+numpy allowlist) 안전 로드.
    """

    def _do_i_compile(self):  # #4
        return False

    def initialize(self):
        super().initialize()  # #3 GradScaler 는 모듈 레벨 patch 가 생성 시점에 신 API 로 처리
        _maybe_apply_stunet_warmstart(self)  # DDP-safe warm-start (env PENGWIN_STUNET_PRETRAINED)
        # [V0.x][FIX:DDP_BADB][2026-06-04] V300 BADB 는 output=base+delta·refinement 를
        # delta 초기값 0 으로 시작한다(V291 byte-identical). delta=0 이면 refinement conv
        # 파라미터가 첫 iteration 에 grad 0 → DDP 의 unused-parameter 검사가 실패한다
        # ("parameters that were not used in producing loss"). delta 가 움직이기 시작하면
        # 이후 grad 가 흐르므로, find_unused_parameters=True 로 re-wrap 해 학습을 허용한다.
        # 같은 .module 을 재포장하므로 파라미터 텐서 동일성이 유지되어 optimizer 는 무영향.
        if getattr(self, "is_ddp", False):
            from torch.nn.parallel import DistributedDataParallel as _DDP
            if isinstance(self.network, _DDP):
                self.network = _DDP(
                    self.network.module,
                    device_ids=[self.local_rank],
                    find_unused_parameters=True,
                )

    def configure_optimizers(self):  # #1/#2/#5
        optimizer, _orig_sched = super().configure_optimizers()
        self.lr_scheduler = _PengwinPolyLR(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, self.lr_scheduler

    def load_checkpoint(self, filename_or_checkpoint):  # #6
        if isinstance(filename_or_checkpoint, str):
            _register_numpy_safe_globals()
            _orig_load = torch.load

            def _safe_load(f, *a, **k):
                k.setdefault("weights_only", True)
                return _orig_load(f, *a, **k)

            torch.load = _safe_load
            try:
                return super().load_checkpoint(filename_or_checkpoint)
            finally:
                torch.load = _orig_load
        return super().load_checkpoint(filename_or_checkpoint)


class PengwinTrainerSTUNetBaseAnatomyV301(_StunetCleanTrainerMixin, PengwinTrainer):
    """[V0.x][BACKBONE] Ds539 anatomy 5-class 용 STU-Net-B 백본 trainer.

    ResEnc-L → STU-Net-B(58.26M) 교체. TotalSegmentator(59개 뼈: sacrum/hip/femur 포함)
    사전학습 warm-start 전제. do_split(grouped split)·loss(DC+CE)·env profile 등
    PengwinTrainer 동작은 그대로 유지하고 네트워크만 STU-Net 으로 바꾼다.
    fine-tune 레시피: lr 1e-3 (STU-Net _ft 권장), epochs 1000.
    """

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=unpack_dataset, device=device)
        self.initial_lr = 1e-3
        self.num_epochs = 1000
        # [V0.x][ES][2026-06-01] warm-start 용 early-stop 재튜닝.
        # 상속 기본값(PengwinTrainer: MIN_EPOCHS=100, PATIENCE=50)은 from-scratch 기준이다.
        # STU-Net 뼈 사전학습 warm-start 는 수십 epoch 안에 수렴하므로, 수렴 후 낭비 학습을
        # 막도록 ES 를 낮춘다(MIN_EPOCHS=30 부터 EMA-dice 정체 PATIENCE=25 epoch 시 종료).
        # 상속된 while-loop run_training + on_validation_epoch_end 의 ES 로직이 그대로 적용되며,
        # checkpoint_best 는 상시 저장되므로 종료 시점 최적 가중치가 보존된다. env 로 튜닝 가능.
        import os as _os_es
        self.ES_MIN_EPOCHS = int(_os_es.environ.get("PENGWIN_ES_MIN_EPOCHS", "30"))
        self.ES_PATIENCE = int(_os_es.environ.get("PENGWIN_ES_PATIENCE", "25"))
        self.ES_MIN_DELTA = float(_os_es.environ.get("PENGWIN_ES_MIN_DELTA", "5e-3"))
        # [V0.x][PARTIAL-LABEL][2026-06-02] marginal loss용 case→labeled-class 맵 로드.
        self._marginal_loss = None
        self._case_labeled_map = {}
        self._load_case_labeled_map()

    def _load_case_labeled_map(self):
        """case_id → 라벨된 fg 클래스 리스트. pelvic 케이스={1,2,3}, femur 케이스={4} (disjoint).
        GT 존재 클래스로 사전 산출된 json(result/reports/ds539_case_labeled_classes.json) 로드."""
        import json as _json
        path = os.environ.get("PENGWIN_CASE_LABELED_MAP",
                              str(RESULT_REPORT / "ds539_case_labeled_classes.json"))
        try:
            raw = _json.load(open(path))
            self._case_labeled_map = {str(k): [int(x) for x in v] for k, v in raw.items()}
            self.print_to_log_file(
                f"[marginal] case→labeled 맵 로드: {len(self._case_labeled_map)} cases ({path})")
        except Exception as e:  # pragma: no cover
            self._case_labeled_map = {}
            self.print_to_log_file(f"[marginal] 맵 로드 실패({e}) → 전부 labeled 폴백(표준 동작)")

    def _set_marginal_mask(self, keys):
        """batch['keys'](case 식별자) → per-sample labeled_mask[B,C] 를 marginal loss 에 설정."""
        if self._marginal_loss is None or keys is None:
            return
        ncls = self.label_manager.num_segmentation_heads
        B = len(keys)
        mask = torch.zeros(B, ncls, dtype=torch.bool, device=self.device)
        mask[:, 0] = True  # bg 항상 labeled
        for b, key in enumerate(keys):
            cid = str(key).replace("PENGWIN_", "").split("_")[0].zfill(3)
            labeled = self._case_labeled_map.get(cid)
            if not labeled:
                mask[b, :] = True  # 미상 → 전부 labeled(안전 폴백)
            else:
                for c in labeled:
                    if 0 <= int(c) < ncls:
                        mask[b, int(c)] = True
        self._marginal_loss.labeled_mask = mask

    def _build_loss(self):
        """표준 DC+CE 대신 Marginal Dice+CE(부분 라벨 충돌 근본 해결). DS wrapper 는
        nnUNet 관례 그대로(가중치 1/2^i, 최저 scale 0, DDP+no-compile 시 1e-6)."""
        import numpy as _np
        ncls = self.label_manager.num_segmentation_heads
        marg = MarginalDiceCELoss(
            num_classes=ncls,
            batch_dice=self.configuration_manager.batch_dice,
            smooth=1e-5, weight_ce=1.0, weight_dice=1.0)
        self._marginal_loss = marg
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = _np.array([1 / (2 ** i) for i in range(len(scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            return DeepSupervisionWrapper(marg, weights)
        return marg

    def train_step(self, batch: dict) -> dict:
        self._set_marginal_mask(batch.get("keys"))
        return super().train_step(batch)

    def validation_step(self, batch: dict) -> dict:
        # marginal mask 설정(val loss 일관) + 미라벨 예측을 bg 로 접어 메트릭을 marginal-aware 로.
        self._set_marginal_mask(batch.get("keys"))
        data = batch["data"]
        target = batch["target"]
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]
        axes = [0] + list(range(2, output.ndim))
        output_seg = output.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        del output_seg
        # [MARGINAL] 미라벨 클래스 예측을 bg(0) 로 이동 → 그 클래스는 라벨된 케이스에서만 평가됨
        lm = getattr(self._marginal_loss, "labeled_mask", None)
        if lm is not None:
            lm = lm.to(predicted_segmentation_onehot.device)
            B, C = predicted_segmentation_onehot.shape[:2]
            for c in range(1, C):
                unl = ~lm[:, c]
                if bool(unl.any()):
                    sl = predicted_segmentation_onehot[:, c]
                    moved = sl * unl.view(B, *([1] * (sl.ndim - 1))).to(sl.dtype)
                    predicted_segmentation_onehot[:, 0] += moved
                    predicted_segmentation_onehot[:, c] -= moved
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=None)
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]
        return {"loss": l.detach().cpu().numpy(), "tp_hard": tp_hard,
                "fp_hard": fp_hard, "fn_hard": fn_hard}

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision: bool = True):
        return _build_stunet_from_plan("base", arch_init_kwargs, num_input_channels,
                                       num_output_channels,
                                       enable_deep_supervision=enable_deep_supervision)


class PengwinTrainerSTUNetBaseABBCPhase1V302(_StunetCleanTrainerMixin, PengwinTrainer):
    """[PHASE-1 CLEAN] Ds538 leak-free instance-label ABBC 4-class STU-Net-B trainer.

    Successor to the FAILED InstanceConnectivity V301. The loss-level topology penalty there was
    mis-scaled (~25x the base loss: raw conn ~8.5 vs base ~0.32 from boost=8) and collapsed training
    the instant it ramped in (epoch 9: train_loss 0.32→1.04, n_pred 3→10, recall/precision→0). The
    warmup only delayed the collapse. Worse, its premise was a measurement artifact: the per-epoch
    "recall problem" came from a core-ONLY proxy decode (no regrow), while the REAL submission decoder
    (watershed regrow) already yields held-out instance-F1 ~0.85. So #1 (topology-in-loss) is retired.

    This trainer trains the plain boundary-weighted ABBC head and lets the real decoder make instances:
    - Backbone STU-Net-B. Head 4ch ABBC [bg,border,boundary,core], FORCED to 4 (plan num_classes=24).
    - Input CT-only (1ch). Target = the nnUNet seg-label IS the per-anatomy fragment instance map
      (relabeled 1..K; -1=ignore, 0=bg) — NO sidecar. The loss builds the ABBC target on the fly.
    - Loss V291 boundary-weighted ABBC (boundary class CE+Dice ×5; boundary is only ~5% of support).
    - Validation/ES: the REAL submission decoder decode_task1_v288_abbc (core-seed watershed REGROW +
      small-CC merge), NOT the old core-only proxy — so the per-epoch instance metrics and the
      F1-driven ES match the eval/leaderboard decode. pseudo-Dice DISABLED (head=4 != num_classes=24).
    - Grouped 5-fold split + ES-enforce run_training + warm-start hook inherited (PENGWIN_STUNET_PRETRAINED
      = 97pt anatomy ckpt, 1ch→1ch, head-skip). See [[pengwin-instance-label-nosidecar]].
    """

    NUM_EPOCHS_DEFAULT = 1000
    ES_PATIENCE = 25
    ES_MIN_DELTA = 5e-3
    ES_MIN_EPOCHS = 30
    BG, BORDER, BOUNDARY, CORE = 0, 1, 2, 3

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset=unpack_dataset, device=device)
        self.initial_lr = float(os.environ.get("PENGWIN_INITIAL_LR", "1e-3"))
        self.ES_MIN_EPOCHS = int(os.environ.get("PENGWIN_ES_MIN_EPOCHS", str(self.ES_MIN_EPOCHS)))
        self.ES_PATIENCE = int(os.environ.get("PENGWIN_ES_PATIENCE", str(self.ES_PATIENCE)))
        self.ES_MIN_DELTA = float(os.environ.get("PENGWIN_ES_MIN_DELTA", str(self.ES_MIN_DELTA)))
        self.print_to_log_file(
            "[ABBCPhase1V302] backbone=STUNetB out=4 loss=LeakFreeInstanceABBCLoss(boundary-weighted) "
            f"decode=real-watershed-regrow ds=off num_epochs={self.num_epochs} initial_lr={self.initial_lr} "
            f"ES(min={self.ES_MIN_EPOCHS},pat={self.ES_PATIENCE},delta={self.ES_MIN_DELTA}) "
            "warmstart_env=PENGWIN_STUNET_PRETRAINED"
        )

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision: bool = True):
        # plan declares ~24 fragment labels -> num_output_channels arrives as 24; the ABBC head is 4.
        return _build_stunet_from_plan(
            "base", arch_init_kwargs, num_input_channels,
            num_output_channels=4, enable_deep_supervision=False,
        )

    def _build_loss(self):
        # Leak-free boundary-weighted ABBC: reads the nnUNet seg-label directly as the per-anatomy
        # fragment instance map (1..K; -1 ignore, 0 bg), builds the ABBC class target on the fly,
        # CE+Dice over [bg,border,boundary,core] with boundary class up-weighted ×5 and the nnUNet
        # ignore region masked out. NO connectivity term (retired #1 collapsed training).
        return LeakFreeInstanceABBCLoss()

    @staticmethod
    def _decode_instances(logits: torch.Tensor) -> np.ndarray:
        """4ch ABBC logits [B,4,Z,Y,X] -> instance map [B,Z,Y,X] via the REAL submission decoder:
        core(ch3)>=thr CC seeds -> skimage watershed REGROW over support(bg<thr) -> small-CC merge.
        This is decode_task1_v288_abbc (eval/leaderboard decode) — NOT the old core-only proxy that
        left fragments eroded and tanked the per-epoch recall. Lazy import avoids the eval<->core cycle."""
        from eval import decode_task1_v288_abbc, task1_v288_probabilities_from_logits
        min_vox = int(os.environ.get("PENGWIN_MIN_CC_VOX", "100"))
        bg_thr = float(os.environ.get("PENGWIN_BG_THRESH", "0.5"))
        core_thr = float(os.environ.get("PENGWIN_CORE_THRESH", "0.5"))
        arr = logits.detach().float().cpu().numpy()
        out = np.zeros((arr.shape[0],) + arr.shape[2:], dtype=np.int32)
        for b in range(arr.shape[0]):
            probs = task1_v288_probabilities_from_logits(arr[b])  # [4,Z,Y,X] softmax
            decoded, _ = decode_task1_v288_abbc(
                probs, background_threshold=bg_thr, core_threshold=core_thr,
                min_component_voxels=min_vox)
            out[b] = decoded.astype(np.int32)
        return out

    @staticmethod
    def _val_abbc_logits(logits):
        """Hook: which channels feed the per-epoch instance decode + the [B,4,...] shape check.
        The ABBC head is 4ch; affinity subclasses override to slice the ABBC channels [:, :4]."""
        return logits

    def validation_step(self, batch: dict) -> dict:
        from utils import instance_iouf
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            del data
            loss = self.loss(output, target)
        logits = output[0] if isinstance(output, (list, tuple)) else output
        logits = self._val_abbc_logits(logits)   # affinity trainers slice the ABBC channels [:, :4]
        if logits.ndim != 5 or int(logits.shape[1]) != 4:
            raise ValueError(f"Phase-1 ABBC expects val logits [B,4,Z,Y,X], got {tuple(logits.shape)}")
        gt = target[0] if isinstance(target, list) else target
        if gt.ndim == 5 and int(gt.shape[1]) == 1:
            gt = gt[:, 0]
        gt = gt.long().clamp_min(0).cpu().numpy()
        pred_inst = self._decode_instances(logits)
        iou, rec, prec, npred, ngt = [], [], [], [], []
        for b in range(gt.shape[0]):
            try:
                r = instance_iouf(pred_inst[b], gt[b])
                ng = max(int(r["n_gt_fragments"]), 0)
                npd = max(int(r["n_pred_fragments"]), 0)
                rows = r.get("per_gt", []) or []
                # match at IoU>=0.5 (standard instance match)
                matched = [row for row in rows if float(row.get("best_iou", 0.0)) >= 0.5]
                matched_pred = len({row["best_pred_id"] for row in matched if row.get("best_pred_id") is not None})
                iou.append(float(r["iou_f_mean"]))
                rec.append(len(matched) / ng if ng else 0.0)          # recall = matched GT / all GT (under-seg)
                prec.append(matched_pred / npd if npd else 0.0)       # precision = matched pred / all pred (over-seg)
                npred.append(float(npd)); ngt.append(float(ng))
            except Exception:
                iou.append(0.0); rec.append(0.0); prec.append(0.0); npred.append(0.0); ngt.append(0.0)
        a = lambda v: np.asarray([float(np.mean(v)) if v else 0.0], dtype=np.float64)
        return {"loss": loss.detach().cpu().numpy(), "iouf": a(iou), "recall": a(rec),
                "precision": a(prec), "npred": a(npred), "ngt": a(ngt)}

    def on_validation_epoch_end(self, val_outputs):
        outputs = collate_outputs(val_outputs)
        loss_here = float(np.mean(outputs["loss"]))
        score = float(np.mean(outputs["iouf"]))      # ES driver = instance IoU-F
        recall = float(np.mean(outputs["recall"])) if "recall" in outputs else 0.0
        precision = float(np.mean(outputs["precision"])) if "precision" in outputs else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        npred = float(np.mean(outputs["npred"])) if "npred" in outputs else 0.0
        ngt = float(np.mean(outputs["ngt"])) if "ngt" in outputs else 0.0
        # ES + checkpoint_best now driven by F1 (balances recall vs precision -> SEES over-seg),
        # not IoU-F alone which is count-blind. IoU-F still shown in the [instance] line below.
        self.logger.log("mean_fg_dice", f1, self.current_epoch)
        self.logger.log("dice_per_class_or_region", [score, recall, precision, f1], self.current_epoch)
        self.logger.log("val_losses", loss_here, self.current_epoch)
        # multi-angle instance metrics (recall=under-seg, precision=over-seg, n_pred/n_gt=merge signal)
        self.print_to_log_file(
            f"[instance] IoU-F={score:.4f} recall={recall:.4f} precision={precision:.4f} "
            f"F1={f1:.4f} | n_pred/n_gt={npred:.1f}/{ngt:.1f}"
        )
        completed = self.current_epoch + 1
        if completed >= self.ES_MIN_EPOCHS:
            ema = float(self.logger.my_fantastic_logging["ema_fg_dice"][-1])
            if ema > self._es_best_dice + self.ES_MIN_DELTA:
                self._es_best_dice = ema
                self._es_no_improve = 0
            else:
                self._es_no_improve += 1
                if self._es_no_improve >= self.ES_PATIENCE and not self._es_triggered:
                    self._es_triggered = True
                    self.print_to_log_file(
                        f"Early stopping: no instance F1 improvement for {self.ES_PATIENCE} epochs. "
                        f"Best EMA F1={self._es_best_dice:.4f}. Stopping at epoch {completed}."
                    )
                    self.num_epochs = completed

    def perform_actual_validation(self, save_probabilities: bool = False):
        # nnUNet's end-of-training sliding-window validation allocates a num_classes(=24)-channel logits
        # buffer (predicted_logits[sl] += prediction) but our ABBC head outputs 4 -> 24-vs-4 mismatch.
        # The per-epoch instance_iouf already drives ES + checkpoint_best selection; the proper full
        # instance evaluation is run separately via eval.py / experiments tooling (Plan Phase 1.5).
        self.print_to_log_file(
            "[Phase1] perform_actual_validation SKIPPED (head=4 != plan num_classes=24; "
            "ES/checkpoint driven by per-epoch instance_iouf; full instance eval via eval.py)."
        )


class PengwinTrainerSTUNetBaseAffinityV308(PengwinTrainerSTUNetBaseABBCPhase1V302):
    """[TIER-1] V302 + an AFFINITY head decoded by AVERAGE-LINKAGE agglomeration, to break the
    touching-fragment MERGE ceiling (recall ~0.71 = ~40% of GT fragments merged at decode).

    Deep-research (GASP CVPR'22): mutex-watershed is GASP-AbsMax = least noise-robust (= why V303 over-split);
    average-linkage agglomeration on a LEARNED affinity is the fix. (Loss-level X-CAC = within-noise; decode
    tweaks fuzzy = over-split; V307's UNBALANCED affinity BCE collapsed to affinity~1 everywhere because ~95%
    of pairs are same-instance — all RETIRED, see docs/Experiments.md + [[pengwin-affinity-agglo-direction]].)

    IDENTICAL to V302 EXCEPT: (1) head 4 -> 4+K channels (4 ABBC mask/Dice + K affinity offsets);
    (2) loss = LeakFreeInstanceABBCAffinityLoss = ABBC + CLASS-BALANCED per-offset same-instance BCE
    (0.5*(L_same+L_diff)) so the rare cross-fragment fracture edges aren't drowned; (3) the per-epoch
    instance proxy decodes only the ABBC channels [:4] via _val_abbc_logits (fast) — the affinity
    average-linkage decode (PENGWIN_AFFINITY_DECODE) is run OFFLINE. Warm-start: 97pt base, head reinit
    on shape mismatch. K = len(loss.AFFINITY_HEAD_OFFSETS).
    """

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision: bool = True):
        from loss import AFFINITY_HEAD_OFFSETS
        return _build_stunet_from_plan(
            "base", arch_init_kwargs, num_input_channels,
            num_output_channels=4 + len(AFFINITY_HEAD_OFFSETS), enable_deep_supervision=False,
        )

    def _build_loss(self):
        from loss import LeakFreeInstanceABBCAffinityLoss
        return LeakFreeInstanceABBCAffinityLoss()

    @staticmethod
    def _val_abbc_logits(logits):
        # per-epoch instance proxy + shape-check use the ABBC channels only; affinity decode is offline.
        return logits[:, :4]



#!/usr/bin/env python3
"""PENGWIN 2026 Task 1 — Training orchestrator.

Subcommands:
    train             Train an active dataset (wraps nnUNetv2_train CLI)
    queue             Run the planned fold training queue
    status            Summarize training logs and early-stop ETA
    stunet-finetune   STU-Net warm-start fine-tune launcher (absorbed run_finetuning_stunet.py;
                      forwards remaining args verbatim to nnU-Net, monkeypatching the pretrained loader)

Active datasets: 539 (PelvicFemurAnatomyV3 anatomy) + 538 (PelvicFemurBICMFragmentV5 fracture).
532/533/537 retired.

Usage:
    python train.py train 539 --gpu 0                                 # anatomy
    python train.py stunet-finetune 538 3d_fullres 0 -tr <ABBC trainer> -num_gpus 2   # fracture DDP

Hard-mining/self-training orchestration was removed from the active workflow.
"""
from __future__ import annotations
import argparse, ast, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from core import (
    DATASETS, NN_RES, RESULT_DATE, RESULT_REPORT, RESULT_WEIGHT,
    configure_nnunet_env, get_logger,
)
configure_nnunet_env()
log = get_logger(__name__)


def _nnunet_train_bin() -> str:
    """Resolve nnU-Net training CLI in the current runtime environment."""
    explicit = os.environ.get("PENGWIN_NNUNET_TRAIN_BIN", "").strip()
    if explicit:
        return explicit
    found = shutil.which("nnUNetv2_train")
    if found:
        return found
    raise FileNotFoundError(
        "nnUNetv2_train not found. Install nnunetv2 or set PENGWIN_NNUNET_TRAIN_BIN."
    )


def _write_json(path: Path, payload) -> None:
    """Write indented JSON and create the parent directory when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def run_stunet_finetune(argv=None):
    """STU-Net warm-start fine-tune 런처 (formerly run_finetuning_stunet.py).

    목적
    ----
    nnU-Net 2.5.2 의 표준 `nnUNetv2_train` 은 pretrained 적재 시
    `nnunetv2.run.run_training.load_pretrained_weights` 를 호출하는데, 이 기본 로더는
    (1) STUNet 의 seg head 네이밍(`seg_outputs.*`)을 skip 하지 못하고(`.seg_layers.` 만 skip),
    (2) 입력 채널 inflation 을 처리하지 못해 STUNet warm-start 에서 깨진다.

    본 런처는 STU-Net 공식 run_finetuning_stunet.py 와 동일한 monkey-patch 방식으로,
    표준 nnU-Net 학습 진입점(run_training_entry)을 그대로 쓰되 pretrained 로더만 우리
    `model.load_stunet_pretrained_weights`(prefix 처리 + stem inflate + 안전 역직렬화)로
    교체한다. 설치된 nnU-Net 파일은 건드리지 않는다.

    사용 예 (인자는 전부 nnUNetv2_train / run_training_entry 와 동일하게 전달된다)
    -------
        PYTHONPATH=/workspace/code_task1 \
        nnUNet_raw=.../result/raw nnUNet_preprocessed=.../result/preprocessed \
        nnUNet_results=.../result/results \
        python -m train stunet-finetune 539 3d_fullres 0 \
            -tr PengwinTrainerSTUNetBaseAnatomyV301 \
            -p nnUNetResEncUNetLPlans \
            -pretrained_weights result/weights/stunet_base_TotalSeg.pth \
            --npz [-num_gpus 2]
    """
    from unittest.mock import patch
    import nnunetv2.run.run_training as _rt
    from model import load_stunet_pretrained_weights

    if argv is not None:
        sys.argv = [sys.argv[0]] + list(argv)
    # nnU-Net 의 run_training 네임스페이스에 import 된 load_pretrained_weights 를
    # 우리 STUNet 로더로 교체. 시그니처(network, fname, verbose) 호환.
    with patch.object(_rt, "load_pretrained_weights", load_stunet_pretrained_weights):
        _rt.run_training_entry()


PRETRAINED_DEFAULT = str(RESULT_WEIGHT / "pelvic_s1_swa.pth")
DEFAULT_FOLD0_ORDER = [538, 539]

# Speed best practices for nnUNet training (reduces bottleneck on dual-GPU).
# Applied automatically in train(); can be overridden via extra_env.
SPEED_ENV = {
    "nnUNet_compile": "t",                  # torch.compile (~10-20% speedup on TITAN RTX)
    "nnUNet_n_proc_DA": "16",               # data augmentation workers (default 12)
    "OMP_NUM_THREADS": "1",                 # prevent CPU oversubscription (per worker)
    "MKL_NUM_THREADS": "1",                 # same for MKL
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",  # better memory fragmentation
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",   # cuBLAS workspace (deterministic + faster)
}

# Stable profile trades throughput for reliability by disabling torch.compile
# and lowering augmentation workers. It is still useful for v2 fold resumes
# on memory-constrained GPUs.
STABLE_ENV = {
    "nnUNet_compile": "f",
    "nnUNet_n_proc_DA": "8",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}

# Single-worker profile for tiny overfit/debug runs. nnU-Net unpacks .npz
# files into .npy sidecars before the training dataloader starts. On very
# small rebuilt datasets, multi-process augmentation can race with a stale or
# partial unpacked file and surface as numpy memmap length errors. Keep this
# profile explicit so V5/V5.1 one-case gates can be reproduced without changing
# the normal `speed`/`stable` production defaults.
SINGLE_ENV = {
    "nnUNet_compile": "f",
    "nnUNet_n_proc_DA": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}

ENV_PROFILES = {
    "speed": SPEED_ENV,
    "stable": STABLE_ENV,
    "single": SINGLE_ENV,
}

LOSS_PROFILES = [
    "dc_ce",
    "bd_dou_005",
    "bd_dou_01",
    "bd_dou_03",
    "tversky_07",
    "tversky_08",
    "combo_tversky_bd005",
    "bicm_v5_sparse",
    "bicm_v5_support_geometry",
    "bicm_v62_semantic_boundary",
    "bicm_v64_topology_precision",
    "bicm_v65_balanced_contact_core",
    "bicm_v67_semantic_edge_pair",
    "bicm_v6_factorized",
    "bicm_v250_oracle_aligned_direct",
    "bicm_v6_precision",
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
    "bicm_v160_core_only_center_seed_offset_sampler",
    "bicm_v161_core_heatmap_center_seed_offset_sampler",
    "boundary_fragment_v3",
    "boundary_fragment_v3_core_ridge",
    "boundary_fragment_v3_core_ridge_recall",
    "boundary_fragment_v3_core_ridge_recall_masscap",
    "boundary_fragment_v3_core_ridge_recall_pos_masscap",
    "boundary_fragment_v3_core_ridge_recall_soft_pos_masscap",
    "boundary_fragment_v3_core_ridge_recall_very_soft_pos_masscap",
    "boundary_fragment_v3_core_ridge_recall_logit_calibrated",
    "boundary_fragment_v3_core_ridge_recall_precision_logit_calibrated",
    "boundary_fragment_v3_core_ridge_recall_shell_contrast_logit_calibrated",
    "boundary_fragment_v3_core_ridge_recall_strong_shell_contrast_logit_calibrated",
    "boundary_fragment_v3_core_ridge_recall_binary_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_precision_binary_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_positive_floor_binary_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_positive_floor_masscap_binary_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_candidate_binary_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_candidate_masscap_binary_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_candidate_logit_ridge_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_candidate_logit_ridge_precision_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_candidate_soft_ridge_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_candidate_soft_ridge_contrast_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_binary_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_barrier_contrast_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_seed_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_sparse_seed_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core_compact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core_compact_seed_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core_compact_sparse_seed_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core_recall_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core_coverage_open_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core_coverage_guarded_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_balanced_contact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_fragment_peak_core_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_weak_peak_compact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_strong_peak_compact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_strong_peak_balanced_contact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_strong_peak_tight_balanced_contact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_strong_peak_soft_contact_ridge_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_strong_peak_precision_soft_contact_ridge_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_strong_peak_contact_shell_contrast_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_strong_peak_strong_contact_shell_contrast_barrier_head",
    "boundary_fragment_v3_core025_strong_peak_decoder_contact_head",
    "boundary_fragment_v3_core025_strong_peak_xyz_affinity_head",
    "boundary_fragment_v3_core025_strong_peak_affinity13_seed_healed_head",
    "boundary_fragment_v3_core025_strong_peak_affinity13_contact_hard_negative_seed_healed_head",
    "boundary_fragment_v3_core025_strong_peak_mutex13_seed_healed_head",
    "boundary_fragment_v3_core025_strong_peak_mutex13_support_leak_guard_seed_healed_head",
    "boundary_fragment_v3_core025_strong_peak_mutex13_seed_stable_support_leak_guard_seed_healed_head",
    "boundary_fragment_v3_core025_strong_peak_mutex13_seed_peak_support_leak_guard_seed_healed_head",
    "boundary_fragment_v3_core025_strong_peak_mutex13_seed_peak_body_support_leak_guard_seed_healed_head",
    "boundary_fragment_v3_core025_strong_peak_mutex13_fracture_seed_calibrated_support_leak_guard_seed_healed_head",
    "boundary_fragment_v3_core025_strong_peak_pairwise_softmax_mutex13_seed_healed_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_pairwise_softmax_mutex13_seed_healed_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_pairwise_softmax_seed_healed_v273_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_pairwise_softmax_seed_calibrated_v274_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_global_coord_pairwise_v284_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_fragment_position_v275_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_fragment_position_dice_v276_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_separator_gap_v277_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_separator_energy_v278_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_separator_softmax_v287_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_abbc_v288_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_abbc_sdf_v289_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_abbc_sdf_fdm_v290_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_abbc_v291_head",
    "boundary_fragment_v3_core025_strong_peak_center_flow_head",
    "boundary_fragment_v3_core025_strong_peak_center_peak_flow_head",
    "boundary_fragment_v3_core025_strong_peak_center_peak_flow_calibrated_head",
    "boundary_fragment_v3_core025_strong_peak_dense_center_heatmap_flow_calibrated_head",
    "boundary_fragment_v3_core025_strong_peak_topology_constrained_center_heatmap_flow_calibrated_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_center_heatmap_flow_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_spatial_embedding_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_spatial_embedding_contrastive_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_coord_spatial_embedding_v279_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_query_mask_v280_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_query_mask_pn_v281_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_global_coord_query_mask_v285_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_global_coord_query_mask_pn_v286_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_free_embedding_v282_head",
    "boundary_fragment_v3_core025_strong_peak_no_contact_global_coord_free_embedding_v283_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_mid_peak_compact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_contact_preserve_peak_compact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_mid_peak_coverage_compact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_original_contact_mid_peak_compact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_mid_peak_contact_cap_compact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_mid_peak_tight_contact_cap_compact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_mid_peak_support_seed_compact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_core025_mid_peak_conservative_coretail_compact_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_weak_fragment_mass_core_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_loose_fragment_mass_core_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_instance_core_guard_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_instance_core_strong_guard_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_masscap_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_dense_candidate_weak_masscap_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_soft_dense_candidate_barrier_head",
    "boundary_fragment_v3_core_ridge_recall_soft_dense_candidate_barrier_contrast_head",
    "boundary_fragment_v3_core_ridge_recall_masscap_tight",
    "boundary_fragment_v3_core_seed",
    "boundary_fragment_v3_ridgefit_precision",
    "factorized_instance_v4_embedding_v163",
    "bicm_v6_support_core",
    "bicm_v6_support_precision",
    "bicm_v6_support_surface",
]

CE_WEIGHT_PROFILES = ["off", "auto"]
OVERSAMPLE_PROFILES = [
    "default",
    "weak0123",
    "bicm_v5_sparse",
    "bicm_v6_support_mixed",
    "bicm_v7_contact_mixed",
    "bicm_v7_contact15_mixed",
    "bicm_v11_contact_roi",
    "bicm_v18_roi_balanced",
    "boundary_fragment_v3_v5_remap",
    "boundary_fragment_v3_sidecar",
    "boundary_fragment_v3_sidecar_core_recall_high_edge",
    "boundary_fragment_v3_sidecar_core_ridge_high_edge",
    "boundary_fragment_v3_sidecar_core_ridge",
    "boundary_fragment_v3_sidecar_core_seed",
    "boundary_fragment_v3_sidecar_precision",
    "factorized_instance_v4_v163",
]
# [cleanup 2026-06-07] TRAINERS is derived from the actual PengwinTrainer* classes in `core`
# (the single source of truth). The old hand-maintained list had drifted to 258 entries —
# ~235 phantom (no such class) AND missing the active STU-Net trainers — so `--trainer
# <active>` failed argparse validation and the list misled anyone reading it. dir(core)
# is always correct.
import core as _core_mod
TRAINERS = sorted(n for n in dir(_core_mod) if n.startswith("PengwinTrainer"))


def _resolve_training_profiles(trainer: str,
                               loss_profile: str | None,
                               ce_class_weights: str | None,
                               oversample_profile: str | None) -> tuple[str, str, str]:
    """Apply trainer-specific defaults without changing explicit user choices."""
    # The active STU-Net trainers resolve their loss profile inside the
    # trainer class (core._build_loss); the live launcher is the
    # `stunet-finetune` subcommand / run_stunet_finetune() in this module
    # 과거 trainer별 252개 분기는 제거했다. 명시한 사용자 값은 그대로 전달하고,
    # 값이 없을 때만 아래 공통 기본값을 쓴다.
    return (
        loss_profile or "dc_ce",
        ce_class_weights or "off",
        oversample_profile or "default",
    )


# =============================================================================
# train — wraps nnUNetv2_train CLI
# =============================================================================
def train(ds_id: int, fold: int = 0, gpu: int = 0,
          plans: str = "nnUNetResEncUNetLPlans",
          config: str = "3d_fullres",
          init_weights: str | None = None,
          full_init_network: bool = False,
          trainer: str | None = None,
          loss_profile: str | None = None,
          ce_class_weights: str | None = None,
          oversample_profile: str | None = None,
          env_profile: str = "speed",
          continue_training: bool = False,
          tag: str | None = None,
          extra_env: dict | None = None,
          background: bool = False) -> int:
    """Launch nnUNetv2_train.

    - Foreground is the default so this process waits and reaps nnU-Net cleanly.
      That matters in this container because PID 1 is `sleep`, not an init
      reaper, so detached orphan training processes can remain as zombies after
      completion.
    - SPEED_ENV (torch.compile, n_proc_DA=16, expandable_segments, OMP=1) is
      applied unless overridden via `extra_env`.
    - GPU selection by setting CUDA_VISIBLE_DEVICES (rather than
      `--device cuda:N`) so nnUNet sees device as cuda:0 internally.
    - `--background` keeps the old detached behavior for manual use, but the
      foreground mode is preferred for long runs that should tail cleanly.

    Args:
        ds_id: active dataset ID in the current registry (532 or 533).
        fold: 0..4 for 5-fold cross-val.
        gpu: CUDA index (0 or 1).
        plans: nnUNet plans file. Default ResEncL = SOTA backbone.
        config: 3d_fullres | 3d_lowres | 2d. Default 3d_fullres for full quality.
        init_weights: pretrained .pth path; None = train from scratch.
        full_init_network: include segmentation heads in pretrained transfer.
            Required for same-architecture refinement runs such as V94.
        extra_env: extra environment vars to override SPEED_ENV defaults.
        trainer: active custom nnU-Net trainer.
        loss_profile: explicit weak-class/boundary ablation from LOSS_PROFILES.
            None lets the selected trainer choose its default.
        ce_class_weights: off | auto. Auto computes clipped median-frequency
            CE weights from rebuilt labels and writes an audit JSON.
        oversample_profile: default | weak0123. None follows selected trainer.
        env_profile: speed | stable | single. Stable disables torch.compile and
            lowers data augmentation workers for memory-constrained resumes.
            Single disables augmentation multiprocessing for tiny overfit/debug
            runs where unpacked sidecars must be read deterministically.
        continue_training: pass nnU-Net `--c` and resume optimizer/EMA from
            checkpoint_latest.pth. Use this only when latest exists.

    Returns: process PID.
    """
    if env_profile not in ENV_PROFILES:
        raise ValueError(f"env_profile must be one of {sorted(ENV_PROFILES)}")
    cfg = DATASETS[ds_id]
    trainer = trainer or cfg["trainer"]
    if trainer not in TRAINERS:
        raise ValueError(f"trainer must be one of {TRAINERS}")
    loss_profile, ce_class_weights, oversample_profile = _resolve_training_profiles(
        trainer, loss_profile, ce_class_weights, oversample_profile
    )
    if loss_profile not in LOSS_PROFILES:
        raise ValueError(f"loss_profile must be one of {LOSS_PROFILES}")
    if ce_class_weights not in CE_WEIGHT_PROFILES:
        raise ValueError(f"ce_class_weights must be one of {CE_WEIGHT_PROFILES}")
    if oversample_profile not in OVERSAMPLE_PROFILES:
        raise ValueError(f"oversample_profile must be one of {OVERSAMPLE_PROFILES}")
    cmd = [
        _nnunet_train_bin(),
        str(ds_id), config, str(fold),
        "-tr", trainer, "-p", plans,
    ]
    if continue_training:
        latest = NN_RES / cfg["name"] / f"{trainer}__{plans}__{config}" / f"fold_{fold}" / "checkpoint_latest.pth"
        if not latest.exists():
            raise FileNotFoundError(
                f"--continue-training requires checkpoint_latest.pth, missing: {latest}"
            )
        cmd.append("--c")
        init_weights = None
    if init_weights:
        cmd += ["-pretrained_weights", str(init_weights)]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # [REPRO][Scope:runtime_env]
    # Training profile defaults are explicit and auditable. `speed` is used for
    # normal full training, `stable` for memory-constrained resumes, and
    # `single` for one-case gates where deterministic sidecar reads matter more
    # than throughput. The selected env vars are written into launch JSON below.
    for k, v in ENV_PROFILES[env_profile].items():
        env[k] = v
    if continue_training:
        # PyTorch 2.6 changed the default torch.load behavior to
        # weights_only=True when the caller does not pass an explicit value.
        # nnU-Net resume checkpoints include optimizer/training metadata, so
        # a plain `--c` resume can fail before writing a traceback to the
        # nnU-Net log. Keep this scoped to resume paths; fresh warm-starts use
        # a tensor-only checkpoint generated by _ensure_weights_only_pretrained.
        env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    env["PENGWIN_LOSS_PROFILE"] = loss_profile
    env["PENGWIN_CE_CLASS_WEIGHTS"] = ce_class_weights
    env["PENGWIN_OVERSAMPLE_PROFILE"] = oversample_profile
    if init_weights and trainer in {
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
        "PengwinTrainerBICMJointSupportProductV111",
        "PengwinTrainerBICMEdgeGraphAssignmentV112",
        "PengwinTrainerBICMAdaptiveBoundaryProductV113",
        "PengwinTrainerBICMAdaptiveBoundaryProductV113HighLR",
        "PengwinTrainerBICMEncoderAdapterV114",
        "PengwinTrainerBICMSemanticOracleAdapterV115",
        "PengwinTrainerBICMGraphCostSeparatorV116",
        "PengwinTrainerBICMSupportGateSemanticContactV117",
        "PengwinTrainerBICMWarmSupportGateSemanticContactV118",
        "PengwinTrainerBICMAllNetworkAdaptiveBoundaryV119",
        "PengwinTrainerBICMSameFragmentAffinityV120",
        "PengwinTrainerBICMWarmSameFragmentAffinityV121",
        "PengwinTrainerBICMWarmEdgeProductPrecisionV122",
        "PengwinTrainerBICMHighRecallGateCleanupV123",
        "PengwinTrainerBICMSupportConditionedEdgePrecisionV124",
        "PengwinTrainerBICMSaturatedEdgeDenseGateV125",
        "PengwinTrainerBICMLocalAdjacencyProductV126",
        "PengwinTrainerBICMLocalAdjacencyProductV126FromV96",
        "PengwinTrainerBICMEdgeResetLocalAdjacencyProductV127",
        "PengwinTrainerBICMSupportBridgeSuppressionV128",
        "PengwinTrainerBICMAffinitySharpeningV129",
        "PengwinTrainerBICMSupportTopologyRepairV130",
        "PengwinTrainerBICMAllNetworkSupportTopologyRepairV131",
        "PengwinTrainerBICMSupportVetoGateV132",
        "PengwinTrainerBICMSupportVetoSemanticV133",
        "PengwinTrainerBICMSupportVetoGateHighLRV134",
        "PengwinTrainerBICMSupportVetoGateUltraLRV135",
        "PengwinTrainerBICMSupportTopologyHighLRV136",
        "PengwinTrainerBICMSupportTopologyUltraLRV137",
        "PengwinTrainerBICMSupportTopologyMidLRV138",
        "PengwinTrainerBICMSupportTopologyUpperMidLRV139",
        "PengwinTrainerBICMSupportTopologyLowBracketLRV140",
        "PengwinTrainerBICMSupportTopologyHighBracketLRV141",
        "PengwinTrainerBICMSupportTopologyCoreSeedLowLRV142",
        "PengwinTrainerBICMSupportTopologyCoreSeedMidLRV143",
        "PengwinTrainerBICMStrongCoreSeedLowLRV144",
        "PengwinTrainerBICMStrongCoreSeedMidLRV145",
        "PengwinTrainerBICMFragmentSeedPresenceLowLRV146",
        "PengwinTrainerBICMFragmentSeedPresenceMidLRV147",
        "PengwinTrainerBICMCoreHeatmapSeedLowLRV148",
        "PengwinTrainerBICMCoreHeatmapSeedMidLRV149",
        "PengwinTrainerBICMCoreOnlyHeatmapSeedMidLRV150",
        "PengwinTrainerBICMCoreOnlyHeatmapSeedHighLRV151",
        "PengwinTrainerBICMCoreOnlyCenterSeedMidLRV152",
        "PengwinTrainerBICMCoreHeatmapCenterSeedMidLRV153",
        "PengwinTrainerBICMCoreOnlyCenterSeedSamplerV154",
        "PengwinTrainerBICMCoreHeatmapCenterSeedSamplerV155",
        "PengwinTrainerBICMCoreOnlyCenterSeedAffinitySamplerV158",
        "PengwinTrainerBICMCoreHeatmapCenterSeedAffinitySamplerV159",
        "PengwinTrainerBICMCoreOnlyCenterSeedOffsetSamplerV160",
        "PengwinTrainerBICMCoreHeatmapCenterSeedOffsetSamplerV161",
    }:
        # [AUDIT][Risk:High][Scope:warm_start_head_transfer]
        # nnU-Net's generic pretrained loader skips `.seg_layers.` by default.
        # V94+ are same-architecture V89 checkpoint refinements, so skipping
        # the final 10-channel heads leaves calibrated rows random and
        # invalidates any row-preservation claim. Force full transfer here.
        full_init_network = True
    if full_init_network:
        env["PENGWIN_LOAD_PRETRAINED_SEG_LAYERS"] = "1"
    if extra_env:
        env.update(extra_env)

    print(f"[train] Ds{ds_id} {cfg['name']} (fold {fold}) on GPU {gpu}")
    print(
        f"        trainer={trainer} loss_profile={env['PENGWIN_LOSS_PROFILE']} "
        f"ce_weights={env['PENGWIN_CE_CLASS_WEIGHTS']} "
        f"oversample={env['PENGWIN_OVERSAMPLE_PROFILE']} env_profile={env_profile}"
    )
    mode = "background" if background else "foreground"
    print(f"        mode = {mode}")
    if background:
        print("        logs: disabled (status/checkpoints only)")
        print("        WARN: background mode can leave zombies if PID 1 does not reap children")
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    else:
        print("        stdout/stderr: inherited; nnU-Net log also stays in the fold directory")
        proc = subprocess.Popen(cmd, env=env)
    print(f"        PID = {proc.pid}")
    launch_tag = tag or f"ds{ds_id}_fold{fold}_{int(time.time())}"
    status_path = RESULT_REPORT / f"train_launch_{launch_tag}.json"
    launch_payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": ds_id,
        "dataset_name": cfg["name"],
        "fold": fold,
        "gpu": gpu,
        "pid": proc.pid,
        "plans": plans,
        "config": config,
        "trainer": trainer,
        "continue_training": bool(continue_training),
        "init_weights": init_weights,
        "full_init_network": bool(full_init_network),
        "loss_profile": env["PENGWIN_LOSS_PROFILE"],
        "ce_class_weights": env["PENGWIN_CE_CLASS_WEIGHTS"],
        "oversample_profile": env["PENGWIN_OVERSAMPLE_PROFILE"],
        "env_profile": env_profile,
        "env_overrides": {k: env.get(k) for k in ENV_PROFILES[env_profile]},
        "mode": mode,
        "stdout_stderr": "DEVNULL" if background else "inherit",
        "logs": "nnU-Net fold training_log_*.txt",
    }
    _write_json(status_path, launch_payload)
    print(f"        launch_json = {status_path}")
    if not background:
        try:
            return_code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                return_code = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                return_code = proc.wait()
            launch_payload.update({
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "returncode": int(return_code),
                "interrupted": True,
            })
            _write_json(status_path, launch_payload)
            raise
        launch_payload.update({
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "returncode": int(return_code),
        })
        _write_json(status_path, launch_payload)
        if return_code != 0:
            raise SystemExit(return_code)
    return proc.pid


def _start_train_for_queue(ds_id: int, fold: int, gpu: int,
                           plans: str, config: str,
                           trainer: str,
                           loss_profile: str,
                           ce_class_weights: str,
                           oversample_profile: str,
                           env_profile: str,
                           queue_tag: str) -> subprocess.Popen:
    """Start one nnU-Net train process for the managed queue.

    This intentionally keeps a `Popen` handle instead of using `train()`, whose
    detached launch returns only a PID. The queue needs to know exactly when a
    dataset finished before it assigns the next dataset to the freed GPU.
    """
    cfg = DATASETS[ds_id]
    cmd = [
        _nnunet_train_bin(),
        str(ds_id), config, str(fold),
        "-tr", trainer, "-p", plans,
    ]
    init_weights = None
    if init_weights:
        cmd += ["-pretrained_weights", init_weights]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if env_profile not in ENV_PROFILES:
        raise ValueError(f"env_profile must be one of {sorted(ENV_PROFILES)}")
    for k, v in ENV_PROFILES[env_profile].items():
        env[k] = v
    env["PENGWIN_LOSS_PROFILE"] = loss_profile
    env["PENGWIN_CE_CLASS_WEIGHTS"] = ce_class_weights
    env["PENGWIN_OVERSAMPLE_PROFILE"] = oversample_profile

    # No wrapper log files. Queue state is JSON-only; nnU-Net checkpoints remain
    # the durable source for model weights and training artifacts.
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    print(
        f"[queue] start Ds{ds_id} {cfg['name']} fold={fold} gpu={gpu} "
        f"trainer={trainer} pid={proc.pid} logs=disabled env_profile={env_profile} "
        f"loss={loss_profile} ce={ce_class_weights} oversample={oversample_profile} "
        f"init={init_weights or 'scratch'}",
        flush=True,
    )
    return proc


def train_queue(ds_ids: list[int] | None = None,
                fold: int = 0,
                gpus: list[int] | None = None,
                plans: str = "nnUNetResEncUNetLPlans",
                config: str = "3d_fullres",
                trainer: str | None = None,
                loss_profile: str | None = None,
                ce_class_weights: str | None = None,
                oversample_profile: str | None = None,
                env_profile: str = "speed",
                poll_sec: int = 120,
                continue_on_error: bool = False,
                tag: str = "fold0_rebuild_20260507") -> Path:
    """Run the planned fold training order with a small GPU pool.

    The queue starts jobs in `ds_ids` order, keeps at most one process per GPU,
    and writes a status JSON after every state transition. This is the practical
    way to execute the full rebuild plan on a two-GPU workstation without
    manually babysitting nine long nnU-Net runs.
    """
    ds_ids = ds_ids or list(DEFAULT_FOLD0_ORDER)
    gpus = gpus or [0, 1]
    for ds_id in ds_ids:
        if ds_id not in DATASETS:
            raise ValueError(f"unknown dataset {ds_id}. Choose from {sorted(DATASETS)}")
    if not gpus:
        raise ValueError("at least one GPU id is required")
    trainer = trainer or DATASETS[ds_ids[0]]["trainer"]
    if trainer not in TRAINERS:
        raise ValueError(f"trainer must be one of {TRAINERS}")
    loss_profile, ce_class_weights, oversample_profile = _resolve_training_profiles(
        trainer, loss_profile, ce_class_weights, oversample_profile
    )
    if loss_profile not in LOSS_PROFILES:
        raise ValueError(f"loss_profile must be one of {LOSS_PROFILES}")
    if ce_class_weights not in CE_WEIGHT_PROFILES:
        raise ValueError(f"ce_class_weights must be one of {CE_WEIGHT_PROFILES}")
    if oversample_profile not in OVERSAMPLE_PROFILES:
        raise ValueError(f"oversample_profile must be one of {OVERSAMPLE_PROFILES}")

    status_path = RESULT_REPORT / f"train_queue_{tag}.json"
    pending = list(ds_ids)
    active: dict[int, dict] = {}
    completed = []
    failed = []

    def write_status() -> None:
        status = {
            "tag": tag,
            "fold": fold,
            "plans": plans,
            "config": config,
            "trainer": trainer,
            "loss_profile": loss_profile,
            "ce_class_weights": ce_class_weights,
            "oversample_profile": oversample_profile,
            "env_profile": env_profile,
            "pending": pending,
            "active": {
                str(gpu): {
                    "dataset": job["dataset"],
                    "pid": job["proc"].pid,
                    "started_at": job["started_at"],
                }
                for gpu, job in active.items()
            },
            "completed": completed,
            "failed": failed,
            "done": not pending and not active,
        }
        _write_json(status_path, status)

    print(f"[queue] datasets={ds_ids} fold={fold} gpus={gpus}", flush=True)
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            ds_id = pending.pop(0)
            proc = _start_train_for_queue(
                ds_id, fold, gpu, plans, config, trainer, loss_profile,
                ce_class_weights, oversample_profile, env_profile, tag,
            )
            active[gpu] = {
                "dataset": ds_id,
                "proc": proc,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            write_status()

        time.sleep(max(10, int(poll_sec)))

        for gpu, job in list(active.items()):
            ret = job["proc"].poll()
            if ret is None:
                continue
            record = {
                "dataset": job["dataset"],
                "gpu": gpu,
                "pid": job["proc"].pid,
                "returncode": ret,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            if ret == 0:
                completed.append(record)
                print(f"[queue] done Ds{job['dataset']} gpu={gpu}", flush=True)
            else:
                failed.append(record)
                print(f"[queue] failed Ds{job['dataset']} gpu={gpu} rc={ret}", flush=True)
                if not continue_on_error:
                    pending.clear()
            del active[gpu]
            write_status()

    write_status()
    print(f"[queue] status saved: {status_path}", flush=True)
    return status_path


def _parse_training_log(path: Path, class_names: list[str] | None = None) -> dict:
    """Parse a nnU-Net training log into a compact status row.

    nnU-Net logs pseudo Dice without background. For Dataset537 V5 that order is
    `[exterior_context, interior_shell, core, contact_surface]`. The last value
    is the fracture/contact class, but promotion still requires fixed-decoder
    IoU-F because semantic Dice alone does not prove fragment topology.
    """
    epoch_re = re.compile(r"Epoch (\d+)")
    epoch_time_re = re.compile(r"Epoch time: ([0-9.]+) s")
    dice_re = re.compile(r"(?:Pseudo dice|Diagnostic pseudo dice) \[(.*)\]")
    best_re = re.compile(r"New best EMA (?:pseudo Dice|contact-energy score): ([0-9.]+)")
    contact_re = re.compile(r"Contact-energy val (\{.*\})")
    row = {
        "log": str(path),
        "last_epoch": None,
        "last_epoch_time_sec": None,
        "mean_recent_epoch_time_sec": None,
        "last_pseudo_dice": None,
        "pseudo_dice_history": [],
        "best_ema_pseudo_dice": None,
        "best_ema_contact_energy_score": None,
        "best_epoch": None,
        "training_done": False,
        "contact_energy_history": [],
    }
    epoch_times = []
    current_epoch = None
    for line in path.read_text(errors="ignore").splitlines():
        if m := epoch_re.search(line):
            current_epoch = int(m.group(1))
            row["last_epoch"] = current_epoch
        if m := epoch_time_re.search(line):
            value = float(m.group(1))
            row["last_epoch_time_sec"] = value
            epoch_times.append(value)
        if m := dice_re.search(line):
            # [QC][MetricParsing]
            # nnU-Net may log small rare-class values as scientific notation
            # (`np.float32(1e-04)`). Dropping those values shifts class names and
            # corrupts the audit trail for V5 core/contact collapse analysis.
            number_re = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
            vals = [float(x) for x in re.findall(rf"np\.float32\(({number_re})\)", m.group(1))]
            if not vals:
                vals = [float(x) for x in re.findall(number_re, m.group(1))]
            row["last_pseudo_dice"] = vals
            labels = class_names or ["Sacrum", "LeftHip", "RightHip", "Femur"]
            by_class = {
                labels[i]: vals[i] for i in range(min(len(labels), len(vals)))
            }
            row["last_pseudo_dice_by_class"] = by_class
            row["pseudo_dice_history"].append({
                "epoch": current_epoch,
                "values": vals,
                "by_class": by_class,
            })
            row["pseudo_dice_excludes_background"] = True
        if m := best_re.search(line):
            value = float(m.group(1))
            row["best_ema_pseudo_dice"] = value
            if "contact-energy" in line:
                row["best_ema_contact_energy_score"] = value
            row["best_epoch"] = current_epoch
        if m := contact_re.search(line):
            try:
                metrics = ast.literal_eval(m.group(1))
                row["last_contact_energy_val"] = metrics
                row["contact_energy_history"].append({
                    "epoch": current_epoch,
                    **metrics,
                })
            except (SyntaxError, ValueError):
                pass
        if "Training done." in line:
            row["training_done"] = True
    if epoch_times:
        recent = epoch_times[-20:]
        row["mean_recent_epoch_time_sec"] = float(sum(recent) / len(recent))
    separator_name = "contact_surface" if class_names and "contact_surface" in class_names else "border"
    if class_names and separator_name in class_names and row["pseudo_dice_history"]:
        border_values = [
            item["by_class"][separator_name]
            for item in row["pseudo_dice_history"]
            if separator_name in item["by_class"]
        ]
        zero_epochs = [
            item["epoch"]
            for item in row["pseudo_dice_history"]
            if item["by_class"].get(separator_name) == 0.0
        ]
        row["separator_pseudo_dice_qc"] = {
            "class_order": class_names,
            "observed_values": border_values,
            "zero_epochs": zero_epochs,
            "last_value": border_values[-1] if border_values else None,
        "interpretation": (
            "Sparse class-3 separator signal; inspect target/predicted contact "
            "fractions and official decoder fragment metrics before promotion."
        ),
        }
    return row


def _training_metric_class_names(ds_id: int) -> list[str]:
    """Names for nnU-Net pseudo dice entries, excluding background.

    [METRIC][Scope:status]
    These names are for status parsing only. Dataset537 V5 is still selected by
    the later fixed decoder gate, not by nnU-Net's mean foreground Dice.
    """
    cfg = DATASETS[ds_id]
    if cfg["kind"] == "bicm_v5":
        return [
            "exterior_context", "interior_shell", "core", "contact_surface",
        ]
    return list(cfg["anatomies"])


def training_status(ds_id: int = 532,
                    folds: list[int] | None = None,
                    plans: str = "nnUNetResEncUNetLPlans",
                    config: str = "3d_fullres",
                    trainer: str | None = None,
                    out_json: Path | None = None) -> dict:
    """Summarize active fold logs, performance, and early-stop ETA."""
    folds = folds or [0, 1, 2, 3, 4]
    cfg = DATASETS[ds_id]
    trainer = trainer or cfg["trainer"]
    root = NN_RES / cfg["name"] / f"{trainer}__{plans}__{config}"
    rows = []
    for fold in folds:
        fdir = root / f"fold_{fold}"
        logs = sorted(fdir.glob("training_log_*.txt"))
        if not logs:
            continue
        row = _parse_training_log(logs[-1], class_names=_training_metric_class_names(ds_id))
        row["fold"] = fold
        row["checkpoint_best"] = str(fdir / "checkpoint_best.pth") if (fdir / "checkpoint_best.pth").exists() else None
        row["checkpoint_latest"] = str(fdir / "checkpoint_latest.pth") if (fdir / "checkpoint_latest.pth").exists() else None
        row["checkpoint_final"] = str(fdir / "checkpoint_final.pth") if (fdir / "checkpoint_final.pth").exists() else None
        row["completed"] = bool(row["training_done"] or row["checkpoint_final"])
        if not row["completed"] and row["last_epoch"] is not None and row["best_epoch"] is not None:
            min_epoch = 100
            stop_epoch = max(min_epoch, int(row["best_epoch"]) + 50)
            remaining = max(0, stop_epoch - int(row["last_epoch"]))
            row["earliest_early_stop_epoch"] = stop_epoch
            row["epochs_to_earliest_stop"] = remaining
            if row["mean_recent_epoch_time_sec"]:
                row["eta_to_earliest_stop_hours"] = remaining * row["mean_recent_epoch_time_sec"] / 3600.0
        rows.append(row)
    status = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": ds_id,
        "dataset_name": cfg["name"],
        "plans": plans,
        "config": config,
        "trainer": trainer,
        "pseudo_dice_note": (
            "nnU-Net pseudo dice excludes background; values map to "
            f"{_training_metric_class_names(ds_id)}. Dataset537 V5 uses a plain "
            "5-class BICM semantic contract, so `mean_fg_dice` is diagnostic only; "
            "promotion still requires Task1 official-aligned proxy metrics."
        ),
        "folds": rows,
    }
    if out_json:
        _write_json(out_json, status)
    return status


# =============================================================================
# CLI
# =============================================================================
def main():
    # `stunet-finetune` forwards ALL remaining args verbatim to nnU-Net's
    # run_training_entry (which has its own parser using -tr/-p/-pretrained_weights
    # style flags), so it is intercepted before our argparse to avoid clashing.
    if len(sys.argv) > 1 and sys.argv[1] == "stunet-finetune":
        run_stunet_finetune(sys.argv[2:])
        return

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train a dataset")
    p_train.add_argument("dataset", type=int)
    p_train.add_argument("--fold", type=int, default=0)
    p_train.add_argument("--gpu", type=int, default=0)
    p_train.add_argument("--init", help="path to pretrained_weights .pth")
    p_train.add_argument("--full-init-network", action="store_true",
                         help="include segmentation heads in pretrained transfer")
    p_train.add_argument("--plans", default="nnUNetResEncUNetLPlans")
    p_train.add_argument("--config", default="3d_fullres")
    p_train.add_argument("--trainer", default=None, choices=TRAINERS,
                         help="defaults to dataset registry trainer")
    p_train.add_argument("--loss-profile", default=None,
                         choices=LOSS_PROFILES)
    p_train.add_argument("--ce-class-weights", default=None,
                         choices=CE_WEIGHT_PROFILES)
    p_train.add_argument("--oversample-profile", default=None,
                         choices=OVERSAMPLE_PROFILES)
    p_train.add_argument("--env-profile", default="speed",
                         choices=sorted(ENV_PROFILES),
                         help="speed for normal training, stable for OOM-prone resumes, single for tiny overfit/debug")
    p_train.add_argument("--continue-training", action="store_true",
                         help="resume from checkpoint_latest.pth with nnU-Net --c")
    p_train.add_argument("--tag", default=None,
                         help="result/train_launch_<tag>.json name; defaults to timestamp")
    p_train.add_argument("--background", action="store_true",
                         help="detach and return immediately; not recommended when PID 1 is not a reaper")

    p_queue = sub.add_parser("queue", help="Run planned fold training queue")
    p_queue.add_argument("--datasets", default=",".join(map(str, DEFAULT_FOLD0_ORDER)),
                         help="comma-separated dataset order")
    p_queue.add_argument("--fold", type=int, default=0)
    p_queue.add_argument("--gpus", default="0,1",
                         help="comma-separated GPU ids used as a small pool")
    p_queue.add_argument("--plans", default="nnUNetResEncUNetLPlans")
    p_queue.add_argument("--config", default="3d_fullres")
    p_queue.add_argument("--trainer", default=None, choices=TRAINERS)
    p_queue.add_argument("--loss-profile", default=None,
                         choices=LOSS_PROFILES)
    p_queue.add_argument("--ce-class-weights", default=None,
                         choices=CE_WEIGHT_PROFILES)
    p_queue.add_argument("--oversample-profile", default=None,
                         choices=OVERSAMPLE_PROFILES)
    p_queue.add_argument("--env-profile", default="speed",
                         choices=sorted(ENV_PROFILES))
    p_queue.add_argument("--poll-sec", type=int, default=120)
    p_queue.add_argument("--continue-on-error", action="store_true")
    p_queue.add_argument("--tag", default=f"fold0_active_{RESULT_DATE}")

    p_status = sub.add_parser("status", help="Summarize training logs and early-stop ETA")
    p_status.add_argument("--dataset", type=int, default=532)
    p_status.add_argument("--folds", nargs="*", type=int, default=[0, 1])
    p_status.add_argument("--plans", default="nnUNetResEncUNetLPlans")
    p_status.add_argument("--config", default="3d_fullres")
    p_status.add_argument("--trainer", default=None, choices=TRAINERS)
    p_status.add_argument("--out-json")

    args = p.parse_args()

    if args.cmd == "train":
        if args.dataset not in DATASETS:
            print(f"unknown dataset {args.dataset}. Choose from {list(DATASETS)}")
            sys.exit(1)
        train(args.dataset, fold=args.fold, gpu=args.gpu,
              plans=args.plans, config=args.config, init_weights=args.init,
              full_init_network=args.full_init_network,
              trainer=args.trainer,
              loss_profile=args.loss_profile,
              ce_class_weights=args.ce_class_weights,
              oversample_profile=args.oversample_profile,
              env_profile=args.env_profile,
              continue_training=args.continue_training,
              tag=args.tag,
              background=args.background)

    elif args.cmd == "queue":
        ds_ids = [int(x) for x in args.datasets.split(",") if x.strip()]
        gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
        train_queue(
            ds_ids=ds_ids,
            fold=args.fold,
            gpus=gpus,
            plans=args.plans,
            config=args.config,
            trainer=args.trainer,
            loss_profile=args.loss_profile,
            ce_class_weights=args.ce_class_weights,
            oversample_profile=args.oversample_profile,
            env_profile=args.env_profile,
            poll_sec=args.poll_sec,
            continue_on_error=args.continue_on_error,
            tag=args.tag,
        )
    elif args.cmd == "status":
        status = training_status(
            ds_id=args.dataset,
            folds=args.folds,
            plans=args.plans,
            config=args.config,
            trainer=args.trainer,
            out_json=Path(args.out_json) if args.out_json else None,
        )
        print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()

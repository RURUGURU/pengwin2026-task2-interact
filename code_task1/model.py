"""PENGWIN 2026 Task 1 model utilities.

The active V5 fragment path intentionally uses the stock-compatible
`PengwinTrainer` Dice+CE contract; V3/V4 custom trainers are no longer active.

This module provides:
    - SWA (Stochastic Weight Averaging) over checkpoints
    - Trainer name to import-path mapping
    - Pretrained weight discovery
    - Optional plan patching hook
"""
from __future__ import annotations
import argparse
import copy
import json
import glob as _glob
from pathlib import Path
from typing import Dict, List

import torch

from core import (
    DATASETS, NN_PREP, NN_RES, RESULT_DATE, RESULT_REPORT, RESULT_WEIGHT,
    configure_nnunet_env, get_logger,
)
configure_nnunet_env()
log = get_logger(__name__)


# =============================================================================
# STU-Net network architecture — PENGWIN 2026 Task 1 백본 후보 (V0.x).
# (consolidated here from the former stunet.py module — 2026-06-12)
#
# 출처 / 라이선스
# --------------
# - 원본: https://github.com/uni-medical/STU-Net (nnUNet-2.2/.../STUNetTrainer.py)
# - 논문: Huang et al., "STU-Net: Scalable and Transferable Medical Image
#   Segmentation Models Empowered by Large-Scale Supervised Pre-training",
#   arXiv:2304.06716.
# - 라이선스: Apache License 2.0 (license-clean). TotalSegmentator(1204 CT, 104 구조 /
#   59개 뼈: sacrum, hip_left/right, femur_left/right 포함) 4000-epoch 사전학습 가중치
#   전이를 위해 본 네트워크를 vendor.
#
# [V0.x][2026-06-01] vendoring 원칙
# --------------------------------
# - 클래스/속성 이름(conv_blocks_context, upsample_layers, conv_blocks_localization,
#   seg_outputs, BasicResBlock.conv1/norm1/conv2/norm2/conv3, Upsample_Layer_nearest.conv)
#   은 **원본과 byte-identical** 로 유지한다. 사전학습 state_dict 키가 그대로 매칭돼야
#   warm-start 가 동작하기 때문이다. 이름을 절대 바꾸지 말 것.
# - nnU-Net 2.5.2 용 trainer(build_network_architecture 새 시그니처)는 core.py 에 별도
#   정의하고, 본 섹션에서는 STUNet nn.Module 만 제공한다(프레임워크 비의존).
# - STUNet 은 `self.decoder.deep_supervision` 를 노출하므로 nnU-Net 의
#   set_deep_supervision_enabled 및 우리 BADB 래퍼(.decoder forward)와 호환된다.
#   단 `.encoder` 속성은 없으므로 래퍼에서 방어적으로 처리해야 한다.
#
# 변형 변이(architecture variants)
# ------------------------------
# - small : dims=[16·(1,2,4,8,16,16)], depth=[1]*6  (~14.6M)
# - base  : dims=[32·(1,2,4,8,16,16)], depth=[1]*6  (~58.26M)  ← PENGWIN 채택
# - large : dims=[64·(1,2,4,8,16,16)], depth=[2]*6  (~440.30M)
# - huge  : dims=[96·(1,2,4,8,16,16)], depth=[3]*6  (~1.46B)
# =============================================================================
from torch import nn


class Decoder(nn.Module):
    """nnU-Net 호환을 위한 최소 decoder 스텁.

    nnU-Net 의 set_deep_supervision_enabled 는 `network.decoder.deep_supervision`
    을 직접 토글한다. STUNet 은 단일 forward 안에서 deep supervision 출력을 만들지만,
    이 플래그를 노출하기 위해 stub decoder 를 둔다.
    """

    def __init__(self):
        super().__init__()
        self.deep_supervision = True


class BasicResBlock(nn.Module):
    """Conv-IN-LReLU x2 + (선택적) 1x1 shortcut 의 residual block."""

    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1,
                 stride=1, use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        if use_1x1conv:
            self.conv3 = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)


class Upsample_Layer_nearest(nn.Module):
    """nearest interpolate + 1x1 conv 업샘플."""

    def __init__(self, input_channels, output_channels, pool_op_kernel_size, mode='nearest'):
        super().__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x


class STUNet(nn.Module):
    """STU-Net: residual U-Net (6 stage, 고정 dims/depth per variant).

    pool_op_kernel_sizes(strides) 는 nnU-Net plan 에서 주입된다. conv 가중치는 stride 와
    무관하게 shape 가 결정되므로(dims·kernel 고정) 사전학습 가중치가 그대로 로드된다.
    """

    def __init__(self, input_channels, num_classes, depth=[1, 1, 1, 1, 1, 1],
                 dims=[32, 64, 128, 256, 512, 512],
                 pool_op_kernel_sizes=None, conv_kernel_sizes=None,
                 enable_deep_supervision=True):
        super().__init__()
        self.conv_op = nn.Conv3d
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.final_nonlin = lambda x: x
        self.decoder = Decoder()
        self.decoder.deep_supervision = enable_deep_supervision
        self.upscale_logits = False

        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        num_pool = len(pool_op_kernel_sizes)

        assert num_pool == len(dims) - 1

        # encoder
        self.conv_blocks_context = nn.ModuleList()
        stage = nn.Sequential(
            BasicResBlock(input_channels, dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0], use_1x1conv=True),
            *[BasicResBlock(dims[0], dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0]) for _ in range(depth[0] - 1)])
        self.conv_blocks_context.append(stage)
        for d in range(1, num_pool + 1):
            stage = nn.Sequential(
                BasicResBlock(dims[d - 1], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d],
                              stride=self.pool_op_kernel_sizes[d - 1], use_1x1conv=True),
                *[BasicResBlock(dims[d], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d]) for _ in range(depth[d] - 1)])
            self.conv_blocks_context.append(stage)

        # upsample_layers
        self.upsample_layers = nn.ModuleList()
        for u in range(num_pool):
            upsample_layer = Upsample_Layer_nearest(dims[-1 - u], dims[-2 - u], pool_op_kernel_sizes[-1 - u])
            self.upsample_layers.append(upsample_layer)

        # decoder
        self.conv_blocks_localization = nn.ModuleList()
        for u in range(num_pool):
            stage = nn.Sequential(
                BasicResBlock(dims[-2 - u] * 2, dims[-2 - u], self.conv_kernel_sizes[-2 - u], self.conv_pad_sizes[-2 - u], use_1x1conv=True),
                *[BasicResBlock(dims[-2 - u], dims[-2 - u], self.conv_kernel_sizes[-2 - u], self.conv_pad_sizes[-2 - u]) for _ in range(depth[-2 - u] - 1)])
            self.conv_blocks_localization.append(stage)

        # outputs
        self.seg_outputs = nn.ModuleList()
        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(nn.Conv3d(dims[-2 - ds], num_classes, kernel_size=1))

        self.upscale_logits_ops = []
        for usl in range(num_pool - 1):
            self.upscale_logits_ops.append(lambda x: x)

    def forward(self, x):
        skips = []
        seg_outputs = []

        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.conv_blocks_localization)):
            x = self.upsample_layers[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self.decoder.deep_supervision:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]


# 변형별 dims/depth — core.py 의 trainer build_network_architecture 가 참조.
STUNET_VARIANTS = {
    "small": {"dims": [16 * x for x in [1, 2, 4, 8, 16, 16]], "depth": [1] * 6},
    "base":  {"dims": [32 * x for x in [1, 2, 4, 8, 16, 16]], "depth": [1] * 6},
    "large": {"dims": [64 * x for x in [1, 2, 4, 8, 16, 16]], "depth": [2] * 6},
    "huge":  {"dims": [96 * x for x in [1, 2, 4, 8, 16, 16]], "depth": [3] * 6},
}


def load_stunet_pretrained_weights(network, fname, verbose=False, inflate="ct0"):
    """STU-Net TotalSegmentator 사전학습 가중치를 warm-start 로 적재한다.

    STU-Net 공식 run_finetuning_stunet.py 의 로더를 우리 파이프라인에 맞게 개선:

    1. **BADB 래퍼 prefix 처리** — fracture 모델은 STUNet 이 `_V300...Network.base` 에
       들어가 state_dict 키가 `base.conv_blocks_context...` 가 된다. pretrained 키
       (prefix 없음)에 동일 prefix 를 부여해 매칭한다. anatomy 모델은 prefix="".
    2. **하드 assert 제거** — BADB refinement block(`...boundary_refine...`) 처럼
       pretrained 에 없는 신규 모듈은 자신의 초기값(BADB 는 zero-init)을 유지한다.
       원본 로더는 비-seg 키 전부가 pretrained 에 존재한다고 assert 하여 신규 모듈에서
       깨진다.
    3. **seg head skip** — STUNet 의 출력 head 는 `seg_outputs.*` 네이밍(nnU-Net 의
       `.seg_layers.` 가 아님). 클래스 수가 다르므로(105 → 5/4) 항상 reinit.
    4. **입력 stem inflate** — pretrained 는 1ch(CT). 우리 입력이 N>1 채널이면:
         - inflate="ct0"  (기본): 채널0 = pretrained CT 가중치, 나머지 채널 = 0.
           우리 입력(ct_lut, anatprob, sdf)은 CT modality 복제가 아니므로 tiling 보다
           "보조 채널은 0 에서 학습" 이 안정적이다(표준 신규-채널 전이 init).
         - inflate="repeat": STU-Net 공식(멀티모달 동일 modality 가정) — CT 가중치를
           모든 채널에 복제.

    반환: dict(loaded, skipped_seg, kept_init, total_model_keys) — QA 용 통계.
    """
    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP
    try:
        from torch._dynamo import OptimizedModule
    except Exception:  # pragma: no cover
        OptimizedModule = ()

    # [SECURITY][2026-06-01] weights_only=True 로 안전 역직렬화 — 임의 pickle 코드 실행
    # (외부 체크포인트發 RCE) 차단. nnU-Net v1 .model 체크포인트는 메타데이터에 numpy
    # 스칼라/배열을 담으므로, 임의코드 실행과 무관한 numpy 재구성 global 만 명시 allowlist
    # 한다(여전히 weights_only=True 안전 경로 유지). 부득이 unsafe 가 필요하면 출처를 검증한
    # 뒤 PENGWIN_ALLOW_UNSAFE_TORCH_LOAD=1 을 명시.
    import os as _os
    _unsafe = _os.environ.get("PENGWIN_ALLOW_UNSAFE_TORCH_LOAD", "") == "1"
    if not _unsafe:
        try:
            import numpy as _np
            import torch.serialization as _ts
            _safe = [_np.ndarray, _np.dtype,
                     _np.core.multiarray.scalar, _np.core.multiarray._reconstruct]
            try:
                import numpy.dtypes as _ndt
                _safe += [getattr(_ndt, _n) for _n in dir(_ndt) if _n.endswith("DType")]
            except Exception:
                pass
            _ts.add_safe_globals(_safe)
        except Exception:
            pass
    try:
        saved = torch.load(fname, map_location="cpu", weights_only=not _unsafe)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            f"체크포인트 안전 로드 실패 ({fname}): {e}\n"
            f"출처를 검증한 뒤에만 PENGWIN_ALLOW_UNSAFE_TORCH_LOAD=1 로 재시도하세요."
        ) from e
    if isinstance(saved, dict) and "network_weights" in saved:
        pretrained = saved["network_weights"]
    elif isinstance(saved, dict) and "state_dict" in saved:
        pretrained = saved["state_dict"]
    else:
        pretrained = saved  # raw state_dict 가정

    mod = network
    if isinstance(mod, DDP):
        mod = mod.module
    if OptimizedModule and isinstance(mod, OptimizedModule):
        mod = mod._orig_mod
    model_dict = mod.state_dict()

    # STUNet 키 prefix 탐지 (anatomy: "" / fracture BADB: "base.")
    anchor = "conv_blocks_context.0.0.conv1.weight"
    if anchor in model_dict:
        prefix = ""
    else:
        cands = [k[: -len(anchor)] for k in model_dict if k.endswith(anchor)]
        if not cands:
            raise RuntimeError("STUNet anchor 키 부재 — STUNet 백본이 아니거나 구조 불일치")
        prefix = cands[0]

    pretrained = {prefix + k: v for k, v in pretrained.items()}
    skip_token = prefix + "seg_outputs"

    # 입력 stem inflate
    stem1 = prefix + "conv_blocks_context.0.0.conv1.weight"
    stem3 = prefix + "conv_blocks_context.0.0.conv3.weight"
    num_inputs = model_dict[stem1].shape[1]
    if num_inputs > 1 and stem1 in pretrained and pretrained[stem1].shape[1] == 1:
        for sk in (stem1, stem3):
            w1 = pretrained[sk]  # [out, 1, k, k, k]
            if inflate == "repeat":
                pretrained[sk] = w1.repeat(1, num_inputs, 1, 1, 1)
            else:  # "ct0"
                w = torch.zeros((w1.shape[0], num_inputs, *w1.shape[2:]), dtype=w1.dtype)
                w[:, 0:1] = w1
                pretrained[sk] = w

    loaded, skipped_seg, kept_init = [], [], []
    use = {}
    for k, v in model_dict.items():
        if skip_token in k:
            skipped_seg.append(k)
            continue
        if k in pretrained and pretrained[k].shape == v.shape:
            use[k] = pretrained[k]
            loaded.append(k)
        else:
            kept_init.append(k)  # BADB 등 신규 모듈 → 초기값 유지

    model_dict.update(use)
    mod.load_state_dict(model_dict)

    stats = {
        "loaded": len(loaded),
        "skipped_seg": len(skipped_seg),
        "kept_init": len(kept_init),
        "total_model_keys": len(model_dict),
        "prefix": prefix or "(none)",
        "inflate": inflate if num_inputs > 1 else "n/a(1ch)",
        "num_inputs": num_inputs,
    }
    print(f"[STU-Net warm-start] {fname}: loaded={stats['loaded']} "
          f"skipped_seg={stats['skipped_seg']} kept_init={stats['kept_init']} "
          f"prefix='{stats['prefix']}' inflate={stats['inflate']}")
    if verbose:
        for k in kept_init:
            print("  kept-init(new module):", k)
    return stats



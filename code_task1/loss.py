"""PENGWIN 2026 Task 1 — Custom loss components (v7).

Provides optional BoundaryDoU, Tversky, and class-weighted CE components on top
of nnU-Net's default Dice + CE. The production default is still plain DC+CE,
matching the PENGWIN 2024 top CT recipes; weak-class/boundary weighting are
explicit ablation profiles.

    - 4 anatomies (Sa/LH/RH/Femur), 1-200 instance IDs.
    - Pelvic cases (001-120, 151-200) carry only Sa/LH/RH.
    - Femur cases (251-420) carry only Femur.
    - Fragment counts are highly variable (2026 audit: Pelvic can exceed
      20 total fragments; RightHip is the densest hard anatomy).
    - Boundary voxels at fracture surfaces are the hardest signal —
      Dice ignores them; CE treats them uniformly.

BoundaryDoU is not enabled by default in the trainer; use loss profiles for
controlled ablation before promoting it.

Reference:
    BoundaryDoU — Sun et al., "Boundary Difference Over Union Loss for
    Medical Image Segmentation," MICCAI 2023.
    https://arxiv.org/abs/2308.00220 (2D in paper; we adapt to 3D below.)
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi

# Single source of truth for the anatomy<->instance-ID encoding. Instance IDs in
# this module are torch tensors, so we use the registry's scalar bound rather than
# its numpy mask helpers: every "(x > 0) & (x <= 150)" support mask and "x > 150"
# guard becomes "<= MAX_INSTANCE_ID", which spans Femur (151-200). This is a safe
# superset — pelvic ROIs hold no 151-200 voxels, femur ROIs are no longer dropped.
# NOTE: "150" inside class names (BICMV150...) is a version tag, NOT an instance id.
from utils import MAX_INSTANCE_ID


# =============================================================================
# 1. Boundary DoU loss (3D) — surface-aware regularizer
# =============================================================================
class BoundaryDoULoss3D(nn.Module):
    """3D adaptation of Sun et al. MICCAI'23 Boundary DoU loss.

    Algorithm (per foreground class c, each batch item):
      1. Identify "interior" voxels of the target: voxels whose 6 axis-aligned
         neighbors are ALL inside the target. Boundary band = target − interior.
      2. Compute boundary fraction α_c = 1 − boundary / total (per class).
         Clip to ≤ 0.8 for numerical stability (paper recommendation).
      3. Loss_c = (z + y − 2·intersect) / (z + y − (1 + α)·intersect)
         where z = ||score||², y = ||target||², intersect = ⟨score, target⟩.
         When α → 1 (target mostly interior), denom → numer, loss saturates.
         When α → 0 (target is mostly boundary), loss reduces to DoU acting
         like standard Dice but emphasizing boundary alignment.
      4. Mean across foreground classes (background excluded).

    Why 3D adaptation matters:
        Original paper does per-slice 2D conv on (B, H, W). For volumetric CT
        (B, C, D, H, W), 2D convs would only see in-plane neighbors; we use a
        7-tap 3D cross kernel so erosion respects through-plane connectivity.
        This is critical for PENGWIN where fracture surfaces span multiple
        slices.

    Args:
        n_classes: total number of classes including background. Background
            (class 0) is skipped in the loss sum.
        smooth: numerical smoothing constant (1e-5 matches paper).
        background_index: class index treated as background (default 0).

    Forward:
        logits:      (B, C, D, H, W) raw network outputs (pre-softmax)
        target:      (B, D, H, W) integer class labels OR (B, 1, D, H, W)
    Returns: scalar loss (mean over foreground classes).
    """

    def __init__(self, n_classes: int, smooth: float = 1e-5,
                 background_index: int = 0):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth
        self.background_index = background_index
        # 6-connected 3D cross kernel (D=H=W=3). 7 ones total: center + 6 neighbors.
        # Convolving binary target with this counts how many of {self ∪ 6 axis
        # neighbors} are foreground. Maximum value 7 = "fully interior".
        kernel = torch.zeros(1, 1, 3, 3, 3)
        kernel[0, 0, 1, 1, 1] = 1   # center
        kernel[0, 0, 0, 1, 1] = 1   # -z
        kernel[0, 0, 2, 1, 1] = 1   # +z
        kernel[0, 0, 1, 0, 1] = 1   # -y
        kernel[0, 0, 1, 2, 1] = 1   # +y
        kernel[0, 0, 1, 1, 0] = 1   # -x
        kernel[0, 0, 1, 1, 2] = 1   # +x
        # Buffer (not parameter) — moves with .to(device) automatically.
        self.register_buffer("kernel", kernel)
        self._kernel_sum = float(kernel.sum().item())

    def _alpha(self, target_c: torch.Tensor) -> torch.Tensor:
        """Boundary fraction-derived α coefficient for one class.

        target_c: (B, D, H, W) binary {0,1} mask.
        Returns scalar α ∈ [-1, 0.8] (clipped per paper).
        """
        if target_c.sum() == 0:
            # Empty class in this batch — degenerate; return α=0 (≡ standard Dice form).
            return target_c.new_zeros(())

        x = target_c.unsqueeze(1).float()                # (B,1,D,H,W) — float32
        # AMP+device safe: kernel buffer was registered without .cuda(), and
        # nnU-Net's `loss = loss.to(device)` may not always propagate to
        # buffers added in subclasses. Explicitly match BOTH device + dtype.
        kernel = self.kernel.to(device=x.device, dtype=x.dtype)
        conv = F.conv3d(x, kernel, padding=1)       # neighbor sum
        # Voxel is "interior" iff ALL 7 kernel positions land on foreground.
        # Strict equality avoids surface tangents counting as interior.
        interior = (conv >= self._kernel_sum) & (x > 0)
        n_interior = interior.float().sum()
        n_total = x.sum() + self.smooth
        boundary_frac = 1.0 - (n_interior / n_total)
        # α = 1 - boundary_frac (so high α = "mostly interior" → loss less aggressive)
        alpha = 1.0 - boundary_frac
        # Map [0,1] → [-1, 1] then clamp ≤ 0.8 (paper Sec. 3.2 stability).
        alpha = (2 * alpha - 1).clamp(max=0.8)
        return alpha

    def _per_class(self, score_c: torch.Tensor,
                   target_c: torch.Tensor) -> torch.Tensor:
        """DoU-like loss for one class (paper Eq. 8 in 3D form)."""
        alpha = self._alpha(target_c)
        target_f = target_c.float()
        intersect = (score_c * target_f).sum()
        y_sum = (target_f * target_f).sum()
        z_sum = (score_c * score_c).sum()
        num = z_sum + y_sum - 2 * intersect + self.smooth
        den = z_sum + y_sum - (1.0 + alpha) * intersect + self.smooth
        return num / den

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """Compute mean boundary DoU loss across foreground classes.

        Inputs are aligned with nnU-Net convention:
            logits:  (B, C, D, H, W)  -- pre-softmax (any dtype: AMP fp16 OK)
            target:  (B, D, H, W) or (B, 1, D, H, W) integer labels

        AMP-safe: disables `torch.cuda.amp.autocast` inside, so all internal
        arithmetic (softmax + conv3d + DoU formula) runs in float32 regardless
        of the surrounding autocast context. Without this guard, calling
        `tensor.float()` inside autocast still gets re-cast to half on
        autocast-enabled ops (conv3d, mul, etc.), which then mismatches our
        float32 kernel buffer. nnU-Net's GradScaler handles the float32 scalar
        return correctly. Boundary loss is <5% of total compute so float32 is
        cheap.
        """
        if target.dim() == 5:
            target = target.squeeze(1)
        with torch.cuda.amp.autocast(enabled=False):
            logits_f32 = logits.float()
            probs = F.softmax(logits_f32, dim=1)
            n_fg = 0
            loss = logits_f32.new_zeros(())
            for c in range(self.n_classes):
                if c == self.background_index:
                    continue
                target_c = (target == c).long()
                score_c = probs[:, c]
                # Skip empty-on-both-sides classes to avoid degenerate gradients.
                if target_c.sum() == 0 and score_c.sum() < self.smooth:
                    continue
                loss = loss + self._per_class(score_c, target_c)
                n_fg += 1
            if n_fg == 0:
                return loss
            return loss / n_fg


# =============================================================================
# 2. Compound loss (Dice + CE + BoundaryDoU) wrapper
# =============================================================================
class DC_CE_BD_loss(nn.Module):
    """Compound loss: w_dc·Dice + w_ce·CE + w_bd·BoundaryDoU.

    Designed as a drop-in replacement for nnU-Net's `DC_and_CE_loss` so the
    deep-supervision wrapper continues to work unchanged. Wraps an existing
    `DC_and_CE_loss` instance (passed in) and adds a BoundaryDoU3D term.

    Default weights chosen for PENGWIN:
        w_dc = 1.0, w_ce = 1.0, w_bd = 0.3
        — Sun et al. recommend ~0.3 for additive boundary; matches typical
        compound loss budget where boundary pulls without dominating.

    Why we wrap rather than re-implement DC+CE:
        nnU-Net's DC_and_CE_loss handles `ignore_label`, `batch_dice`,
        DDP smoothing, and edge cases. Re-implementing risks subtle bugs.

    Args:
        dc_ce_loss:  pre-built nnU-Net DC_and_CE_loss instance.
        n_classes:   number of segmentation classes (incl. background).
        weight_dc_ce: weight for the DC+CE term (default 1.0).
        weight_bd:   weight for the boundary DoU term (default 0.3).

    Forward:
        logits:  (B, C, D, H, W)
        target:  (B, 1, D, H, W) integer labels (nnU-Net convention)
    """

    def __init__(self, dc_ce_loss: nn.Module, n_classes: int,
                 weight_dc_ce: float = 1.0, weight_bd: float = 0.3):
        super().__init__()
        self.dc_ce = dc_ce_loss
        self.bd = BoundaryDoULoss3D(n_classes=n_classes)
        self.weight_dc_ce = weight_dc_ce
        self.weight_bd = weight_bd
        # Expose `.dc` so nnU-Net's torch.compile path (which calls
        # `loss.dc = torch.compile(loss.dc)` in nnUNetTrainer._build_loss)
        # still works without modification.
        if hasattr(dc_ce_loss, "dc"):
            self.dc = dc_ce_loss.dc

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        l_dc_ce = self.dc_ce(logits, target)
        l_bd = self.bd(logits, target)
        return self.weight_dc_ce * l_dc_ce + self.weight_bd * l_bd


# =============================================================================
# 3. Tversky loss (3D) — weak-class / recall-oriented ablation
# =============================================================================
class SoftTverskyLoss3D(nn.Module):
    """Weighted multi-class Tversky loss for 3D anatomy segmentation.

    Tversky generalizes Dice with separate false-positive and false-negative
    weights:

        T = TP / (TP + alpha * FP + beta * FN)

    By default this keeps foreground-only behavior. Ablations can pass
    `active_classes` to include a targeted subset, including background.

    Args:
        n_classes: total classes including background.
        alpha: false-positive weight. Smaller values tolerate extra foreground.
        beta: false-negative weight. Larger values emphasize recall.
        smooth: numerical smoothing.
        background_index: class skipped only when active_classes is omitted.
        active_classes: explicit class IDs to include. If None, use all classes
            except background_index.
        class_weights: optional dense list or {class_id: weight}; weights are
            normalized by the sum of weights actually used in the batch.
    """

    def __init__(self, n_classes: int, alpha: float = 0.3, beta: float = 0.7,
                 smooth: float = 1e-5, background_index: int = 0,
                 active_classes: list[int] | tuple[int, ...] | None = None,
                 class_weights: list[float] | tuple[float, ...] | dict[int, float] | None = None):
        super().__init__()
        self.n_classes = int(n_classes)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)
        self.background_index = int(background_index)
        if active_classes is None:
            active_classes = [c for c in range(self.n_classes) if c != self.background_index]
        self.active_classes = tuple(int(c) for c in active_classes if 0 <= int(c) < self.n_classes)

        weight_arr = torch.ones(self.n_classes, dtype=torch.float32)
        if class_weights is not None:
            if isinstance(class_weights, dict):
                for k, v in class_weights.items():
                    k = int(k)
                    if 0 <= k < self.n_classes:
                        weight_arr[k] = float(v)
            else:
                arr = torch.as_tensor(class_weights, dtype=torch.float32)
                if arr.numel() != self.n_classes:
                    raise ValueError(
                        f"class_weights must have {self.n_classes} values, got {arr.numel()}"
                    )
                weight_arr = arr.reshape(self.n_classes)
        self.register_buffer("class_weights", weight_arr)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 5:
            target = target.squeeze(1)
        with torch.cuda.amp.autocast(enabled=False):
            logits_f32 = logits.float()
            probs = F.softmax(logits_f32, dim=1)
            loss = logits_f32.new_zeros(())
            weight_sum = logits_f32.new_zeros(())
            for c in self.active_classes:
                target_c = (target == c).float()
                score_c = probs[:, c]
                # Empty-on-both classes add noise but no useful gradient.
                if target_c.sum() == 0 and score_c.sum() < self.smooth:
                    continue
                tp = (score_c * target_c).sum()
                fp = (score_c * (1.0 - target_c)).sum()
                fn = ((1.0 - score_c) * target_c).sum()
                tversky = (tp + self.smooth) / (
                    tp + self.alpha * fp + self.beta * fn + self.smooth
                )
                weight = self.class_weights.to(device=logits_f32.device, dtype=logits_f32.dtype)[c]
                loss = loss + weight * (1.0 - tversky)
                weight_sum = weight_sum + weight
            if weight_sum <= 0:
                return loss
            return loss / weight_sum


class DC_CE_TV_BD_loss(nn.Module):
    """Compound loss: DC+CE plus optional Tversky and BoundaryDoU terms.

    This wrapper exists for controlled split-anatomy upgrades:

    - `tversky_07`: DC+CE + 0.5 * Tversky(alpha=0.3, beta=0.7)
    - `tversky_08`: DC+CE + 0.5 * Tversky(alpha=0.2, beta=0.8)
    - `combo_tversky_bd005`: DC+CE + 0.35 * Tversky + 0.05 * BoundaryDoU
    - `abbc_contact_energy_v1`: Contact-Energy ABBC core/contact Tversky for
      stable seeds and precise fracture surfaces

    We keep DC+CE as the anchor term so the optimization landscape remains
    close to the already-good baseline. Tversky/BD are small corrective forces,
    not replacements.
    """

    def __init__(self, dc_ce_loss: nn.Module, n_classes: int,
                 weight_dc_ce: float = 1.0,
                 weight_tversky: float = 0.0,
                 tversky_alpha: float = 0.3,
                 tversky_beta: float = 0.7,
                 weight_bd: float = 0.0,
                 tversky_active_classes: list[int] | tuple[int, ...] | None = None,
                 tversky_class_weights: list[float] | tuple[float, ...] | dict[int, float] | None = None):
        super().__init__()
        self.dc_ce = dc_ce_loss
        self.weight_dc_ce = float(weight_dc_ce)
        self.weight_tversky = float(weight_tversky)
        self.weight_bd = float(weight_bd)
        self.tversky = SoftTverskyLoss3D(
            n_classes=n_classes,
            alpha=tversky_alpha,
            beta=tversky_beta,
            active_classes=tversky_active_classes,
            class_weights=tversky_class_weights,
        )
        self.bd = BoundaryDoULoss3D(n_classes=n_classes) if weight_bd > 0 else None
        if hasattr(dc_ce_loss, "dc"):
            self.dc = dc_ce_loss.dc

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.weight_dc_ce * self.dc_ce(logits, target)
        if self.weight_tversky > 0:
            loss = loss + self.weight_tversky * self.tversky(logits, target)
        if self.bd is not None and self.weight_bd > 0:
            loss = loss + self.weight_bd * self.bd(logits, target)
        return loss


# =============================================================================
# 4. Class weight estimation — frequency-inverse with PENGWIN tweaks
# =============================================================================
def compute_class_weights(label_array_or_iter, n_classes: int,
                          min_weight: float = 0.5,
                          max_weight: float = 2.0,
                          smoothing: float = 1.0) -> np.ndarray:
    """Compute frequency-inverse class weights for cross-entropy.

    Args:
        label_array_or_iter: Either a single np.ndarray of integer labels OR
            an iterable of arrays (will be summed). For PENGWIN we pass a
            list of training-set GT label arrays (one per case).
        n_classes: total class count (incl. background).
        min_weight, max_weight: clamp range. Default (0.5, 2.0) prevents
            the rare class (e.g., a 0.05% sec fragment) from dominating
            gradients while still up-weighting it ~2×.
        smoothing: pseudocount added to each class count to avoid div-by-zero
            for classes absent in the sample. 1.0 ≡ Laplace smoothing.

    Returns:
        np.ndarray of shape (n_classes,), dtype float32.

    Algorithm:
        counts[c] = #voxels labeled c (across all provided arrays)
        freq[c] = (counts[c] + smoothing) / sum(counts + smoothing)
        weight[c] = 1 / freq[c], normalized so mean(weight) = 1.0,
                    then clamped to [min_weight, max_weight].

    Why PENGWIN specifics:
        Background dominates most voxels. Standard inverse frequency can give
        extreme rare-class weights, which can hurt calibration and boundary
        quality. Clamping to [0.5, 2.0] keeps this an ablation knob rather than
        a hidden training regime change.
    """
    if isinstance(label_array_or_iter, np.ndarray):
        arrays = [label_array_or_iter]
    else:
        arrays = list(label_array_or_iter)

    counts = np.zeros(n_classes, dtype=np.float64)
    for arr in arrays:
        # bincount is O(N); minlength ensures we cover all classes even if some
        # don't appear in this particular case (Pelvic case has no Femur, etc.).
        bc = np.bincount(arr.ravel(), minlength=n_classes).astype(np.float64)
        if len(bc) > n_classes:
            bc = bc[:n_classes]
        counts[:len(bc)] += bc

    counts_smooth = counts + smoothing
    freq = counts_smooth / counts_smooth.sum()
    weights = 1.0 / freq
    weights /= weights.mean()                  # mean-1 normalization
    weights = np.clip(weights, min_weight, max_weight).astype(np.float32)
    return weights


def compute_median_frequency_class_weights(label_array_or_iter, n_classes: int,
                                           min_weight: float = 0.5,
                                           max_weight: float = 2.0,
                                           smoothing: float = 1.0) -> np.ndarray:
    """Compute clipped median-frequency CE weights.

    This is the trainer's `CE_CLASS_WEIGHTS="auto"` implementation. Median
    frequency weighting is less aggressive than raw inverse frequency:

        weight[c] = median(freq[present classes]) / freq[c]

    The result is clipped to [0.5, 2.0], because PENGWIN IoU-F is very sensitive
    to fragment topology. Extreme CE weights can create more foreground blobs
    and worsen unmatched predicted fragment counts even if anatomy Dice rises.
    """
    if isinstance(label_array_or_iter, np.ndarray):
        arrays = [label_array_or_iter]
    else:
        arrays = list(label_array_or_iter)

    counts = np.zeros(n_classes, dtype=np.float64)
    for arr in arrays:
        bc = np.bincount(arr.ravel(), minlength=n_classes).astype(np.float64)
        counts += bc[:n_classes]
    counts_smooth = counts + smoothing
    freq = counts_smooth / counts_smooth.sum()
    present = counts > 0
    median = np.median(freq[present]) if present.any() else np.median(freq)
    weights = median / freq
    weights = np.clip(weights, min_weight, max_weight).astype(np.float32)
    return weights


# =============================================================================
# 4. Self-test (smoke) — run as `python loss.py`
# =============================================================================
def _smoke_test():
    """Quick sanity check — verify shapes/finite values on small CPU tensors."""
    print("[loss.py smoke]")
    torch.manual_seed(0)
    B, C, D, H, W = 1, 4, 8, 16, 16
    logits = torch.randn(B, C, D, H, W, requires_grad=True)
    target = torch.randint(0, C, (B, D, H, W))

    # 1. BoundaryDoULoss3D
    bd = BoundaryDoULoss3D(n_classes=C)
    val = bd(logits, target)
    assert torch.isfinite(val), f"BD loss not finite: {val}"
    val.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    print(f"  BoundaryDoULoss3D({C}c): {val.item():.4f}  ✓")

    # 2. compute_class_weights
    arr = np.random.randint(0, 4, size=(20, 20, 20))
    w = compute_class_weights(arr, n_classes=4)
    assert w.shape == (4,) and np.all(w > 0) and np.all(np.isfinite(w))
    print(f"  compute_class_weights: weights={w} (mean={w.mean():.3f})  ✓")

    print("[loss.py smoke] OK")


# ============================================================================
# Leak-free ABBC target primitives (extracted from the retired BoundaryFragment
# loss hierarchy so the active LeakFreeInstance{ABBC,XCAC}Loss are SELF-CONTAINED).
# torch-only; build the ABBC 4-class target + the 13-neighbour interface stencil
# from the raw per-anatomy fragment instance map (1..200; same-anatomy = (id-1)//50).
# ============================================================================
INSTANCE_ID_MAX = 200
AFFINITY13_OFFSETS_ZYX = (
    (0, 0, 1), (0, 1, -1), (0, 1, 0), (0, 1, 1),
    (1, -1, -1), (1, -1, 0), (1, -1, 1), (1, 0, -1), (1, 0, 0), (1, 0, 1),
    (1, 1, -1), (1, 1, 0), (1, 1, 1),
)


def _offset_slices(shape, offset_zyx):
    src, dst = [], []
    for dim, delta in zip(shape, offset_zyx):
        if abs(int(delta)) >= int(dim):
            raise ValueError(f"affinity offset {offset_zyx} invalid for shape {shape}")
        if int(delta) > 0:
            src.append(slice(0, -int(delta))); dst.append(slice(int(delta), None))
        elif int(delta) < 0:
            src.append(slice(-int(delta), None)); dst.append(slice(0, int(delta)))
        else:
            src.append(slice(None)); dst.append(slice(None))
    return tuple(src), tuple(dst)


def separator_gap_targets(instance):
    """support [B,Z,Y,X] bool + (separator & support): true same-anatomy inter-fragment 13-nbr band."""
    if instance.ndim == 5 and int(instance.shape[1]) == 1:
        instance = instance[:, 0]
    if instance.ndim != 4:
        raise ValueError(f"separator target requires instance [B,Z,Y,X], got {tuple(instance.shape)}")
    inst = instance.long()
    if bool(((inst < 0) | (inst > INSTANCE_ID_MAX)).any()):
        raise ValueError(f"instance IDs must be in [0, {INSTANCE_ID_MAX}]")
    support = (inst > 0) & (inst <= INSTANCE_ID_MAX)
    separator = torch.zeros_like(support, dtype=torch.bool)
    shape = tuple(int(v) for v in inst.shape[1:])
    for offset in AFFINITY13_OFFSETS_ZYX:
        ssl_z, dsl_z = _offset_slices(shape, offset)
        ssl = (slice(None), *ssl_z); dsl = (slice(None), *dsl_z)
        a = inst[ssl]; b = inst[dsl]
        a_fg = (a > 0) & (a <= INSTANCE_ID_MAX); b_fg = (b > 0) & (b <= INSTANCE_ID_MAX)
        diff = a_fg & b_fg & (((a - 1) // 50) == ((b - 1) // 50)) & (a != b)
        if bool(diff.any()):
            separator[ssl] = separator[ssl] | diff
            separator[dsl] = separator[dsl] | diff
    return support, separator & support


def abbc_class_target(instance, *, boundary_dilate_vox, core_erode_vox):
    """ABBC class target [B,Z,Y,X] long [0 bg,1 border,2 boundary,3 core] + support, from the instance map."""
    if instance.ndim == 5 and int(instance.shape[1]) == 1:
        instance = instance[:, 0]
    if instance.ndim != 4:
        raise ValueError(f"ABBC target requires instance [B,Z,Y,X], got {tuple(instance.shape)}")
    support, raw_between = separator_gap_targets(instance)
    non_support = ~support
    if int(boundary_dilate_vox) > 0 and bool(raw_between.any()):
        k = 2 * int(boundary_dilate_vox) + 1
        dilated = F.max_pool3d(raw_between.float().unsqueeze(1), kernel_size=k, stride=1, padding=int(boundary_dilate_vox))[:, 0]
        boundary = (dilated > 0.5) & support
    else:
        boundary = raw_between & support
    if int(core_erode_vox) > 0:
        k = 2 * int(core_erode_vox) + 1
        nsd = F.max_pool3d(non_support.float().unsqueeze(1), kernel_size=k, stride=1, padding=int(core_erode_vox))[:, 0]
        support_eroded = support & (nsd <= 0.5)
    else:
        support_eroded = support.clone()
    core = support_eroded & (~boundary)
    border = support & (~core) & (~boundary)
    class_target = torch.zeros_like(instance, dtype=torch.long)
    class_target[border] = 1
    class_target[boundary] = 2
    class_target[core] = 3
    return class_target, support


class LeakFreeInstanceABBCLoss(nn.Module):
    """Leak-free instance-label ABBC loss (boundary-weighted) — the working Phase-1 base loss. NO conn.

    Reads the nnUNet seg-label AS the per-anatomy fragment instance map (1..K; -1=nnUNet ignore, 0=bg)
    — NO sidecar, NO anatomy-context channel. Builds the validated ABBC 4-class target
    [bg,border,boundary,core] on the fly (reusing the V288/V277 classmethods) and trains the 4-channel
    head with valid-masked CE + foreground Dice, boundary class up-weighted (boundary is only ~5% of
    support voxels). The real submission decoder (decode_task1_v288_abbc: core-seed watershed REGROW +
    small-CC merge) turns this field into instances.

    SUPERSEDES the retired InstanceConnectivityABBCLoss: the loss-level merge/split topology penalty was
    mis-scaled (~25x the base — raw conn ~8.5 vs base ~0.32 at boost=8) and collapsed training the
    instant it ramped in. Under the real decoder the plain ABBC head reaches held-out instance-F1 ~0.85,
    so the connectivity term was both unstable AND unnecessary. docs/Plan.md Phase 1 ·
    [[pengwin-instance-label-nosidecar]]. Env: PENGWIN_ABBC_BOUNDARY_WEIGHT (default 5.0).
    """

    NUM_CLASSES = 4
    BG, BORDER, BOUNDARY, CORE = 0, 1, 2, 3
    BOUNDARY_DILATE_VOX = 2
    CORE_ERODE_VOX = 2

    def __init__(self):
        super().__init__()
        self.boundary_class_weight = float(os.environ.get("PENGWIN_ABBC_BOUNDARY_WEIGHT", "5.0"))
        if self.boundary_class_weight <= 0.0:
            raise ValueError(f"boundary_class_weight must be > 0, got {self.boundary_class_weight}")

    @staticmethod
    def _as_full_res(x):
        return x[0] if isinstance(x, (list, tuple)) else x

    def _split_instance(self, target):
        """Return (instance>=0 [B,Z,Y,X] long, ignore mask bool). -1 (nnUNet ignore) -> ignore+bg."""
        t = self._as_full_res(target)
        if t.ndim == 5 and int(t.shape[1]) == 1:
            t = t[:, 0]
        t = t.long()
        ignore = t < 0
        return t.clamp_min(0), ignore

    def _dice_ce(self, logits, class_target, valid, class_weight):
        ce_map = F.cross_entropy(logits, class_target, weight=class_weight, reduction="none")
        ce = ce_map[valid].mean() if bool(valid.any()) else logits.sum() * 0.0
        probs = torch.softmax(logits, dim=1)
        oh = F.one_hot(class_target, num_classes=self.NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
        vmask = valid.unsqueeze(1).float()
        probs = probs * vmask
        oh = oh * vmask
        dims = (0, 2, 3, 4)
        inter = (probs * oh).sum(dims)
        denom = probs.sum(dims) + oh.sum(dims)
        dice = (2 * inter + 1e-5) / (denom + 1e-5)
        dice_loss = 1.0 - dice[1:].mean()  # foreground classes (border/boundary/core)
        return ce + dice_loss

    def forward(self, pred_logits, target):
        logits = self._as_full_res(pred_logits)
        instance, ignore = self._split_instance(target)
        valid = ~ignore
        class_target, _support = abbc_class_target(
            instance, boundary_dilate_vox=self.BOUNDARY_DILATE_VOX, core_erode_vox=self.CORE_ERODE_VOX)
        cw = torch.ones(self.NUM_CLASSES, device=logits.device)
        cw[self.BOUNDARY] = self.boundary_class_weight
        base = self._dice_ce(logits, class_target, valid, cw)
        if not torch.isfinite(base):
            raise ValueError(f"LeakFreeInstanceABBCLoss produced non-finite loss: {float(base.detach().cpu())}")
        return base


# =============================================================================
# Tier-1: affinity head — short+long-range same-instance affinities, decoded by
# AVERAGE-LINKAGE agglomeration (NOT mutex). Direct instance-map supervision is a
# DENSE separation signal vs the noisy ABBC boundary class (which both X-CAC and
# fuzzy decode failed to exploit). Long-range repulsive offsets break the merge.
# =============================================================================
AFFINITY_HEAD_OFFSETS = (
    (1, 0, 0), (0, 1, 0), (0, 0, 1),      # short / attractive (nearest neighbour)
    (3, 0, 0), (0, 3, 0), (0, 0, 3),      # mid-range
    (9, 0, 0), (0, 9, 0), (0, 0, 9),      # long-range / repulsive (the merge-breaking lever)
)


def affinity_targets(instance):
    """instance [B,1,Z,Y,X] or [B,Z,Y,X] int -> (tgt [B,K,Z,Y,X] float, 1=same-instance;
    msk [B,K,Z,Y,X] bool = both endpoints fg = supervised edges). nnUNet ignore (-1) -> msk 0."""
    if instance.ndim == 5 and int(instance.shape[1]) == 1:
        instance = instance[:, 0]
    inst = instance.long()
    B = inst.shape[0]
    shape = tuple(int(v) for v in inst.shape[1:])
    K = len(AFFINITY_HEAD_OFFSETS)
    tgt = torch.zeros((B, K, *shape), dtype=torch.float32, device=inst.device)
    msk = torch.zeros((B, K, *shape), dtype=torch.bool, device=inst.device)
    for k, off in enumerate(AFFINITY_HEAD_OFFSETS):
        ssl_z, dsl_z = _offset_slices(shape, off)
        ssl = (slice(None), *ssl_z)
        dsl = (slice(None), *dsl_z)
        a = inst[ssl]
        b = inst[dsl]
        a_fg = (a > 0) & (a <= INSTANCE_ID_MAX)
        b_fg = (b > 0) & (b <= INSTANCE_ID_MAX)
        valid = a_fg & b_fg
        tgt[(slice(None), k, *ssl_z)] = (valid & (a == b)).float()
        msk[(slice(None), k, *ssl_z)] = valid
    return tgt, msk


class LeakFreeInstanceABBCAffinityLoss(LeakFreeInstanceABBCLoss):
    """[TIER-1] multi-task: ABBC 4-class (mask/Dice, ch 0-3) + K-channel affinity head
    (ch 4..4+K-1, per-offset same-instance BCE from the instance map). Net output = [B, 4+K, ...].
    Decoded offline by average-linkage agglomeration on the affinities. aff_w via PENGWIN_AFF_W."""

    def __init__(self, aff_w=None):
        super().__init__()
        self.aff_w = float(os.environ.get("PENGWIN_AFF_W", "1.0")) if aff_w is None else float(aff_w)
        self.K = len(AFFINITY_HEAD_OFFSETS)

    def forward(self, net_output, target):
        net_output = self._as_full_res(net_output)          # nnUNet may wrap output in a list (deep sup)
        abbc = super().forward(net_output[:, :4], target)    # ABBC 4-class on channels 0-3 (mask/Dice)
        aff_logits = net_output[:, 4:4 + self.K]
        inst = self._as_full_res(target)
        if inst.ndim == 5 and int(inst.shape[1]) == 1:
            inst = inst[:, 0]
        inst = inst.long()
        fg = (inst > 0) & (inst <= INSTANCE_ID_MAX)
        shape = tuple(int(v) for v in inst.shape[1:])
        # per-offset BCE (memory-light), accumulating same- and diff-instance edges SEPARATELY
        same_sum = aff_logits.sum() * 0.0
        diff_sum = aff_logits.sum() * 0.0
        n_same = 0
        n_diff = 0
        for k, off in enumerate(AFFINITY_HEAD_OFFSETS):
            ssl_z, dsl_z = _offset_slices(shape, off)
            ssl = (slice(None), *ssl_z)
            dsl = (slice(None), *dsl_z)
            valid = fg[ssl] & fg[dsl]
            if not bool(valid.any()):
                continue
            sm = inst[ssl] == inst[dsl]
            bce = F.binary_cross_entropy_with_logits(aff_logits[:, k][ssl], sm.float(), reduction="none")
            same_m = valid & sm
            diff_m = valid & (~sm)
            if bool(same_m.any()):
                same_sum = same_sum + bce[same_m].sum(); n_same += int(same_m.sum().item())
            if bool(diff_m.any()):
                diff_sum = diff_sum + bce[diff_m].sum(); n_diff += int(diff_m.sum().item())
        # class-BALANCED: the RARE cross-fragment (fracture) edges weigh equally to the ~95% same-instance
        # pairs, so the head can't collapse to all-same (the retired unbalanced V307 did exactly that).
        aff = 0.5 * (same_sum / max(n_same, 1) + diff_sum / max(n_diff, 1))
        total = abbc + self.aff_w * aff
        if not torch.isfinite(total):
            raise ValueError(f"LeakFreeInstanceABBCAffinityLoss non-finite: abbc={float(abbc)} aff={float(aff)}")
        return total


if __name__ == "__main__":
    _smoke_test()

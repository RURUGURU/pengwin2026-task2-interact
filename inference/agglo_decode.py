"""Tier-0 PoC: oversegment + average-linkage agglomeration decoder on V302 ABBC probs.
ABBC softmax channels: 0=bg, 1=border(outer surface), 2=boundary(inter-fragment fracture
surface), 3=core(eroded interior). Idea (GASP average-linkage / connectomics consensus):
 - oversegment the fragment foreground at EVERY boundary ridge (watershed on boundary-prob),
 - then conservatively agglomerate adjacent supervoxels whose shared interface boundary-prob
   is WEAK (avg-linkage), keeping only interfaces that look like true fracture surfaces (>= T).
This avoids the watershed MERGE (we split everywhere first) and the mutex OVER-split
(avg-linkage merges weak ridges back). Inference-only on the DEPLOYED V302 probs.
"""
import os
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
try:
    from skimage.graph import rag_boundary, merge_hierarchical
except ImportError:
    from skimage.future.graph import rag_boundary, merge_hierarchical

# canonical skimage average-linkage merge functions for rag_boundary (weighted-mean boundary)
def _weight_boundary(graph, src, dst, n):
    d = {'weight': 0.0, 'count': 0}
    cs = graph[src].get(n, d)['count']; cd = graph[dst].get(n, d)['count']
    ws = graph[src].get(n, d)['weight']; wd = graph[dst].get(n, d)['weight']
    c = cs + cd
    return {'count': c, 'weight': (cs * ws + cd * wd) / c if c else 0.0}

def _merge_boundary(graph, src, dst):
    pass

def decode_agglo(probs, T=0.45, min_vox=250, seed_core=0.5, seed_bnd=0.20, min_frac=None,
                 seed_mode=None):
    """probs: float array [4,Z,Y,X]. Returns int instance map (0=bg).
    min_frac (env PENGWIN_AGGLO_MINFRAC, default 0): reabsorb any fragment smaller than
    min_frac x the largest fragment -> kills phantom/sliver over-splits (precision guard).
    seed_mode (v3.13): "ridge" seeds at the affinity-separation ridge instead of strong-core
    interiors. Default None -> env PENGWIN_AGGLO_SEED -> "core" = **identical to v3.12**."""
    bg, border, boundary, core = probs[0], probs[1], probs[2], probs[3]
    fg = bg < 0.5
    if fg.sum() == 0:
        return np.zeros(fg.shape, np.int32)
    elev = boundary.astype(np.float32)  # high = fracture-surface ridge
    # oversegment: seed at strong-core / low-boundary interiors, watershed the boundary ridges
    # [v3.13] seed 규칙. 기본값은 v3.12 와 **완전히 동일**하다 (env 미설정 시 "core").
    #   왜 능선 seed 라는 선택지가 필요한가: decode 출력 인스턴스 수 == seed 연결성분 수다
    #   (agglomeration 은 병합만 하므로 seed 가 상한이다). 대퇴골에서는 병합 임계 T 를 바꿔도
    #   전 지표가 소수점 4자리까지 ±0.0000 이다 — 즉 seed 가 천장이고 T 축은 닫혀 있다.
    _seed_mode = seed_mode or os.environ.get("PENGWIN_AGGLO_SEED", "core")
    if _seed_mode == "ridge":
        _rt = float(os.environ.get("PENGWIN_AGGLO_SEED_RIDGE_T", "0.20"))
        # boundary 자리에는 호출부(decode_affinity_agglo)가 affinity 분리 능선 sep 을 넣어 둔다
        seeds = (boundary < _rt) & fg
    else:
        seeds = (core > seed_core) & (boundary < seed_bnd) & fg
    markers, nm = ndi.label(seeds)
    if nm <= 1:  # fallback: seed at boundary-distance maxima
        d = ndi.distance_transform_edt(boundary < seed_bnd) * fg
        from skimage.feature import peak_local_max
        pk = peak_local_max(d, min_distance=4, labels=fg)
        m = np.zeros(fg.shape, bool); m[tuple(pk.T)] = True
        markers, nm = ndi.label(m)
        if nm == 0:
            markers, nm = ndi.label(fg)
    sv = watershed(elev, markers, mask=fg)
    if sv.max() <= 1:
        out = (sv > 0).astype(np.int32)
    else:
        rag = rag_boundary(sv, elev)
        if 0 in rag.nodes:          # drop the BACKGROUND node so fg regions can't agglomerate into bg
            rag.remove_node(0)
        merged = merge_hierarchical(sv, rag, thresh=T, rag_copy=False, in_place_merge=True,
                                    merge_func=_merge_boundary, weight_func=_weight_boundary)
        out = merged.astype(np.int32) + 1   # shift: merge_hierarchical is 0-based, so no fg region collides with bg=0
    out[~fg] = 0
    out, _ = _relabel(out)
    out = _drop_small(out, min_vox)
    mf = float(os.environ.get("PENGWIN_AGGLO_MINFRAC", "0.0")) if min_frac is None else float(min_frac)
    if mf > 0.0:
        out = _drop_by_ratio(out, mf)
    return out

def _drop_by_ratio(lab, min_frac):
    """Reabsorb fragments whose voxel count < min_frac x the largest fragment (phantom/sliver guard)."""
    ids, cnts = np.unique(lab[lab > 0], return_counts=True)
    if len(cnts) <= 1:
        return lab
    small = set(ids[cnts < min_frac * int(cnts.max())].tolist())
    if not small:
        return lab
    keep = lab.copy()
    for s in small:
        keep[keep == s] = 0
    if (keep > 0).any():
        idx = ndi.distance_transform_edt(keep == 0, return_distances=False, return_indices=True)
        filled = keep[tuple(idx)]
        m = (lab > 0) & (keep == 0)
        keep[m] = filled[m]
    lab2, _ = _relabel(keep)
    return lab2

def _relabel(lab):
    ids = [i for i in np.unique(lab) if i != 0]
    remap = np.zeros(int(lab.max()) + 1, np.int32)
    for k, i in enumerate(ids, 1):
        remap[i] = k
    return remap[lab], len(ids)

def _drop_small(lab, min_vox):
    ids, cnts = np.unique(lab[lab > 0], return_counts=True)
    small = set(ids[cnts < min_vox].tolist())
    if not small:
        return lab
    keep = lab.copy()
    for s in small:
        keep[keep == s] = 0
    if (keep > 0).any() and (lab > 0).any():
        # reassign small voxels to nearest kept region
        idx = ndi.distance_transform_edt(keep == 0, return_distances=False, return_indices=True)
        filled = keep[tuple(idx)]
        m = (lab > 0) & (keep == 0)
        keep[m] = filled[m]
    lab2, _ = _relabel(keep)
    return lab2

def decode_affinity_agglo(abbc_probs, affinities, T=0.45, min_vox=250, short_idx=(0, 1, 2),
                          anatomy=None):
    """[TIER-1] Decode V307's 13ch output into instances by average-linkage agglomeration on the
    LEARNED affinities (vs the noisy ABBC boundary channel that Tier-0 used).
      abbc_probs : [4,Z,Y,X] softmax (bg,border,boundary,core)
      affinities : [K,Z,Y,X] sigmoid, same-instance prob per loss.AFFINITY_HEAD_OFFSETS
    Separation ridge = 1 - min(short-range affinity): high where any short affinity is LOW = a true
    fracture surface. We splice this affinity-separation into the boundary channel and reuse the
    validated avg-linkage decoder (oversegment at ridges -> conservatively merge weak ridges)."""
    short = np.asarray(affinities)[list(short_idx)]
    sep = 1.0 - short.min(axis=0)                 # [Z,Y,X], high = fracture surface
    probs_aff = np.asarray(abbc_probs, dtype=np.float32).copy()
    probs_aff[2] = sep.astype(np.float32)         # replace ABBC boundary with the affinity separation
    # [v3.13] seed 규칙을 **부위별로** 고른다. 기본은 전 부위 core seed = v3.12 동작 그대로.
    #   왜 대퇴골만인가 (54케 층화 실측): 능선 seed 는 femur 에 이득(hd95 −2.11mm)이고
    #   pelvic 에 손해(dice −0.0124)라 전 부위에 걸면 전역 평균에서 상쇄되고 pelvic 이 나빠진다.
    #   그리고 GC Final 160케 중 dice 최악 15케이스가 **전부 femur-only** 다.
    _sm = None
    if anatomy in ("Femur", "LeftFemur", "RightFemur"):
        _sm = os.environ.get("PENGWIN_AGGLO_SEED_FEMUR") or None
    if _sm is None:
        _sm = os.environ.get("PENGWIN_AGGLO_SEED") or None
    return decode_agglo(probs_aff, T=T, min_vox=min_vox, seed_mode=_sm)


def _affinity_subsplit(m, sep, core, T, seed_core, seed_bnd, min_vox):
    """Oversegment ONE base-instance mask `m` at affinity-separation ridges (seeded at high-core,
    low-separation interiors), then avg-linkage agglomerate supervoxels by mean interface separation,
    keeping a split only if its interface sep >= T. Returns an int label map (0 outside m).
    Conservative by construction: if <2 confident seeds are found, returns m as a single piece (no
    split) -> a base instance is broken ONLY when the affinity gives >=2 separated high-confidence
    interiors AND the interface between them survives the T gate."""
    fg = m
    seeds = (core > seed_core) & (sep < seed_bnd) & fg
    markers, nm = ndi.label(seeds)
    if nm <= 1:
        return fg.astype(np.int32)                # not enough evidence -> keep merged (precision-safe)
    sv = watershed(sep.astype(np.float32), markers, mask=fg)
    if sv.max() <= 1:
        return (sv > 0).astype(np.int32)
    rag = rag_boundary(sv, sep.astype(np.float32))
    if 0 in rag.nodes:
        rag.remove_node(0)
    merged = merge_hierarchical(sv, rag, thresh=T, rag_copy=False, in_place_merge=True,
                                merge_func=_merge_boundary, weight_func=_weight_boundary)
    out = merged.astype(np.int32) + 1
    out[~fg] = 0
    return out


def decode_fusion(base, abbc_probs, affinities, T=0.45, min_vox=250, short_idx=(0, 1, 2),
                  seed_core=0.5, seed_bnd=0.20, ridge_sep=0.5, min_ridge_vox=1500):
    """[FUSION] Refine a V302 base partition by sub-splitting each base instance ONLY where the
    LEARNED affinity (V308) confirms a true internal fracture surface. The split is CONFINED to each
    base instance's voxels -- a base boundary is NEVER crossed -> precision floor = V302; recall
    rises only where V302 merged 2 touching fragments that affinity can separate.
      base       : int instance map (0=bg) from the DEPLOYED V302 core-seed watershed decode
      abbc_probs : [4,Z,Y,X] softmax (bg,border,boundary,core) -- core[3] seeds the oversegment
      affinities : [K,Z,Y,X] sigmoid same-instance prob (loss.AFFINITY_HEAD_OFFSETS order)
    T = separation gate (higher = more conservative, closer to V302).

    REAL-FRACTURE GATE (ridge_sep / min_ridge_vox): a base instance is sub-split ONLY if it contains
    a COHERENT high-separation surface -- the LARGEST connected component of {sep >= ridge_sep} inside
    it must exceed min_ridge_vox. Diagnostic (2026-06-25) proved the V308 phantom over-splits are
    single true fragments with sep~0 everywhere (frac>0.5 ~ 0.000, only scattered noise specks) whose
    core channel SPECKLES into many blobs -> the seed-driven watershed shreds them. A true merge has
    ~10k-25k coherent high-sep voxels (294/Femur frac>0.5 = 0.126). This gate keeps the real splits
    and rejects the speckle phantoms (116/RightHip <100 high-sep voxels)."""
    base = np.asarray(base).astype(np.int32)
    abbc = np.asarray(abbc_probs, dtype=np.float32)
    core = abbc[3]
    short = np.asarray(affinities, dtype=np.float32)[list(short_idx)]
    sep = (1.0 - short.min(axis=0)).astype(np.float32)
    hi = sep >= float(ridge_sep)
    out = np.zeros_like(base)
    nxt = 1
    for lab in [int(v) for v in np.unique(base) if v != 0]:
        m = base == lab
        if int(m.sum()) < 2 * int(min_vox):       # too small to be 2 real fragments -> keep as-is
            out[m] = nxt; nxt += 1; continue
        # REAL-FRACTURE GATE: require a coherent high-sep surface (not scattered speckle noise)
        hi_m = hi & m
        if hi_m.sum() < min_ridge_vox:
            out[m] = nxt; nxt += 1; continue       # sep~0 single fragment -> keep V302 (no phantom)
        _hl, _hn = ndi.label(hi_m)
        if _hn == 0 or int(np.bincount(_hl.ravel())[1:].max()) < min_ridge_vox:
            out[m] = nxt; nxt += 1; continue       # only scattered specks, no coherent surface -> keep
        sub = _affinity_subsplit(m, sep, core, T, seed_core, seed_bnd, min_vox)
        sub = _drop_small(np.where(m, sub, 0), min_vox)
        sub_ids = [int(v) for v in np.unique(sub) if v != 0]
        if len(sub_ids) <= 1:                     # affinity found no real internal fracture -> keep V302
            out[m] = nxt; nxt += 1; continue
        for s in sub_ids:                         # adopt the affinity-confirmed split
            out[(sub == s) & m] = nxt; nxt += 1
    out, _ = _relabel(out)
    return out


if __name__ == "__main__":
    import sys, glob, json, SimpleITK as sitk
    RANGES = {'Sacrum': (1, 50), 'LeftHip': (51, 100), 'RightHip': (101, 150), 'Femur': (151, 200)}
    cases = sys.argv[1:] or ["294", "256", "271", "286", "290", "021"]
    T = float(__import__('os').environ.get("AGGLO_T", "0.45"))
    def gtcount(c):
        g = glob.glob(f'{__import__("os").environ.get("PENGWIN_ROOT", ".")}/data/task1_2/extracted/*/{c}/label.mha')
        a = sitk.GetArrayFromImage(sitk.ReadImage(g[0]))
        return {n: len([i for i in np.unique(a) if lo <= i <= hi]) for n, (lo, hi) in RANGES.items()
                if any(lo <= i <= hi for i in np.unique(a))}
    print(f"=== Tier-0 agglo decode (T={T}) vs GT fragment count (per anatomy) ===")
    for c in cases:
        gc = gtcount(c)
        probs_files = sorted(glob.glob(f'/tmp/probs/{c}/probs/probs_*.npy'))
        row = []
        for pf in probs_files:
            anat = pf.split('probs_')[-1].replace('.npy', '')
            p = np.load(pf).astype(np.float32)
            inst = decode_agglo(p, T=T)
            n = len([i for i in np.unique(inst) if i != 0])
            row.append(f"{anat}:agglo={n}/GT={gc.get(anat,'?')}")
        print(f"  case {c}: " + "  ".join(row))

"""클릭 분할의 절단면을 **학습된 affinity 능선**에 맞춘다 — Task 2 의 남은 유일한 원리적 경로.

측정된 문제
===========
2026-08-09, 로컬 GT 로 정답을 확정해 잰 결과 (30케이스 · 클릭 2개 이상인 예측 인스턴스 52건):

    정답      쪼개야 함 19 · 쪼개면 안 됨 33
    절단 방식             자식크기↔실제GT부피 상관    "자식>=1000mm³" 게이트 정확도
    거리 중선 (현 구현)          r = 0.548                    52%
    CT 뼈강도                    r = 0.022                    실패 (채택 0건)
    오라클 (실제 GT 부피)            —                        92%  채택15 옳음15 틀림0
    관측가능 최선 조합                —                        73%  채택17 옳음11 틀림6

무게이트 정확도는 **37%** 다. 이것이 과거 A/B 에서 split 이 +0.906(14배) 터진 정체다
(merge -0.252 를 얻고 MP 10.9 -> 15.3).

왜 거리 중선이 실패하는가
=========================
`task1_pipeline.py:722 apply_click_split` 은 watershed 지형으로 `distance_transform_edt(markers==0)`
을 쓴다. 그러면 물은 두 클릭의 **거리 중간**에서 만난다 — 실제 골절면과 무관하다. 그래서 자식 크기가
실제 조각 크기를 반영하지 않고(r=0.548), 크기 기반 게이트가 원리적으로 작동할 수 없다.

클릭은 GT 조각당 정확히 1개다(실측 144/144). 그런데 evaluator 는 모든 지표 계산 전에 GT·예측 양쪽에서
연결성분 1000mm³ 미만을 제거한다(`evaluate.py:178,228`) — 그리고 클릭된 조각의 **30%가 그 미만**이다.
따라서 "둘 다 채점되는 조각인가"가 분할의 정답이고, 그것은 **조각 크기**로 결정된다.
⇒ 절단면이 실제 골절면을 따라가면 자식 크기 ≈ 실제 조각 크기가 되어, evaluator 자신의 1000mm³ 를
   그대로 게이트로 쓸 수 있다.

무엇을 쓰는가 — 버려지고 있는 6채널
===================================
Stage-B 는 affinity 를 **9채널** 예측한다(`code_task1/loss.py:606`):

    0,1,2   offset (1,0,0)(0,1,0)(0,0,1)   short  (최근접)
    3,4,5   offset 3                        mid
    6,7,8   offset 9                        long   <- loss.py 원문: "the merge-breaking lever"

그런데 배포 decode 는 **0,1,2 만 읽는다**(`inference/agglo_decode.py:111` 의 `short_idx=(0,1,2)`).
9채널을 학습해 컨테이너에 싣고 6채널을 버린다. 여기서 그것을 쓴다.

    sep_short = 1 - min(aff[0:3])          # 최근접 이웃과의 비유사도 = 골절면 후보
    sep_long  = 1 - min(aff[6:9])          # offset 9 만큼 떨어져도 다른 조각 = 강한 분리 증거
    sep       = max(sep_short, sep_long)   # 둘 중 하나라도 분리를 주장하면 능선

watershed 는 낮은 곳부터 채워 **높은 곳에서 만난다**. 지형을 `sep` 로 주면 물이 능선(=골절면)에서
만난다. 거리 중선이 아니다.

🔴 이 파일이 지키는 것
======================
* **판정은 공식 evaluator 로만.** 아래 수치는 전부 로컬 GT 오라클 연구의 것이고, 승격 근거가 아니다.
* **실패해도 원본을 반환한다.** 예외·마커 유실·밴드 부족은 전부 "분할 안 함"으로 떨어진다.
* **밴드를 넘지 않는다.** 호출부가 부위별 지역 맵(`decoded_pp`)에서 부르므로 안전하지만,
  최종 밴드 맵에 직접 쓸 때를 대비해 `band` 인자를 받는다.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

SHORT = (0, 1, 2)
LONG = (6, 7, 8)


def separation_map(aff: np.ndarray, use_long: bool = True) -> np.ndarray:
    """골절면 지도. 높을수록 '여기가 조각 경계'라는 주장."""
    a = np.asarray(aff, dtype=np.float32)
    sep = 1.0 - a[list(SHORT)].min(axis=0)
    if use_long and a.shape[0] >= 9:
        sep = np.maximum(sep, 1.0 - a[list(LONG)].min(axis=0))
    return sep


def apply_click_split_affinity(labels, seeds_pp, aff, *, min_vox: int = 0, radius: int = 1,
                               use_long: bool = True, band: tuple[int, int] | None = None,
                               log=None):
    """클릭 2개 이상인 인스턴스를 **affinity 능선**을 따라 쪼갠다.

    labels   : 부위별 지역 인스턴스 맵 (pp 격자)
    seeds_pp : [(z,y,x), ...] pp 격자 클릭 좌표
    aff      : (9, Z, Y, X) sigmoid affinity — labels 와 같은 격자
    min_vox  : 자식이 이 복셀 수 미만이면 그 분할을 **되돌린다**. evaluator 의 1000mm³ 를
               pp 격자 복셀 부피로 환산해 넘긴다. 0 이면 게이트 없음(대조군).
    band     : (lo, hi) 를 주면 새 라벨을 그 범위 안에서만 할당한다.
    """
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed as _ws

    out = np.asarray(labels, dtype=np.int32).copy()
    if aff is None:
        return out.astype(np.uint16, copy=False), {"skipped": "aff 없음"}
    Z, Y, X = out.shape
    a = np.asarray(aff, dtype=np.float32)
    if a.ndim != 4 or a.shape[1:] != out.shape:
        return out.astype(np.uint16, copy=False), {"skipped": f"aff 형상 불일치 {a.shape}"}

    sep = separation_map(a, use_long=use_long)
    by_inst: dict[int, list] = defaultdict(list)
    for (z, y, x) in seeds_pp:
        if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
            L = int(out[z, y, x])
            if L > 0:
                by_inst[L].append((z, y, x))

    stats = defaultdict(int)
    if band is not None:
        lo, hi = band
        pool = [v for v in range(lo, hi + 1) if v not in set(int(t) for t in np.unique(out))]
    else:
        pool = None
    nxt = int(out.max()) + 1

    for inst, clist in by_inst.items():
        if len(clist) < 2:
            continue
        stats["후보"] += 1
        mask = out == inst
        need = len(clist) - 1
        if pool is not None:
            if len(pool) < need:
                stats["포기(밴드 만석)"] += 1
                continue
            ids = [inst] + pool[:need]
        else:
            ids = [inst] + [nxt + k for k in range(need)]

        markers = np.zeros(out.shape, dtype=np.int32)
        for i, (z, y, x) in enumerate(clist):
            sub = np.zeros(out.shape, dtype=bool)
            sub[max(z - radius, 0):z + radius + 1,
                max(y - radius, 0):y + radius + 1,
                max(x - radius, 0):x + radius + 1] = True
            markers[sub & mask] = ids[i]
        if len(set(int(v) for v in np.unique(markers)) - {0}) < len(clist):
            stats["포기(마커 유실)"] += 1
            continue

        # 🔴 여기가 이 파일의 전부다: 지형이 거리(distance_transform_edt)가 아니라 골절면(sep)이다.
        ws = _ws(sep, markers=markers, mask=mask).astype(np.int32)

        if min_vox > 0:
            sizes = [int((ws == i).sum()) for i in ids]
            if min(sizes) < min_vox:
                # 자식 하나가 채점 임계값 미만 -> GT 는 채점상 1개다. 쪼개면 split 만 는다.
                stats["게이트 기각"] += 1
                continue
        out[mask] = ws[mask]
        stats["채택"] += 1
        if pool is not None:
            pool = pool[need:]
        else:
            nxt += need
        if log:
            log(f"[click-split-aff] inst {inst} -> {len(clist)}조각")
    return out.astype(np.uint16, copy=False), dict(stats)


def min_vox_for_mm3(spacing_zyx, mm3: float = 1000.0) -> int:
    """pp 격자 spacing 에서 mm³ 임계값을 복셀 수로 환산.
    호출부에서 `predictor.configuration_manager.spacing` 를 넘긴다 — 원본 spacing 이 아니다."""
    v = float(np.prod(np.asarray(spacing_zyx, dtype=np.float64)))
    return max(1, int(round(mm3 / max(v, 1e-9))))

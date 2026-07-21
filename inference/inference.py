"""PENGWIN 2026 **Task 2 (PENGWIN-Interact)** — 컨테이너 추론 진입점.

문제 정의
=========
Task 2 는 **Task 1 과 완전히 동일한 분할 목표 + 사전 시뮬레이션된 클릭**을 추가로 받는다.
라벨/출력/평가/제출 규약은 Task 1 과 동일:
    0        = background
    1  – 50  = Sacrum
    51 – 100 = LeftHip (Left Hipbone)
    101–150  = RightHip (Right Hipbone)
    151–200  = Femur

컨테이너 I/O 계약 (Grand Challenge, `--network none`, 비root user)
==================================================================
    입력  /input/images/<slug>/<uuid>.mha          (읽기전용, CT 볼륨 — GC 이미지 인터페이스; glob으로 탐색)
    입력  /input/peripelvic-fragment-clicks.json   (읽기전용, 클릭 좌표+라벨 — JSON은 /input 직하)
    출력  /output/images/pelvic-fracture-segmentation/<입력파일명>.mha  (쓰기, 정수 라벨 0/1–200)
    모델  /opt/ml/model/                            (model.tar.gz 해제 트리, 읽기전용)
    ※ 입력 CT는 flat 경로가 아니라 GC 규약 `/input/images/<slug>/`에 들어온다 → `_resolve_input_ct`가
      glob(`/input/images/**/*.mha`)로 robust 탐색. 출력도 `/output/images/<slug>/` 아래(배포 Task1과 동일).

경로는 환경변수로 덮어쓸 수 있고(로컬 테스트), 파일명이 GC 에서 달라질 때를 대비해
robust glob 으로도 탐색한다. 로컬 실행:
    python inference.py <ct.mha> <clicks.json> [out.mha]

전략 — Task 1 파이프라인 재사용 + 클릭 주입
============================================
우리는 이미 Task 1 의 **Stage-1 해부학(Ds539)** + **Stage-2 조각(Ds538 V308 affinity)** 모델을
가지고 있다. 이 컨테이너는 그 캐스케이드(`task1_pipeline.run_per_anatomy`)를 그대로 구동하고,
클릭을 두 지점에서 주입한다:

  (a) **라우팅 확정** — 각 클릭의 `name` 이 어느 뼈(Femur/Left Hipbone/Right Hipbone/Sacrum)를
      가리키는지 직접 알려준다. 따라서 pelvic/femur family 를 **RF 라우터/Ds539 부피 규칙보다
      확실하게** 결정할 수 있다. 이 family 를 Task 1 캐스케이드의 `anatomies=` 인자로 강제
      주입하면(run_per_anatomy Layer 2b, forced_anatomies 경로), 라우팅 오류(merge 의 상류
      원인 중 하나)를 원천 제거한다. 클릭에 뼈 키워드가 전무하면(=판단 불가) family=None 을
      넘겨 Task 1 자체 라우팅(RF/Ds539)으로 안전하게 fallback 한다. → **활성 경로**.

  (b) **Stage-2 조각 seed** — 각 클릭 좌표는 특정 조각 내부의 한 점이다. 이 좌표를 Stage-2
      decode(core-seed watershed / affinity agglomeration)의 seed/prior 로 주입하면 서로 닿은
      조각(touching fragment)을 클릭 지점에서 강제로 분리해 merge 를 완화할 수 있다. 좌표계
      순서(voxel index x,y,z vs z,y,x vs world-mm)는 baseline repo 확정이 필요하므로, 본
      파일은 이를 **문서화된 훅**(`clicks_to_voxel_seeds`)으로 제공하고 디코드 주입 지점을
      명시한다. 기본 동작에는 영향이 없다(라우팅만 활성). → **문서화된 확장 훅**.

크래시 금지 원칙
================
어떤 실패(모델 없음/클릭 파싱 실패/추론 예외)에도 **최소한 CT 와 동일 geometry 의 all-zero
라벨 맵**을 반드시 출력한다. GC 는 출력이 없으면 케이스를 0 점 처리하지만, all-zero 라도
있으면 파이프라인이 완주하므로 부분 점수/디버그가 가능하다.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import traceback
from pathlib import Path

# task1_pipeline 은 같은 디렉터리(=/opt/app/inference)에 vendoring 된 Task 1 컨테이너
# 파이프라인의 사본이다. 컨테이너 밖(로컬)에서도 import 되도록 이 파일 위치를 sys.path 에
# 넣는다. 무거운 import(numpy/SimpleITK)는 task1_pipeline 모듈 top-level 에서 이미 수행된다.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ---------------------------------------------------------------------------
# 컨테이너 I/O 경로 (환경변수로 override 가능; 컨테이너 기본값은 flat /input,/output).
# ---------------------------------------------------------------------------
DEFAULT_INPUT_CT = os.environ.get("PENGWIN_INPUT_CT", "/input/pelvic-fracture-ct.mha")
DEFAULT_INPUT_CLICKS = os.environ.get(
    "PENGWIN_INPUT_CLICKS", "/input/peripelvic-fragment-clicks.json"
)
# ⚠️ GC 규약: 세그멘테이션(이미지) 출력은 `/output/images/<interface-slug>/<파일명>.mha` 아래에 둔다
#    (배포 Task1 컨테이너와 동일 규약; flat `/output/*.mha`면 GC가 결과를 못 찾는다). 파일명은 입력
#    CT 파일명을 그대로 쓴다(Task1 검증된 관례). slug = task2 스펙의 출력 인터페이스 "pelvic-fracture-segmentation".
DEFAULT_OUTPUT_DIR = os.environ.get(
    "PENGWIN_OUTPUT_DIR", "/output/images/pelvic-fracture-segmentation"
)

# 출력 slug 후보 (2026-07-21 추가).
#
# 문제: Task 2 의 출력 인터페이스 slug 가 문서상 확정되지 않았다. docs/challenge/02-*.md:12 은
# "pelvic-fracture-segmentation" 이라 적고 있으나, **같은 스타일의 문서 줄이 Task 1 에서는 틀린 전례가
# 있다** — 01-*.md:9 은 "peripelvic-fracture-segmentation" 이라 하지만 실제 GC 채점된 배포 컨테이너는
# "peripelvic-fracture-ct-segmentation" 을 쓴다(01-*.md:83 과 일치). PENGWIN-2024 공식 템플릿도
# 입력 pelvic-fracture-ct ↔ 출력 pelvic-fracture-**ct**-segmentation 로 짝지어져 있다.
# slug 가 틀리면 GC 는 산출물을 임포트하지 못하고 해당 런은 실패 처리된다.
#
# 대응: (1) /input/inputs.json 이 있으면 그것이 런타임 권위 소스이므로 거기서 읽는다.
#       (2) 없으면 후보 slug 전부에 동일한 .mha 를 쓴다. GC 는 **선언된 소켓만** 임포트하고 나머지
#           디렉터리는 무시하므로 부작용이 없다(디스크 수십 MB 뿐).
OUTPUT_SLUG_CANDIDATES = (
    "pelvic-fracture-segmentation",
    "pelvic-fracture-ct-segmentation",
    "peripelvic-fracture-ct-segmentation",
    "peripelvic-fracture-segmentation",
)

# 클릭 point `name` 의 뼈 키워드 → family. (Femur 만 femur, 나머지 3뼈는 pelvic.)
_FEMUR_KEYWORDS = ("femur",)
_PELVIC_KEYWORDS = ("hip", "ilium", "sacrum", "pelvi")  # 실데이터 "Left Hip"/"Right Hip" 포함


def log(msg: str) -> None:
    """Task 1 파이프라인 로그와 동일 prefix 로 stdout 에 남긴다(GC 로그에서 grep 용이)."""
    print(f"[pengwin_task2] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 클릭 파싱 + family 라우팅 (code_task2/inference.py 로직 이식).
# ---------------------------------------------------------------------------
def load_clicks(clicks_path):
    """clicks.json 을 파싱해 point 리스트로 정규화한다.

    스펙 예시는 최상위 dict 하나(`name`=전략, `points`=[{name, point}...])지만, 파일이 여러
    전략 dict 의 리스트로 올 수도 있어 둘 다 처리한다. point 좌표는 **그대로** 보관한다
    (순서 미변환 — 좌표계 확정은 seed 훅에서 처리).

    Returns: [{"name": <point name>, "point": [a,b,c], "strategy": <top-level name>}, ...]
             파일이 없거나 파싱 실패 시 빈 리스트(크래시 금지 — 상위에서 라우팅 fallback).
    """
    try:
        with open(clicks_path, "r", errors="ignore") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        log(f"load_clicks: '{clicks_path}' 읽기/파싱 실패 ({exc}) → 클릭 없이 진행")
        return []
    blocks = data if isinstance(data, list) else [data]
    pts = []
    for blk in blocks:
        if not isinstance(blk, dict):
            continue
        strategy = blk.get("name", "")
        for p in blk.get("points", []) or []:
            if isinstance(p, dict) and "point" in p:
                pts.append(
                    {
                        "name": p.get("name", ""),
                        "point": p["point"],
                        "strategy": strategy,
                    }
                )
    return pts


def _bone_family(point_name):
    """point `name`("Femur ... Point 1")에서 family 를 뽑는다. 'femur' | 'pelvic' | None."""
    nl = str(point_name).lower()
    if any(k in nl for k in _FEMUR_KEYWORDS):
        return "femur"
    if any(k in nl for k in _PELVIC_KEYWORDS):
        return "pelvic"
    return None


def route_from_clicks(points):
    """클릭만으로 scan family 를 결정한다(가장 확실한 신호).

    클릭 name 이 뼈를 직접 명시하므로: Femur 클릭이 하나라도 있으면 femur, 아니면 pelvic 뼈
    클릭이 있으면 pelvic. 판단 불가(뼈 키워드 전무)면 family=None → 상위에서 Task 1 자체
    라우팅으로 fallback.

    또한 어떤 pelvic 뼈가 실제로 클릭됐는지(Sacrum/LeftHip/RightHip)도 집계해, pelvic 케이스의
    **anatomy subset** 을 클릭 근거로 좁힐 수 있게 한다(present bone 만 처리 → phantom 뼈로 인한
    FP 를 줄이는 상류 개선; 클릭이 특정 뼈만 지목하면 그 뼈만 강제).

    Returns dict:
        family        : "femur" | "pelvic" | None
        femur_clicks  : int
        pelvic_clicks : int
        pelvic_bones  : set{"Sacrum","LeftHip","RightHip"}   (클릭으로 확인된 pelvic 뼈들)
    """
    fam_counts = {"femur": 0, "pelvic": 0}
    pelvic_bones = set()
    for p in points:
        nm = str(p.get("name", "")).lower()
        fam = _bone_family(nm)
        if fam:
            fam_counts[fam] += 1
        # pelvic 뼈 세부 식별 (anatomy subset 근거).
        if "sacrum" in nm:
            pelvic_bones.add("Sacrum")
        elif "left" in nm and ("hip" in nm or "ilium" in nm):
            pelvic_bones.add("LeftHip")
        elif "right" in nm and ("hip" in nm or "ilium" in nm):
            pelvic_bones.add("RightHip")
    if fam_counts["femur"] > 0:
        family = "femur"
    elif fam_counts["pelvic"] > 0:
        family = "pelvic"
    else:
        family = None
    return {
        "family": family,
        "femur_clicks": fam_counts["femur"],
        "pelvic_clicks": fam_counts["pelvic"],
        "pelvic_bones": pelvic_bones,
    }


def anatomies_from_routing(routing):
    """클릭 라우팅 결과 → Task 1 캐스케이드에 강제 주입할 `anatomies` 튜플.

    - femur : ("Femur",)
    - pelvic: 클릭으로 확인된 pelvic 뼈들만(정렬). 하나도 세부식별 못하면 pelvic 3뼈 전체.
    - None  : None  → Task 1 자체 라우팅(RF/Ds539)으로 fallback.

    반환 튜플의 순서는 Task 1 의 ALL_ANATOMIES 순서(Sacrum,LeftHip,RightHip,Femur)를 따른다.
    """
    fam = routing.get("family")
    if fam == "femur":
        return ("Femur",)
    if fam == "pelvic":
        bones = routing.get("pelvic_bones") or set()
        order = ("Sacrum", "LeftHip", "RightHip")
        subset = tuple(a for a in order if a in bones)
        # 세부식별 실패(전략 name 이 "Left Hipbone" 형식이 아닐 때) → pelvic 3뼈 전체 처리.
        return subset if subset else order
    return None


# ---------------------------------------------------------------------------
# [문서화된 훅] Stage-2 조각 seed 주입.
# ---------------------------------------------------------------------------
def clicks_to_voxel_seeds(points, ref_img):
    """[Stage-2 seed 훅] 각 클릭 좌표를 CT 배열(numpy z,y,x)의 voxel index 로 변환한다.

    ✅ **좌표계 순서 확정(2026-07-12): `point` = numpy index (z, y, x).** 실데이터로 검증 —
    클릭 `point`를 학습셋 `label.mha`에 대입해 클릭 name의 뼈 라벨 범위(Sacrum 1-50 등)에 맞는지
    측정: **(z,y,x) 해석이 62/62(100%) 적중, (x,y,z)는 13/62(21%)**. 따라서 `_ORDER` 기본값 = "zyx"
    (뒤집기 없음). env `PENGWIN_CLICK_ORDER`로 재정의 가능(xyz/zyx/world). 범위는 볼륨 shape로 clip.

    ── 주입 지점(문서) ─────────────────────────────────────────────────────────────
    반환된 seed 는 `task1_pipeline.run_per_anatomy` 의 Stage-2 decode 직전에 주입한다.
    구체적으로 anatomy 별 bbox(로컬 crop) 좌표계로 옮긴 뒤:
        • core-seed watershed 경로: `decode_abbc_core_seed_watershed(...)` 의 core-seed 라벨에
          클릭 위치를 강제 seed(서로 다른 클릭 = 서로 다른 라벨)로 추가 → 닿은 조각 분리.
        • affinity agglomeration 경로: `decode_affinity_agglo(...)` 의 초기 fragment 라벨에
          클릭을 must-link/cannot-link 제약(같은 조각 클릭끼리 must-link, 다른 조각 cannot-link)
          으로 반영.
    본 컨테이너의 기본 경로는 (a) 라우팅만 활성이며, seed 주입은 좌표계 확정 후 활성화한다.

    Returns: [{"name","family","zyx":(z,y,x)}...]  — ref_img 범위로 clip 된 voxel index.
             변환 불가/좌표 이상 시 그 point 는 생략(크래시 금지).
    """
    import numpy as np  # 지연 import (라우팅만 필요한 경로에서 numpy 강제 로드 회피).

    # ref_img 는 원본 CT (LPS 정규화 이전). numpy 배열 shape 는 (z, y, x).
    try:
        size_xyz = ref_img.GetSize()  # SimpleITK 순서 (X, Y, Z)
        shape_zyx = (int(size_xyz[2]), int(size_xyz[1]), int(size_xyz[0]))
    except Exception as exc:  # noqa: BLE001
        log(f"clicks_to_voxel_seeds: ref geometry 조회 실패 ({exc}) → seed 없음")
        return []

    _ORDER = os.environ.get("PENGWIN_CLICK_ORDER", "zyx").strip().lower()  # 실측 확정: 클릭=(z,y,x)
    seeds = []
    for p in points:
        pt = p.get("point")
        if not (isinstance(pt, (list, tuple)) and len(pt) == 3):
            continue
        try:
            a, b, c = (int(round(float(v))) for v in pt)
        except (TypeError, ValueError):
            continue
        if _ORDER == "zyx":
            z, y, x = a, b, c
        elif _ORDER == "world":
            # world-mm → index (x,y,z) → (z,y,x). ref_img 의 physical→index 변환 사용.
            try:
                ix = ref_img.TransformPhysicalPointToIndex((float(a), float(b), float(c)))
                x, y, z = int(ix[0]), int(ix[1]), int(ix[2])
            except Exception:  # noqa: BLE001
                continue
        else:  # 기본 "xyz" index → numpy (z,y,x)
            x, y, z = a, b, c
        # 볼륨 범위로 clip (경계 밖 클릭 방지).
        z = int(np.clip(z, 0, shape_zyx[0] - 1))
        y = int(np.clip(y, 0, shape_zyx[1] - 1))
        x = int(np.clip(x, 0, shape_zyx[2] - 1))
        seeds.append(
            {
                "name": p.get("name", ""),
                "family": _bone_family(p.get("name", "")),
                "zyx": (z, y, x),
            }
        )
    return seeds


# ---------------------------------------------------------------------------
# 경로 resolve 헬퍼 (flat Task2 경로 + GC nested + glob fallback).
# ---------------------------------------------------------------------------
def _resolve_input_ct(explicit=None):
    """CT 입력 경로를 찾는다. 우선순위: 명시 인자 → 기본 flat → /input 하위 재귀 *.mha glob."""
    if explicit and os.path.exists(explicit):
        return explicit
    if os.path.exists(DEFAULT_INPUT_CT):
        return DEFAULT_INPUT_CT
    for pat in (
        "/input/*.mha",
        "/input/*.mhd",
        "/input/images/**/*.mha",
        "/input/**/*.mha",
    ):
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return explicit or DEFAULT_INPUT_CT


def _resolve_input_clicks(explicit=None):
    """clicks.json 경로를 찾는다. 우선순위: 명시 인자 → 기본 flat → /input 하위 *click*.json."""
    if explicit and os.path.exists(explicit):
        return explicit
    if os.path.exists(DEFAULT_INPUT_CLICKS):
        return DEFAULT_INPUT_CLICKS
    for pat in (
        "/input/*click*.json",
        "/input/**/*click*.json",
        "/input/*.json",
        "/input/**/*.json",
    ):
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return explicit or DEFAULT_INPUT_CLICKS


def _resolve_output_seg(ct_path, explicit=None):
    """출력 경로 = GC 이미지 인터페이스 규약 `/output/images/<slug>/<입력CT파일명>` + 부모 디렉터리 생성.

    로컬 테스트로 `explicit`(파일경로)를 주면 그걸 그대로 쓴다. 컨테이너(GC)에선 입력 CT 파일명을
    그대로 이어받아 `DEFAULT_OUTPUT_DIR` 아래에 둔다(Task1 배포 컨테이너와 동일).
    """
    if explicit:
        os.makedirs(os.path.dirname(explicit) or ".", exist_ok=True)
        return explicit
    out_dir = DEFAULT_OUTPUT_DIR
    if not os.environ.get("PENGWIN_OUTPUT_DIR"):
        # 런타임 권위 소스가 있으면 그것을 최우선으로 쓴다(헤지보다 정확하다).
        slug = _slug_from_inputs_json()
        if slug:
            out_dir = os.path.join("/output/images", slug)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, os.path.basename(str(ct_path)))


def _slug_from_inputs_json():
    """`/input/inputs.json` 에서 출력 이미지 소켓 slug 를 읽는다(런타임 권위 소스). 없으면 None.

    GC 는 각 job 의 `/input/inputs.json` 에 선언된 인터페이스 목록을 넣어준다. 이름 키는 GC 버전에
    따라 `slug` / `interface.slug` 등으로 나타나므로 재귀적으로 훑어 후보와 매칭한다. 실패는 전부
    무시하고 None 을 돌려준다 — 이 함수는 어떤 경우에도 추론을 막으면 안 된다.
    """
    path = os.environ.get("PENGWIN_INPUTS_JSON", "/input/inputs.json")
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        log(f"inputs.json 읽기 실패(무시): {exc}")
        return None

    found = []

    def walk(node):
        if isinstance(node, dict):
            for key, val in node.items():
                if key in ("slug", "interface_slug") and isinstance(val, str):
                    found.append(val)
                walk(val)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(blob)
    for slug in found:
        if "segmentation" in slug:
            log(f"inputs.json 에서 출력 slug 확인: {slug}")
            return slug
    log(f"inputs.json 에 segmentation slug 없음 (본 slug: {sorted(set(found))})")
    return None


def _mirror_output(out_path):
    """이미 기록된 산출물을 나머지 후보 slug 디렉터리로 복사한다(slug 불확실성 헤지).

    `explicit`/`PENGWIN_OUTPUT_DIR` 로 경로를 직접 지정한 로컬 테스트에서는 아무것도 하지 않는다.
    실패는 로그만 남기고 삼킨다 — 헤지가 본 산출물을 위험에 빠뜨리면 안 된다.
    """
    import shutil

    if os.environ.get("PENGWIN_OUTPUT_DIR"):
        return
    out_path = str(out_path)
    parent = os.path.dirname(out_path)
    if os.path.basename(os.path.dirname(parent)) != "images":
        return  # GC 규약 경로가 아님(로컬 테스트) → 헤지 안 함
    images_root = os.path.dirname(parent)
    name = os.path.basename(out_path)
    for slug in OUTPUT_SLUG_CANDIDATES:
        target_dir = os.path.join(images_root, slug)
        target = os.path.join(target_dir, name)
        if os.path.abspath(target) == os.path.abspath(out_path):
            continue
        try:
            os.makedirs(target_dir, exist_ok=True)
            shutil.copyfile(out_path, target)
            log(f"slug 헤지 복사 → {target}")
        except Exception as exc:  # noqa: BLE001
            log(f"slug 헤지 복사 실패(무시) {target}: {exc}")


def _write_zero_seg(ref_img, out_path, reason):
    """CT 와 동일 geometry 의 all-zero uint8 라벨 맵을 저장한다(최후 안전망)."""
    import numpy as np
    import SimpleITK as sitk

    arr = np.zeros(sitk.GetArrayFromImage(ref_img).shape, dtype=np.uint8)
    zero = sitk.GetImageFromArray(arr)
    zero.SetSpacing(ref_img.GetSpacing())
    zero.SetOrigin(ref_img.GetOrigin())
    zero.SetDirection(ref_img.GetDirection())
    sitk.WriteImage(zero, str(out_path), useCompression=False)
    log(f"ALL-ZERO 세그 저장 ({reason}) → {out_path}")
    _mirror_output(out_path)


# ---------------------------------------------------------------------------
# Task 1 캐스케이드 실행 (클릭 라우팅 주입).
# ---------------------------------------------------------------------------
def run_task1_cascade(ct_image, image_path, points, routing):
    """[통합] 클릭-라우팅을 주입해 배포 Task-1 캐스케이드를 구동, 분할 라벨을 만든다.

    Stage-1 anatomy(Ds539) → 라우팅(**클릭으로 확정**) → Stage-2 조각(Ds538) → PENGWIN remap.
    실제 세그멘테이션 로직은 vendoring 된 `task1_pipeline.run_per_anatomy` 를 그대로 사용한다
    (단일 소스 재사용 — 로직 중복 없음). 모델 가중치는 `/opt/ml/model` tarball 에서 로드된다
    (task1_pipeline 이 nnUNet_results 등 env 로 경로를 잡는다).

    Args:
        ct_image  : 원본 CT SimpleITK 이미지 (ref geometry).
        image_path: CT 파일 경로(Path) — Task1 RF 라우터 fallback 에 필요.
        points    : load_clicks() 결과.
        routing   : route_from_clicks() 결과 (family + pelvic_bones).

    Returns: CT 와 동일 geometry 의 정수 라벨 numpy 배열 (z,y,x; 0/1–200).
    """
    import task1_pipeline as t1

    forced = anatomies_from_routing(routing)  # 클릭 → 강제 anatomies (None 이면 자동 라우팅)
    log(
        f"클릭 라우팅 → family={routing.get('family')} forced_anatomies={forced} "
        f"(femur={routing.get('femur_clicks')}, pelvic={routing.get('pelvic_clicks')}, "
        f"pelvic_bones={sorted(routing.get('pelvic_bones') or [])})"
    )

    # [Stage-2 seed 훅] 좌표계 확정 후 여기서 생성한 seed 를 run_per_anatomy 의 decode 에
    # 주입한다(현재는 문서/디버그용 — 라우팅만 활성). 실패해도 무해(빈 리스트).
    try:
        seeds = clicks_to_voxel_seeds(points, ct_image)
        log(f"[seed 훅] 클릭 seed {len(seeds)}개 준비(주입은 좌표계 확정 후 활성화)")
    except Exception as exc:  # noqa: BLE001
        log(f"[seed 훅] seed 변환 실패 ({exc}) — 무시하고 라우팅만으로 진행")

    # pelvic ROI fallback 용 bone-skeleton 분해(Task1 main() 과 동일). femur 케이스엔 비어 무시됨.
    prerouted_bone_masks = None
    try:
        img_lps, arr_clipped = t1.canonicalize_and_clip_image(ct_image)
        sp_xyz = img_lps.GetSpacing()
        sp_zyx = (float(sp_xyz[2]), float(sp_xyz[1]), float(sp_xyz[0]))
        prerouted_bone_masks = t1.bone_skeleton_anatomy_decomposition(
            arr_clipped,
            sp_zyx,
            hu_threshold=t1.BONE_HU_THRESHOLD,
            min_component_voxels=t1.BONE_MIN_COMPONENT_VOXELS,
        )
    except Exception as exc:  # noqa: BLE001
        log(f"bone-skeleton preroute 실패 ({exc}) — fallback mask 없이 진행")

    # 핵심 호출: forced anatomies(클릭)로 Task1 캐스케이드 구동.
    label_arr = t1.run_per_anatomy(
        Path(image_path), ct_image, prerouted_bone_masks, anatomies=forced
    )
    return label_arr


# ---------------------------------------------------------------------------
# 진입점.
# ---------------------------------------------------------------------------
def run(input_ct=None, input_clicks=None, output_seg=None):
    """컨테이너 진입: CT+clicks 읽기 → 클릭 라우팅 → Task-1 캐스케이드 → 분할 쓰기.

    어떤 실패에도 all-zero 라도 반드시 출력한다(크래시 금지).
    """
    import SimpleITK as sitk

    ct_path = _resolve_input_ct(input_ct)
    clicks_path = _resolve_input_clicks(input_clicks)
    out_path = _resolve_output_seg(ct_path, output_seg)
    log(f"start: ct={ct_path} clicks={clicks_path} out={out_path}")

    # CT 는 반드시 필요(ref geometry). 못 읽으면 출력 자체가 불가 → 에러 종료.
    ref_img = sitk.ReadImage(ct_path)

    try:
        points = load_clicks(clicks_path)
        routing = route_from_clicks(points)
        log(
            f"클릭 {len(points)}개 파싱 → family={routing['family']} "
            f"(femur={routing['femur_clicks']}, pelvic={routing['pelvic_clicks']})"
        )

        label_arr = run_task1_cascade(ref_img, ct_path, points, routing)

        # Task1 파이프라인과 동일 규약으로 저장(값 0..200 clip → uint8, geometry 유지).
        import numpy as np

        label_arr = np.clip(np.asarray(label_arr), 0, 200).astype(np.uint8, copy=False)
        out_img = sitk.GetImageFromArray(label_arr)
        out_img.SetSpacing(ref_img.GetSpacing())
        out_img.SetOrigin(ref_img.GetOrigin())
        out_img.SetDirection(ref_img.GetDirection())
        sitk.WriteImage(out_img, str(out_path), useCompression=False)
        uniq = np.unique(label_arr)
        log(
            f"분할 저장 완료 → {out_path} "
            f"(shape={label_arr.shape}, n_labels={len(uniq)}, head={uniq[:10].tolist()})"
        )
        # 자기진단: 전부 배경이면 파이프라인이 조용히 죽은 것이다. GC 는 exit 0 만 보고 GREEN 으로
        # 기록하므로 로그에 크게 남겨야 사후에 잡을 수 있다(예: DS538_FOLD 불일치로 가중치 로드 실패).
        if len(uniq) <= 1:
            log(
                "!!! 경고: 산출물이 전부 배경(라벨 1종)이다. 정상 결과가 아니다. "
                "PENGWIN_DS538_FOLD 와 model tarball 의 fold 디렉터리 일치 여부, "
                "그리고 Stage-1/Stage-2 가중치 로드 로그(w0sum)를 확인하라."
            )
        _mirror_output(out_path)
        return out_path
    except Exception as exc:  # noqa: BLE001
        # 어떤 예외에도 all-zero 세그를 출력해 컨테이너가 완주하도록 한다.
        log(f"FATAL: 추론 실패 ({exc})")
        traceback.print_exc()
        try:
            _write_zero_seg(ref_img, out_path, f"예외 fallback: {exc}")
        except Exception as fexc:  # noqa: BLE001
            log(f"all-zero fallback 마저 실패: {fexc}")
            traceback.print_exc()
            raise
        return out_path


def main() -> int:
    import time

    t0 = time.time()
    args = sys.argv[1:]
    try:
        run(
            args[0] if len(args) >= 1 else None,
            args[1] if len(args) >= 2 else None,
            args[2] if len(args) >= 3 else None,
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        # run() 이 CT 조차 못 읽은 극단 케이스: 여기서 마지막으로 zero 시도.
        log(f"main FATAL: {exc}")
        traceback.print_exc()
        try:
            import SimpleITK as sitk

            ct_path = _resolve_input_ct(args[0] if len(args) >= 1 else None)
            # NOTE(2026-07-21): the single positional used to bind to `ct_path`, not `explicit`
            # (signature is _resolve_output_seg(ct_path, explicit=None)). On GC there is no argv, so
            # ct_path became None and the fallback wrote a file literally named "None" with no
            # extension, which GC cannot import -> the last-resort safety net produced an
            # unreadable output instead of a valid all-zero mask. Pass both positions explicitly.
            out_path = _resolve_output_seg(
                ct_path, args[2] if len(args) >= 3 else None
            )
            ref_img = sitk.ReadImage(ct_path)
            _write_zero_seg(ref_img, out_path, f"main fallback: {exc}")
            return 0
        except Exception as fexc:  # noqa: BLE001
            log(f"main fallback 실패: {fexc}")
            return 1
    finally:
        log(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())

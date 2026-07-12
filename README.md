# PENGWIN 2026 — Task 2 (PENGWIN-Interact) 추론 컨테이너

**클릭 기반 대화형 골반 골절 조각 분할** 컨테이너. Task 1 (자동 분할) 파이프라인을 **그대로
재사용**하고, 추가 입력인 클릭(`clicks.json`)을 라우팅/시드로 주입한다.

> Task 2 = **Task 1 + 클릭.** 분할 목표·라벨(0 bg / 1–50 Sacrum / 51–100 LeftHip /
> 101–150 RightHip / 151–200 Femur)·출력 포맷·평가·제출은 Task 1 과 완전히 동일하다.

---

## 1. 컨테이너 I/O 계약 (Grand Challenge)

| 종류 | 경로 | 설명 |
|------|------|------|
| 입력 | `/input/pelvic-fracture-ct.mha` | CT 볼륨 (읽기전용) |
| 입력 | `/input/peripelvic-fragment-clicks.json` | 클릭 좌표 + 뼈 라벨 (읽기전용) |
| 출력 | `/output/pelvic-fracture-segmentation.mha` | 정수 라벨 맵 0/1–200 (쓰기) |
| 모델 | `/opt/ml/model/` | `model.tar.gz` 해제 트리 (읽기전용) |

- 실행: `--network none`, **비root** user, 추가 인자 없음.
- 파일명/디렉터리가 GC 에서 다르게 오더라도 `/input` 하위 재귀 glob 으로 CT(`*.mha`)와
  클릭(`*click*.json`)을 robust 하게 탐색한다.
- 경로는 환경변수(`PENGWIN_INPUT_CT` / `PENGWIN_INPUT_CLICKS` / `PENGWIN_OUTPUT_SEG`)로 override.

### 클릭 JSON 포맷 (공식 스펙)
최상위 `name` = 전략, `points[*].name` = **어느 뼈를 클릭했는지**(Femur / Left Hipbone /
Right Hipbone / Sacrum), `points[*].point` = `[a, b, c]` 좌표.
```json
{
  "name": "Uniformly Sampled Points of Interest",
  "type": "Multiple Points",
  "points": [
    { "name": "Femur Uniformly Sampled Point 1", "point": [62, 307, 135] }
  ],
  "version": { "major": 1, "minor": 0 }
}
```
전략은 4종: Uniformly Sampled / Euclidean Distance Transform / Center of Mass /
Boundary Internal Margin (CT 1개 → 4 케이스). 파서는 단일 dict 와 dict 리스트를 모두 처리한다.

---

## 2. 파이프라인 (클릭 주입 지점)

```
CT ─▶ Stage-1 해부학(Ds539, 5-class) ─▶ 라우팅 ─▶ Stage-2 조각(Ds538 V308 affinity) ─▶ PENGWIN remap ─▶ seg.mha
                                          ▲                    ▲
                                   (a) 클릭 라우팅        (b) 클릭 seed 훅
                                       [활성]                [문서화]
```

- **(a) 라우팅 확정 — 활성.** 각 클릭 `name` 이 pelvic/femur 를 직접 알려준다. 이 family 를
  `task1_pipeline.run_per_anatomy(..., anatomies=<forced>)` 로 강제 주입해 RF 라우터/Ds539
  부피 규칙을 대체한다(라우팅 오류 = merge 상류 원인 제거). pelvic 케이스는 클릭으로 확인된
  뼈(Sacrum/LeftHip/RightHip) subset 만 처리해 phantom 뼈 FP 도 줄인다. 클릭에 뼈 키워드가
  없으면 `family=None` → Task 1 자체 라우팅으로 안전 fallback.
- **(b) Stage-2 조각 seed — 문서화된 훅.** `clicks_to_voxel_seeds()` 가 각 클릭을 CT voxel
  index(z,y,x)로 변환한다. 이 seed 를 decode(core-seed watershed / affinity agglomeration)에
  주입하면 닿은 조각을 클릭 지점에서 분리(merge 완화)할 수 있다. **좌표계 순서**(index x,y,z vs
  z,y,x vs world-mm)는 baseline repo(`PENGWIN2026_Task2_InteractiveSeg_Baseline`)에서 확정 후
  활성화한다(`PENGWIN_CLICK_ORDER` env: `xyz`(기본)/`zyx`/`world`). 현재 기본 경로는 (a)만 활성.

**모델 공유:** Task 1 과 **동일한 `model.tar.gz`** 를 재사용한다(Stage-1 V301 fold_0 +
Stage-2 V308 fold_all + affinity agglomeration T=0.45). Task 2 를 위한 재학습은 없다.

**크래시 금지:** 모델 부재/클릭 파싱 실패/추론 예외 등 **어떤 실패에도** CT 와 동일 geometry 의
all-zero 라벨 맵을 반드시 출력한다.

---

## 3. 빌드 & 로컬 테스트

```bash
# 이미지 빌드 (빌드 컨텍스트 = 이 repo 루트)
IMAGE_TAG=pengwin-task2-interact:latest ./scripts/build_image.sh

# 로컬 스모크 테스트 (실제 모델 없이 클릭 파싱 + I/O + 라우팅까지만 확인하려면
# 모델 tarball 없이도 all-zero fallback 으로 완주한다):
python inference/inference.py <ct.mha> <clicks.json> [out.mha]
```

GC 실행 재현:
```bash
docker run --rm --gpus all --network none \
  -v $PWD/input:/input:ro \
  -v $PWD/output:/output \
  -v $PWD/model_payload:/opt/ml/model:ro \
  pengwin-task2-interact:latest
```

---

## 4. 제출 (Grand Challenge)

1. `./scripts/build_image.sh` 로 이미지 빌드 → GC Algorithm 에 컨테이너 업로드(또는 tar save).
2. **모델 tarball 은 이미지에 포함하지 않는다.** Task 1 과 동일한 `model.tar.gz` 를 GC Models
   탭에 올리면 런타임에 `/opt/ml/model/` 로 해제된다.
3. GC 는 `--network none`, 비root 로 컨테이너를 돌린다. 출력은
   `/output/pelvic-fracture-segmentation.mha`.

평가/랭킹은 Task 1 과 동일 → `../../docs/challenge/04-evaluation-and-ranking.md`.

---

## 5. 디렉터리

```
submission_task2/github_repo/
├── inference/
│   ├── inference.py             # Task 2 entrypoint (클릭 파싱 + 라우팅 주입 + seed 훅)
│   ├── task1_pipeline.py        # vendored Task 1 캐스케이드(단일 소스 사본)
│   ├── agglo_decode.py          # affinity agglomeration decoder
│   ├── target_family_router.py  # RF family 라우터(클릭 모호 시 fallback)
│   ├── pengwin_trainers_shim.py # nnUNet trainer-discovery shim
│   └── __init__.py
├── code_task1/                  # trainer 정의 + helper(shim 소스; 단일 소스)
├── Dockerfile                   # Task 1 미러 + Task 2 entrypoint
├── requirements.txt             # Task 1 과 동일
├── scripts/build_image.sh
├── .gitignore                   # 모델/영상/로그 제외
├── LICENSE                      # MIT
└── README.md
```

> `.pth/.tar.gz/.mha/.npy` 등 가중치·영상·로그는 `.gitignore` 로 커밋 제외한다(GitHub 100MB 한도).

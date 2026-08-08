# Task 2 v3.4 — always-expert Stage-B + T=0.75 (v3.1 배포 경로 유지)

## 왜

Task 2 출력은 Task 1 출력과 사실상 동일하다 (2026-08-08 Final Test 보드 실측):

| 지표 | Task 1 | Task 2 |
|---|---|---|
| Fracture Dice | 0.886 | 0.886 |
| Local Dice | 0.873 | 0.873 |
| HD95 | 11.215 | 11.214 |

클릭을 seed 로 쓰지 않기 때문이다(`PENGWIN_CLICK_INJECT=0`).
⇒ **Task 1 Stage-B 개선이 Task 2 로 그대로 상속된다.**

같은 팀 계정이 Task 1 에서 always-expert + `AGGLO_T=0.75` 로 MP **14.6**,
우리 unified + 0.45 구성이 **17.6** 이었다. 그 구성을 Task 2 사다리에 얹으면:

```
Split      0.150(16위) → 0.063(4위)    +12계단
Topology   0.746(15위) → 0.819(6위)    +9계단
나머지 8지표             1~6계단씩 하락 (F1 −6 이 최대)
Mean Position  9.5 → 8.6   (Δ −0.9)
```

팀 순위는 4위 유지(3위 문턱 4.6)지만 순이득이라 채택한다.

## 무엇이 v3.3 과 다른가

1. `code_task1/core.py` — Stage-B 전문가 로딩용 **별칭 클래스 4개**.
   없으면 `pengwin_trainers_shim` 이 이름을 re-export 못 해 trainer discovery 가 실패한다.
2. `inference/task1_pipeline.py` — `DS538_EXPERT_TRAINERS` 선언 + 부위별 **지연 로딩**.
   env 가 비면 이전과 바이트 동일. **한 번에 하나만 상주**시켜 v3.1~v3.3 메모리 작업을 지킨다.
3. `Dockerfile` — 런타임 ENV. `AGGLO_T` 0.45→**0.75**, 부위별 expert, `RF_CONF_MARGIN=0.15`.

**`PENGWIN_CLICK_INJECT=0` 은 그대로 유지한다** — v3.3 의 클릭 seed 주입은 val 에서
spurious over-split 으로 기각됐고, 2026-08-08 재검증에서도 클릭 전략에 따라 부호가 뒤집혔다
(`center_of_mass` −1.7 / `uniformly_sampled` +1.9). GC 는 전략을 알려주지 않는다.

## 🔴 업로드 시

`model.tar.gz` 는 반드시 팀원 번들:
```
submission_task1/teammate_v35_package/model.tar.gz
1,409,476,486 B
sha256 049c38ea4abf1629a4d5f79a68a27918fd4103941fbf4f500b76211e93192919
```
expert 체크포인트가 없는 번들로는 로드에 실패한다. Task 1 v3.11 과 **같은 번들**을 쓴다.

빌드 로그에서 확인: `w0sum=1.0718e+02` · `n_features=37` ·
`[Sacrum] Stage-B trainer -> ...SacrumExpert...`

## 검증 상태

- shim 이 4개 이름 전부 re-export (실제 import, `PengwinTrainer*` 8개)
- `torch` 가 `run_per_anatomy()` 1410행에서 import 되고 새 코드는 1494행 — 스코프 안전
- `ANATOMY_RANGES` 부위명 ↔ `DS538_EXPERT_TRAINERS` 키 일치
- 문법 검증 통과
- ⚠️ 로컬 컨테이너 빌드/스모크 미실행

# PENGWIN 2026 Task 2 — 최종 보존 소스

Task 1의 fracture-instance cascade에 `peripelvic-fragment-clicks.json`을 결합하는 interactive
segmentation 컨테이너다. 이 작업트리는 원본 release `v3.7@5b8a228`을 기반으로 대회 종료 뒤
문서와 미사용 파일만 정리한 로컬 archive branch다. 외부 저장소에는 push하지 않았다.

## 현재 실행 계약

- V301 anatomy + V308 Sacrum/Hip/Femur expert, 13채널 ABBC+affinity
- 기본 affinity agglomeration `T=0.75`
- click inject ON
- 학습된 short/long affinity를 이용한 watershed split
- 두 자식 각각 300 mm³ 이상일 때만 split 채택
- femur ridge seed와 conditional adaptive decode

모델은 Task 1 최종 payload를 공유하며 상위
`../../submission_task1/model_bundles/v3_5_final_payload/model.tar.gz`가 정본이다. 로컬 v3.7은
`ruruguru` Preliminary 25/36, MP 18.8이었고 Final 행은 없다. 사용자가 지정한 `harp3133t`
Final 대표는 10/28, MP 12.4지만 comment가 비어 있어 정확한 코드 버전은 API로 확정할 수 없다.
두 값 모두 account/submission 행 순위다.

- `inference/`: click parsing, Task 1 cascade, affinity split
- `code_task1/`: trainer discovery에 필요한 segmentation 구현
- `Dockerfile`, `requirements.txt`: non-root 컨테이너 계약
- `scripts/build_image.sh`: 로컬 build helper

과거 v3.4 release note와 시각화 전용 파일은 제거했다. container build, GPU forward, 공식 evaluator
재실행은 이번 archive 정리에서 수행하지 않았다.

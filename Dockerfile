# PENGWIN 2026 **Task 2 (PENGWIN-Interact)** — interactive-segmentation container.
#
# Task 2 = Task 1 (fracture-fragment instance seg, same labels 0–200, same .mha in/out)
#          **+ an extra input** `peripelvic-fragment-clicks.json`.
#
# 이 컨테이너는 Task 1 컨테이너 규약을 그대로 미러하고, 진입점만 Task 2 용으로 바꾼다.
# Task 1 캐스케이드 코드(`task1_pipeline.py`)는 vendoring 되어 그대로 재사용되고, 모델
# 가중치는 Task 1 과 **동일한 model.tar.gz** 를 /opt/ml/model 에 얹어 공유한다.
#
# Layout:
#   /opt/app/inference/inference.py      -> Task 2 entrypoint (클릭 파싱 + 라우팅 주입)
#   /opt/app/inference/task1_pipeline.py -> vendoring 된 Task 1 캐스케이드(단일 소스 사본)
#   /opt/app/inference/agglo_decode.py   -> affinity agglomeration decoder
#   /opt/app/inference/target_family_router.py -> RF family router (클릭 모호할 때 fallback)
#   /opt/app/code_task1/                 -> 내부 helper + trainer 정의(shim 소스)
#   /opt/ml/model/                       -> model.tar.gz 내용(GC 가 런타임에 해제)

FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System libs that SimpleITK / scikit-image need at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app

# --- Python deps -----------------------------------------------------------
# Task 1 과 완전히 동일한 requirements (nnunetv2==2.5.1, torch==2.1.2+cu118 등).
COPY requirements.txt /opt/app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /opt/app/requirements.txt

# --- App code --------------------------------------------------------------
# inference/ 에는 Task 2 entrypoint(inference.py) + vendoring 된 Task 1 캐스케이드
# (task1_pipeline.py, agglo_decode.py, target_family_router.py, pengwin_trainers_shim.py)가
# 모두 들어있다. code_task1/ 는 trainer-discovery shim + trainer 정의 소스로 필요하다.
COPY inference /opt/app/inference
COPY code_task1 /opt/app/code_task1

# --- nnUNet trainer-discovery shim -----------------------------------------
# nnUNet v2 는 `nnunetv2/training/nnUNetTrainer/` 아래만 walk 하여 trainer class 를 찾는다.
# 우리의 PengwinTrainer*ABBC/Affinity 는 /opt/app/code_task1/core.py 에 있어 그 walk 밖이므로,
# build 시점에 tiny shim 을 nnUNet dir 로 복사해 이름으로 re-export 한다. site-packages 는
# root-only 이므로 반드시 USER drop 이전에 실행한다. (Task 1 Dockerfile 과 동일.)
RUN NN_TR_DIR="$(python -c 'import nnunetv2.training.nnUNetTrainer as m; print(m.__path__[0])')" \
    && cp /opt/app/inference/pengwin_trainers_shim.py "$NN_TR_DIR/pengwin_trainers.py" \
    && echo "[pengwin_task2] trainer shim installed at $NN_TR_DIR/pengwin_trainers.py" \
    && python -c "import nnunetv2.training.nnUNetTrainer.pengwin_trainers as m; print('[pengwin_task2] shim re-exports', m.__pengwin_trainer_count__, 'PengwinTrainer classes')"

# --- Runtime environment ---------------------------------------------------
# GC 는 model.tar.gz 를 /opt/ml/model/ 로 해제한다(trailing-dot convention → prefix subdir 없음).
# 비root user 는 /home/user 쓰기 권한이 없어 matplotlib 기본 캐시가 PermissionError 를 낸다 →
# HOME 과 matplotlib/XDG 캐시를 /tmp 로 돌린다.
ENV PENGWIN_ROOT=/opt/ml/model \
    nnUNet_results=/opt/ml/model/nnunet/results \
    nnUNet_preprocessed=/opt/ml/model/nnunet/preprocessed \
    nnUNet_raw=/opt/ml/model/nnunet/raw \
    PYTHONPATH=/opt/app:/opt/app/inference:/opt/app/code_task1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache

# --- Model selection (Task 1 배포 v2.2 = rank 10 과 동일) --------------------
# Stage-2 fracture net = STU-Net Affinity V308(fold_0), affinity agglomeration decode(T=0.45).
# Stage-1 anatomy = V301(fold_0). fusion/bone-reconcile OFF(코드엔 있으나 env 로 비활성).
# Task 2 는 이 동일 모델 tarball(model_v2_2.tar.gz)을 재사용하고, family 라우팅만 클릭으로 대체한다.
#
# !! PENGWIN_DS538_FOLD 는 반드시 0 이어야 한다. "all" 이 아니다 !!
# 이 블록은 Task 1 의 stale v1.9 Dockerfile 에서 복사되어 DS538_FOLD=all 을 물려받았다. 그 값이 유효했던
# model_v1_9.tar.gz 는 이미 삭제되었고, 현존하는 tarball(model_v2_2 / model_v2_3)에는
#     nnunet/results/Dataset538_.../PengwinTrainerSTUNetBaseAffinityV308__.../fold_0/checkpoint_best.pth
# 하나뿐이다 (fold_all 디렉터리 없음). task1_pipeline.py 가 use_folds=("all",) 을 만들면 nnunetv2 2.5.1 이
# isfile 검사도 fallback 도 없이 torch.load(.../fold_all/...) 을 시도 → 예외 → inference.py 의 포괄 except →
# _write_zero_seg → return 0. 즉 Grand Challenge 는 "성공(GREEN)" 으로 기록하면서 전 케이스 0점을 준다.
# 2026-07-21 검증. `tar tzf <model>.tar.gz | grep fold_all` 이 비어있지 않음을 확인하기 전에는 되돌리지 말 것.
#
# PENGWIN_TARGET_ROUTER=1 은 RF pelvic/femur 라우터를 켠다 (코드 기본값은 OFF).
# Task 2 에서는 클릭이 해부부위 튜플을 강제하므로 라우터 경로는 정상적으로는 도달하지 않는다
# (실제 클릭 1360개 전수 검사: pelvic 680 / femur 680, family=None 0건). 따라서 이 플래그는
# 클릭 JSON 이 없거나 파싱 불가한 퇴화 케이스를 위한 무료 보험이다 — 그 경우에만 라우터가 쓰이고,
# 없으면 pre-v2.0 Ds539 부피비 라우팅(GC instance F1 0.572)으로 조용히 퇴화한다.
# PENGWIN_CLICK_INJECT=0 은 배포 config(=v3.1, 2nd place)이다. 클릭 seed-injection(v3.3)은
# watershed 강제 마커로 코어를 쪼개는 실험이었으나 val 에서 REFUTED 되었다(rank 9 vs v3.1 rank 2:
# 쉬운 val 케이스에 spurious over-split 을 더함). 따라서 클릭은 seed 주입 없이 family 라우팅에만
# 쓰인다(=v3.1 동작). 0 으로 유지할 것.
# [v3.4 = v3.1 배포 경로 + always-expert Stage-B + T=0.75]
#
# Task 2 출력은 Task 1 출력과 사실상 동일하다 (2026-08-08 보드 실측: dice 0.886/0.886,
# local dice 0.873/0.873, HD95 11.214/11.215). 클릭을 seed 로 안 쓰기 때문이다
# (PENGWIN_CLICK_INJECT=0 유지 — v3.3 의 seed 주입은 val 에서 over-split 으로 기각됐다).
# 따라서 **Task 1 의 Stage-B 개선이 Task 2 로 그대로 상속된다.**
#
# 같은 팀 계정이 Task 1 Final Test 에서 always-expert + T=0.75 로 MP 14.6 을 냈고
# 우리 unified + 0.45 구성은 17.6 이었다. 그 구성을 Task 2 보드 사다리에 얹어 계산하면:
#     Split      0.150(16위) -> 0.063(4위)    +12계단
#     Topology   0.746(15위) -> 0.819(6위)    +9계단
#     Mean Position 9.5 -> 8.6   (Δ -0.9)
# 나머지 8지표는 1~6계단씩 내려간다(F1 이 -6계단으로 가장 큼). 순이득이라 채택한다.
#
# ⚠️ model.tar.gz 는 반드시 팀원 번들이어야 한다 —
#    sha256 049c38ea4abf1629a4d5f79a68a27918fd4103941fbf4f500b76211e93192919.
#    expert 체크포인트가 없는 번들로는 로드에 실패한다.
ENV PENGWIN_DS539_TRAINER=PengwinTrainerSTUNetBaseAnatomyV301 \
    PENGWIN_DS539_FOLD=0 \
    PENGWIN_DS538_TRAINER=PengwinTrainerSTUNetBaseAffinityV308DeployedVal \
    PENGWIN_DS538_TRAINER_SACRUM=PengwinTrainerSTUNetBaseAffinityV308SacrumExpertDeployedVal \
    PENGWIN_DS538_TRAINER_HIP=PengwinTrainerSTUNetBaseAffinityV308HipExpertDeployedVal \
    PENGWIN_DS538_TRAINER_FEMUR=PengwinTrainerSTUNetBaseAffinityV308FemurExpertDeployedVal \
    PENGWIN_DS538_FOLD=0 \
    PENGWIN_DS538_OUT_CH=13 \
    PENGWIN_AFFINITY_DECODE=1 \
    PENGWIN_AGGLO_T=0.75 \
    PENGWIN_FUSION_DECODE=0 \
    PENGWIN_CLICK_INJECT=0 \
    PENGWIN_STAGEA_BONE_RECONCILE=0 \
    PENGWIN_ROUTE_CC_MODE=largest \
    PENGWIN_TARGET_ROUTER=1 \
    PENGWIN_RF_CONF_MARGIN=0.15 \
    PENGWIN_TARGET_ROUTER_PATH=/opt/ml/model/stage1_router/stage1_target_router_fold0.joblib

# Grand Challenge security policy: container must not run as root.
RUN groupadd -r user && useradd --no-log-init -r -g user user

USER user:user

# GC runs the container with --network none, no extra args → Task 2 entrypoint.
ENTRYPOINT ["python", "/opt/app/inference/inference.py"]

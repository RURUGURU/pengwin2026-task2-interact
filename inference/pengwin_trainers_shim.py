"""PENGWIN ABBC trainer를 위한 nnUNet trainer-discovery shim.

nnUNet v2의 `recursive_find_python_class`는 오직
`nnunetv2/training/nnUNetTrainer/` 내부 파일만 탐색합니다. 우리의 custom
trainer는 `/opt/app/code_task1/core.py`에 있어서 해당 탐색 경로에
포함되지 않습니다. 이 shim은 Docker build 시점에 nnUNet trainer
디렉터리로 복사되어, nnUNet의 discovery가 이 module의 name attribute로
trainer class를 찾을 수 있도록 합니다.

동작 방식:
    1. /opt/app/code_task1를 sys.path에 추가 (Docker PYTHONPATH와 일치).
    2. `import core`로 V288-V291 trainer 정의를 로드.
    3. 모든 `PengwinTrainer*` class를 module level로 re-export하여
       nnUNet 내부에서 `getattr(module, trainer_class_name)`이 성공하도록 함.

이 코드는 의도적으로 `code_task1/__init__.py`에 의존하지 않습니다
(upstream package에 __init__이 없음).
"""
import os
import sys

# 컨테이너 기본값은 /opt/app/code_task1 (Dockerfile 의 COPY 위치). 로컬 dev 에서는
# PENGWIN_CODE_DIR 로 override 해 같은 파일을 그대로 site-packages 에 넣어 쓸 수 있다.
_CODE_DIR = os.environ.get("PENGWIN_CODE_DIR", "/opt/app/code_task1")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import core as _pengwin_core  # noqa: E402

# 모든 PengwinTrainer* class를 module level로 re-export하여 nnUNet의
# recursive_find_python_class가 getattr()로 가져올 수 있게 함.
for _name in dir(_pengwin_core):
    if _name.startswith("PengwinTrainer"):
        globals()[_name] = getattr(_pengwin_core, _name)

# Optional: build-time / debug 검증을 위해 개수를 노출.
__pengwin_trainer_count__ = sum(
    1 for _n in dir(_pengwin_core) if _n.startswith("PengwinTrainer")
)

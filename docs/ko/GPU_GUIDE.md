# GPU로 실행하는 방법

**Valen 로컬 추론은 이미 GPU를 사용하도록 구현되어 있습니다.** 일반적인 NVIDIA GPU 한 장 환경이라면 Python 판정 코드를 고칠 필요 없이 GPU용 실행 환경을 준비하고 모델 설정 JSON의 `device`를 지정하면 됩니다. 기본값도 `cuda`입니다.

이 문서는 설치·설정 절차이며 실제 GPU 실행을 완료했다는 기록은 아닙니다. 현재 저장소에서 검증한 것은 CPU 예제와 모의 모델을 사용한 계약 테스트입니다. 최소 GPU 메모리나 처리 속도는 아직 측정하지 않았습니다.

## 1. 먼저 실행 방식을 선택하세요

| 방식 | GPU가 필요한 곳 | 이 프로젝트에서 바꿀 것 |
|---|---|---|
| Valen을 직접 불러오기 | Python 또는 Jupyter 커널이 실행되는 컴퓨터 | Valen 설정의 `model_dir`, `revision`, `device` |
| Cosmos·다른 VLM 서버 호출 | 모델 서버 | HTTP 설정의 `endpoint`, `model`, `revision` |
| CPU 기본 예제 | GPU 불필요 | `demo` 또는 노트북 기본값 유지; 실제 영상 추론은 하지 않음 |

브라우저를 어느 컴퓨터에서 열었는지는 GPU 위치와 관계없습니다. **노트북의 Python 커널이 어디에서 실행되는지가 중요합니다.** 예를 들어 Mac의 브라우저에서 GPU 서버의 Jupyter를 열면 계산은 GPU 서버에서 합니다. Mac에서 커널도 실행한다면 `device="cuda"`만 바꾸어 NVIDIA GPU를 사용할 수는 없습니다. 현재 프로젝트는 Apple GPU의 `mps`나 AMD ROCm 환경을 검증하지 않았습니다.

아래 명령은 **NVIDIA GPU가 있는 Linux 서버의 Decision 저장소 폴더**를 기준으로 설명합니다. 서버와 모델은 사용자가 준비해야 하며 이 안내를 읽는 것만으로 클라우드 인스턴스가 생성되지는 않습니다.

## 2. Valen용 Python 환경을 준비하세요

기존 영상 생성 모델의 환경과 분리하면 패키지 버전 충돌을 줄일 수 있습니다. 다음 예는 서버에 Python 3.12가 준비되어 있다고 가정합니다.

```bash
python3.12 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip
```

이어서 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에서 서버 OS와 드라이버에 맞는 CUDA 빌드를 선택해 **현재 가상환경에 `torch`와 `torchvision`을 함께 설치**하세요. 공식 명령의 `pip` 또는 `pip3` 부분을 `python -m pip`로 바꾸면 어느 Python에 설치하는지 명확해집니다. GPU와 드라이버 정보 없이 CUDA 다운로드 주소 하나를 모든 환경의 정답으로 고정하지 않습니다.

PyTorch 설치를 마쳤으면 나머지를 설치합니다.

```bash
# 2026-10-11에 확인한 Valen-2B 모델 카드의 Transformers 버전입니다.
python -m pip install 'transformers==5.4.0'

# 영상 읽기와 노트북 도구를 설치합니다. 이 명령 자체는 PyTorch를 설치하지 않습니다.
python -m pip install -e '.[video,notebook]'
python -m pip check

python -m ipykernel install --user --name av-decision-gpu --display-name 'Decision GPU'
```

확인 시점의 Valen 모델 카드는 Python 3.10 이상, PyTorch, Transformers 5.4.0, torchvision, Pillow, av를 안내합니다. 다른 snapshot을 사용한다면 그 snapshot의 요구사항을 우선 확인하세요. Flash Attention 2는 선택 사항이며 현재 코드는 `sdpa`를 사용합니다. [Valen 공식 모델 카드](https://huggingface.co/Valen-Team/Valen-2B)

`.[video,notebook]`는 영상 처리와 Jupyter 패키지용입니다. GPU용 PyTorch를 이 선택 설치에 고정하지 않은 이유는 드라이버·OS에 맞는 빌드를 별도로 선택해야 하기 때문입니다.

## 3. 같은 Python에서 GPU가 보이는지 확인하세요

GPU 서버의 터미널에서 `nvidia-smi`로 GPU와 드라이버 상태를 확인합니다. 이어서 실행할 Python이나 노트북 셀에서 다음을 확인하세요. 터미널에서는 되는데 노트북에서 안 된다면 선택한 커널 환경이 다른 경우부터 확인합니다.

```python
import sys
import torch

print("실행 Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA 빌드:", torch.version.cuda)
print("GPU 접근 가능:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU를 사용할 수 없습니다. 드라이버·PyTorch 설치·커널을 확인하세요.")

for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    print(index, properties.name, f"전체 GPU 메모리: {properties.total_memory / 1024**3:.1f} GiB")
```

`torch.cuda.is_available()`은 접근 가능 여부를 확인하는 시작점입니다. 모델이 들어갈 만큼 메모리가 남았거나 특정 연산을 모두 지원한다는 보증은 아닙니다. 위에서 출력하는 메모리는 **전체 용량**이며 남은 용량이 아닙니다. PyTorch 설치 확인 절차는 [공식 설치 문서](https://pytorch.org/get-started/locally/)를 참고하세요.

## 4. 모델 설정 JSON을 수정하세요

[examples/provider.valen.json](../../examples/provider.valen.json)을 복사하거나 실습용으로 수정합니다. 아래의 `revision`은 실제 모델 commit 또는 관리 중인 고정 버전으로 바꾸세요.

```json
{
  "schema_version": 1,
  "provider": "valen",
  "model_dir": "/models/Valen-2B-reviewed-snapshot",
  "revision": "REPLACE_WITH_ACTUAL_MODEL_COMMIT",
  "device": "cuda:0",
  "sampling": {
    "sample_fps": 4.0,
    "max_frames": 32,
    "max_duration_s": 8.0
  }
}
```

| 항목 | 무엇을 적나요? |
|---|---|
| `model_dir` | **커널이 실행되는 GPU 서버**에 있는 모델 폴더의 경로입니다. 자신의 Mac에만 있는 경로를 넣지 않습니다. |
| `revision` | 그 폴더에 반입한 모델의 버전입니다. 문자열과 실제 파일의 동일성은 사용자가 관리합니다. |
| `device` | `cuda`는 현재 CUDA 장치, `cuda:0`은 프로세스에 보이는 첫 GPU입니다. |
| `sampling` | 입력 영상에서 추출할 프레임과 구간 한도입니다. GPU 전용 설정은 아니며 판정 근거에도 영향을 줍니다. |

모델 폴더에는 가중치뿐 아니라 해당 snapshot의 설정·토크나이저·프로세서·추론 코드도 필요합니다. 어댑터는 `local_files_only=True`로 읽으므로 실행 도중 모델 파일을 자동으로 내려받지 않습니다. `trust_remote_code=True`로 snapshot 안의 코드를 실행하는 경로입니다.

현재 [ValenProvider](../../src/av_decision/semantic_providers.py)는 내부에서 모델을 `.to(self.device).eval()`로 올립니다. 따라서 GPU 전환을 위해 `judge()`나 기준 JSON의 질문을 바꿀 필요는 없습니다. `load_provider()`만 실행하면 연결 객체만 생성되며, 첫 영상의 `observe()`에서 실제 모델을 로드합니다.

## 5. 명령행 또는 노트북에서 실행하세요

### 명령행

설정 파일을 수정하고 실제 영상 목록 `records.json`을 준비한 뒤 실행합니다.

```bash
av-decision run \
  --records /data/review-input/records.json \
  --provider-config examples/provider.valen.json \
  --out .study-runs/valen-gpu-001
```

GPU 환경에서도 `av-decision demo`를 실행하면 가상 관측값만 처리합니다. 실제 추론을 하려면 `run`이 필요합니다. 입력의 `video_path`와 분석 `window`도 GPU 서버에서 읽을 수 있어야 합니다. 판별 조건은 필요할 때만 명령 앞부분에 `av-decision --rubric 기준.json run ...` 형태로 지정합니다.

설정 파일을 쓰지 않는 기존 방식에서는 `--provider valen --model-dir ... --revision ... --device cuda:0`을 지정합니다. `--provider-config`와 `--device`를 함께 쓰면 오류가 나므로 파일 방식에서는 JSON 안의 `device`를 바꾸세요.

### Jupyter 노트북

GPU 서버의 가상환경에서 Jupyter를 실행하고, 자신의 서버 접속 방식으로 브라우저에서 엽니다.

```bash
jupyter lab notebooks/S1_Semantic_Video_Verification.ipynb
```

1. 커널을 **Decision GPU**로 선택하고, 앞 절의 GPU 확인 코드를 실행합니다.
2. 노트북의 실제 입력 설정 셀에서 `MANIFEST_PATH`, `VIDEO_ROOT`, `SOURCE_GROUP`, `SPLIT`을 채웁니다. 값은 서버 기준입니다.
3. 같은 셀에서 `PROVIDER_CONFIG = REPO / "examples/provider.valen.json"`을 지정하고 `IMPORT_REAL=True`로 바꿉니다. 처음에는 `RUN_REAL=False`로 둡니다.
4. 입력 읽기와 보정 셀을 실행해 영상별 `target`, `window`, `source_observation_hz`를 검토합니다. 원본 2Hz를 10Hz라고 적어 보류를 피하지 않습니다.
5. 입력 검토가 끝나면 실제 모델 실행 셀 전에 `RUN_REAL=True`로 지정하고 실행합니다. 설정 셀을 통째로 다시 실행했다면 입력 보정 셀도 순서대로 다시 실행합니다.
6. `report.json`의 오류, 보류 이유, 실제 관측 라벨을 확인합니다. 후보도 최종 검수 전 상태입니다.

GPU나 모델을 바꾸어 다시 실행할 때는 커널을 재시작하는 편이 이전 모델의 메모리와 설정이 남는 일을 줄입니다. 기존 출력 폴더는 덮어쓰지 않으므로 새 실행 폴더를 사용합니다.

## 6. GPU가 여러 장이면 어떻게 하나요?

**기본 실행은 모델 하나를 GPU 한 장에 올립니다.** 여러 파일을 자동 배정하려면 `run`에 `--gpus auto` 또는 `--gpus 0,1`을 추가합니다. GPU마다 완전한 모델을 하나씩 올리는 방식이며, 여러 장의 메모리로 모델 하나를 쪼개는 기능은 아닙니다. [GPU 자동 분산 안내](GPU_DISTRIBUTION.md)에 구현과 실패 처리를 설명했습니다.

예를 들어 프로세스에 보이는 두 번째 GPU를 쓰려면 `device`를 `cuda:1`로 지정합니다. 또는 실행 전에 보이는 GPU를 제한할 수 있습니다.

```bash
# 이미 할당받은 환경에서 GPU 1을 선택하는 예입니다.
CUDA_VISIBLE_DEVICES=1 av-decision run \
  --records /data/review-input/records.json \
  --provider-config examples/provider.valen.json \
  --out .study-runs/valen-selected-gpu-001
```

이 경우 선택된 GPU 하나가 프로세스 내부에서는 `cuda:0`으로 보이므로 JSON에는 `cuda:0`을 넣습니다. `cuda:1`까지 같이 지정하면 보이지 않는 GPU를 가리킬 수 있습니다. 관리형 노트북이나 컨테이너에서 이미 장치를 할당했다면 그 환경의 매핑을 따르세요. 환경변수는 Python·Jupyter를 시작하기 전에 설정합니다. 장치 노출과 순서의 의미는 [NVIDIA CUDA 환경변수 문서](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)를 참고하세요.

`--gpus`를 사용하면 입력 배정·출력 순서 통합·확신도 폴더 복사를 실행기가 담당합니다. 사용자가 GPU별 입력 목록을 직접 나눌 필요는 없습니다. 모델 설정 JSON에는 `device_map`, `tensor_parallel_size`, 자동 텐서 배치 설정이 없습니다.

## 7. 메모리가 부족하면 무엇부터 바꾸나요?

먼저 같은 GPU에서 다른 모델이 메모리를 점유하는지 확인합니다. 노트북에서 여러 모델을 차례로 만들었다면 커널을 재시작하고 필요한 모델 하나만 올려 봅니다. 타인의 프로세스를 임의로 종료하지 않습니다.

모델 자체는 올라가지만 긴 영상에서 메모리가 부족하다면, 사건 전후가 남도록 **입력 record의 `window`를 짧게 지정**하는 방법부터 시험합니다. 다음은 한 입력 항목 안에 넣는 부분 예입니다.

```json
"window": {"start_s": 1.0, "end_s": 4.0}
```

3초 구간을 초당 4장으로 분석하면 대략 12장 안팎을 사용하지만, 실제 프레임 수는 타임스탬프에 따라 달라집니다. 끼어들기 시작이나 종료를 잘라 내면 사건을 판단할 근거가 사라질 수 있으므로 사람이 구간을 확인해야 합니다.

`max_frames`는 허용 한도입니다. 이 숫자만 작게 바꿔도 프레임을 자동으로 균등하게 줄여 주지 않으며, 한도를 넘으면 오류가 납니다. `max_duration_s`도 긴 영상을 자동으로 자르는 옵션이 아닙니다. 실제 자를 구간은 `window`로 지정합니다.

`sample_fps`를 낮추면 입력이 줄 수 있지만 시간 공백이 커져 사건 판정이 보류될 수 있습니다. 예를 들어 2fps는 보통 0.5초 간격으로, 기본 허용 공백 0.3초보다 큽니다. GPU 메모리를 줄이기 위해 정확성 기준부터 느슨하게 바꾸지는 마세요.

Valen은 `dtype="auto"`를 유지합니다. 공식 설명은 학습된 FP32 파라미터와 내부 BF16 연산 경로를 구분합니다. 전체 가중치를 임의로 `.half()`나 BF16으로 바꾸면 결과가 달라질 수 있습니다. 양자화·CPU offload·여러 GPU 분산은 현재 구현의 검증 범위 밖이며, 도입하려면 어댑터 수정과 정확도 재검증이 필요합니다. [Valen 정밀도 안내](https://huggingface.co/Valen-Team/Valen-2B)

짧은 구간에서도 모델 자체를 올리지 못하면 충분한 메모리의 환경이나 다른 모델을 검토해야 합니다. 이 프로젝트는 “몇 GB면 반드시 실행된다”는 수치를 아직 제공하지 않습니다.

## 8. 원격 GPU 서버를 호출하는 경우

Cosmos 또는 범용 VLM HTTP 방식은 **모델 서버가 GPU를 사용**합니다. Decision을 실행하는 클라이언트는 영상 프레임을 추출하고 요청을 보내며, 로컬에 CUDA PyTorch가 필요하지 않습니다. 영상 처리를 위해서는 `.[video]` 설치가 필요합니다.

이때는 [Cosmos 설정](../../examples/provider.cosmos-http.json) 또는 [VLM 설정](../../examples/provider.vlm-http.json)의 서버 주소와 모델 이름을 수정합니다. GPU 종류·메모리 분산·서버 배치는 서버의 실행 설정에서 관리합니다. HTTP 설정 JSON에 `device`를 넣으면 이 프로젝트는 지원하지 않는 키로 거부합니다.

GPU 서버로 바꾸더라도 HTTP 응답의 임의 확신도를 확률로 사용하지 않는 규칙은 같습니다. 현재 HTTP 경로는 관측 라벨을 저장하고 `review`로 남깁니다. 서버 GPU 사용 여부와 후보 자동 분기 여부는 서로 다른 문제입니다.

## 9. 오류와 결과를 해석하세요

| 증상 | 먼저 확인할 내용 |
|---|---|
| `torch.cuda.is_available()`이 `False` | GPU 노출·드라이버·CUDA용 PyTorch·선택한 노트북 커널 |
| `No module named torch` 또는 `torchvision` | 모델을 실행하는 동일한 Python 환경에 설치했는지 |
| `invalid device ordinal` | `CUDA_VISIBLE_DEVICES` 적용 후 보이는 GPU 번호 |
| GPU 메모리 부족 | 다른 모델 점유·모델 로딩 자체의 실패인지 긴 입력에서의 실패인지 |
| 모델 폴더를 못 찾음 | GPU 서버 기준 경로·snapshot 파일의 완전성 |
| 프레임 한도 초과 | 실제 `window` 길이와 `sample_fps`, `max_frames`의 조합 |
| 실제 실행했는데 모두 `review` | 확률·원본 관측 주기·샘플 간격·모델 오류; HTTP 경로는 현재 기본적으로 보류 |

영상별 예외는 보고서의 `error_type`, `error`에 남고 해당 영상은 보류됩니다. CLI는 처리 오류가 하나라도 있으면 종료 코드 2를 반환합니다. GPU 연결 여부만 보지 말고 보고서의 오류도 확인하세요.

실행 환경은 Python·PyTorch·Transformers 버전, 모델 revision, GPU 이름·메모리, 프레임 수, 구간 길이를 함께 기록하세요. 현재 `elapsed_s`는 영상 추출과 최초 모델 로드 등을 포함하므로 순수 GPU 추론 시간으로 해석하지 않습니다. CPU 검증과 실제 GPU 검증의 구분은 [검증 기록](SEMANTIC_VALIDATION.md)에 있습니다.

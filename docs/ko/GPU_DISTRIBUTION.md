# GPU 자동 분산 추론의 구현과 사용법

이 프로젝트는 **여러 GPU에 파일을 자동 배정하는 데이터 병렬 추론**을 구현했습니다. GPU마다 Valen 모델을 한 개씩 올리고, 작업을 먼저 마친 GPU가 다음 영상이나 이미지를 처리합니다. 출력은 입력 순서로 다시 모으고, 확신도 → 상황 순서의 폴더와 통합 보고서를 한 번 생성합니다.

## 1. 두 종류의 분산을 구분하세요

| 목적 | 방식 | 현재 구현 |
|---|---|---|
| 많은 영상을 빠르게 나눠 처리 | GPU마다 완전한 모델 한 개, 서로 다른 파일 분석 | `--gpus auto`로 구현 |
| 큰 모델 하나가 GPU 한 장에 안 들어감 | 모델 층·가중치를 여러 장에 분할 | 별도 모델 병렬 어댑터가 필요; 현재 Valen 경로에는 미구현 |

첫 방식은 GPU 메모리를 합치는 기능이 아닙니다. 사용할 GPU **각각**에 모델과 입력이 들어갈 만큼 메모리가 있어야 합니다. GPU 수가 늘면 데이터 처리량이 개선될 여지가 있지만, 파일 읽기·모델 로드·보고서 저장 비용 때문에 속도가 GPU 수에 정확히 비례한다고 보장하지는 않습니다.

## 2. 실행 예

GPU용 설치와 실제 모델 준비는 [GPU 실행 안내](GPU_GUIDE.md)를 따릅니다. 같은 호스트에서 현재 프로세스에 보이는 CUDA 장치를 대상으로 합니다.

```bash
av-decision run \
  --records .study-runs/media-records.json \
  --provider-config examples/provider.valen.json \
  --classification-config examples/classification.ko.json \
  --gpus auto \
  --worker-timeout-s 600 \
  --out .study-runs/distributed-001
```

- `--gpus auto`: 현재 보이는 모든 CUDA GPU 중 입력 수만큼만 작업 프로세스를 만듭니다.
- `--gpus 0,2`: 현재 보이는 장치 번호 0과 2만 사용합니다. 번호 중복과 범위 초과를 거부합니다.
- `--worker-timeout-s`: 작업 프로세스 초기화 또는 파일 한 개 처리에 허용하는 시간입니다. 첫 파일에는 실제 모델 로드 시간도 들어갑니다.

자동 선택은 장치 개수를 기준으로 하며, 다른 작업의 메모리를 검사해 비어 있는 GPU를 예약하는 스케줄러는 아닙니다. 자신에게 할당된 장치만 보이도록 실행 환경을 설정하거나 `--gpus`에 번호를 지정하세요. `CUDA_VISIBLE_DEVICES`로 제한했다면 제한 후의 논리 번호를 사용합니다.

분산 실행에서는 모델 설정 JSON의 `device`를 작업별 `cuda:N`으로 대체합니다. 모델 경로와 revision, 입력 샘플링 조건은 모든 작업에서 같습니다. `provider`가 HTTP이면 이 옵션을 거부합니다. 원격 VLM 서버의 GPU 분산은 그 서버의 설정에서 담당해야 합니다.

## 3. 코드에서 어떻게 동작하나요?

구현 파일은 [distributed.py](../../src/av_decision/distributed.py)입니다.

| 함수 | 역할 |
|---|---|
| `select_devices()` | CUDA 접근 가능 여부와 현재 보이는 장치 번호 검사 |
| `run_distributed()` | 설정 확인, 전체 입력 검사, 프로세스 실행, 통합 보고서 생성 |
| `_execute_workers()` | 대기열과 프로세스 통신 관리, 완료된 GPU에 다음 입력 배정 |
| `_worker()` | 자기 GPU 지정, 모델 객체 생성, 파일별 반복 처리 |
| `process_record()` | 단일 실행과 공통으로 쓰는 관측·판정·원본 응답 저장 |
| `finish_batch()` | 입력 순서로 모인 결과를 확신도 폴더와 보고서로 저장 |

실행 순서는 다음과 같습니다.

1. 전체 입력의 중복 `clip_id`와 원본 그룹의 분할 충돌을 먼저 검사합니다. GPU별 입력만 따로 검사해 전체 충돌을 놓치지 않습니다.
2. `spawn` 방식으로 GPU마다 독립 Python 프로세스를 시작합니다.
3. 자식 프로세스가 자신의 CUDA 장치를 지정하고 연결 객체를 만듭니다. 모델 가중치는 첫 입력에서 로드하며, 이후 입력에서 같은 모델을 재사용합니다.
4. 부모가 각 준비된 프로세스에 한 파일씩 보냅니다. 결과가 돌아온 프로세스에 다음 파일을 배정합니다.
5. 근거 폴더와 raw 파일은 전체 입력 순번으로 이름을 정해 프로세스 간 충돌을 피합니다.
6. 모든 결과를 원래 순서로 모은 뒤 부모만 원본 복사·CSV·HTML·JSON·MD 보고서를 만듭니다.

CUDA의 프로세스 생성 제약 때문에 일반적인 `fork` 대신 `spawn`을 사용했습니다. 관련 원칙은 [PyTorch 다중 프로세스 안내](https://docs.pytorch.org/docs/2.14/notes/multiprocessing.html)를 참고하세요. 이 프로젝트는 추론 작업이므로 학습용 DDP의 gradient 동기화를 사용하지 않습니다.

작업 하나가 빠른 GPU에 치우칠 수 있으며, 모든 GPU에 같은 수의 파일을 억지로 배정하지 않습니다. 최종 결과에는 실제 처리 장치가 영상별 `provider.device`에 기록됩니다. 전체 실행의 장치 목록과 실패 목록은 최상위 `provider.devices`, `provider.worker_errors`에 있습니다.

## 4. 실패와 시간 초과 처리

일반적인 모델·디코더 예외는 `process_record()`가 해당 입력의 오류로 기록합니다. GPU 프로세스가 비정상 종료되거나 제한 시간을 초과하면 부모가 감지하고 그 입력을 `review` 및 확신도 `error`로 남깁니다. 살아 있는 다른 프로세스에는 아직 시작하지 않은 입력을 계속 배정합니다.

모든 작업 프로세스가 종료되어도 대기 중 입력을 결과에서 지우지 않습니다. 실행하지 못했다는 오류 행을 만들고 전체 입력 개수를 보존합니다. 시작·통신·시간 초과 오류는 `worker_errors`에 기록하며 CLI는 종료 코드 2를 반환합니다. 자동 재시도는 하지 않습니다. 재시도할 때는 입력을 추려 새 출력 폴더로 실행하세요.

정리 단계에서는 이번 실행에서 만든 프로세스만 종료합니다. 다른 사용자의 GPU 작업은 건드리지 않습니다. 키보드 중단 등으로 실행 자체를 취소하면 이미 저장한 일부 근거만 남을 수 있으며 완성된 보고서를 보장하지는 않습니다.

## 5. 노트북과 Python에서 호출하기

S1 노트북은 실제 설정 셀의 `GPUS`를 `"auto"` 또는 `"0,1"`로 바꾸면 이 함수를 사용합니다. 프로세스 실행 코드는 import 가능한 패키지 모듈에 있어 노트북 셀에서 자식 함수를 새로 정의하지 않습니다.

일반 Python 스크립트에서는 다음처럼 시작 부분을 보호합니다. 보호하지 않으면 `spawn`으로 새 Python이 스크립트를 읽을 때 작업이 재귀적으로 시작될 수 있습니다.

```python
from av_decision.classification import load_classification
from av_decision.distributed import run_distributed
from av_decision.provider_factory import load_provider_config
from av_decision.semantic import load_rubric, read_json


def main():
    report = run_distributed(
        read_json(".study-runs/media-records.json"),
        load_provider_config("examples/provider.valen.json"),
        load_rubric(),
        ".study-runs/distributed-python-001",
        devices="auto",
        classification=load_classification("examples/classification.ko.json"),
        worker_timeout_s=600,
    )
    print(report["classification_summary"])


if __name__ == "__main__":
    main()
```

## 6. 모델 하나를 여러 GPU로 나누려면

이 절은 **추가 구현 설계**입니다. 현재 실행 옵션에 `device_map="auto"`를 넣어 켜는 기능은 없습니다.

Hugging Face Accelerate는 지원되는 모델에서 층별 가중치를 여러 장치에 배치하는 Big Model Inference를 제공합니다. `device_map="auto"`가 GPU 용량을 우선 사용하고 필요하면 CPU나 디스크로 넘길 수 있으므로, 데이터 병렬 처리처럼 여러 파일이 동시에 빨라진다는 뜻은 아닙니다. [Accelerate 공식 설명](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)

Valen 전용 모델 병렬 어댑터를 추가하려면 다음을 확인하고 구현해야 합니다.

1. 사용할 snapshot의 사용자 정의 `predict()`와 decision head가 분산 장치에 있는 텐서를 처리할 수 있는지 확인합니다. 일반 생성 모델용 예제를 그대로 붙이지 않습니다.
2. `from_pretrained()`의 장치 배치 기능을 지원하는지 확인하고, 모델의 분할 불가 블록과 GPU별 메모리 한도를 정의합니다.
3. 모델을 분산 로드한 뒤 전체 객체에 `.to("cuda:0")`를 다시 호출하지 않도록 별도 로딩 경로를 만듭니다. 현재 단일 장치 어댑터는 `.to(self.device)`를 호출하므로 그대로 쓸 수 없습니다.
4. 영상 인코더 출력, 공통 상태, decision head 입력이 서로 다른 장치에 있을 때 필요한 이동을 구현하고, CPU·디스크 offload 여부를 보고서에 기록합니다.
5. `dtype="auto"`와 모델의 FP32/BF16 처리 규칙을 유지한 상태에서 단일 GPU 결과와 선택지·확률을 비교합니다.
6. 메모리·오류·처리 시간을 실제 GPU에서 측정한 뒤 별도 provider 이름으로 등록합니다. 하나의 분산 모델이 여러 GPU를 차지하므로 현재의 GPU별 모델 복제 실행기와 동시에 켜지 않도록 검사합니다.

현재 선택한 데이터 병렬 방식은 Valen의 기존 추론 API와 정밀도 경로를 유지하면서 많은 파일을 나눠 처리하기 위한 구현입니다. 한 장에 모델이 들어가지 않는 문제는 해결하지 못합니다.

## 7. 검증한 범위

CPU의 실제 `spawn` 프로세스와 가상 모델을 사용해 입력 순서 유지, 단일 실행과 같은 판정, 작업 프로세스 비정상 종료, 초기화 시간 초과, 미처리 입력 보존을 검사했습니다. 장치 자동 선택과 중복·범위 오류도 검사했습니다.

실제 여러 GPU에서의 메모리 사용량·속도·Valen 호환성은 아직 측정하지 않았습니다. 테스트 통과를 실측 성능처럼 해석하지 않습니다. 결과 폴더와 보고서의 구조는 [확신도 분류 안내](CONFIDENCE_CLASSIFICATION.md)에 있습니다.

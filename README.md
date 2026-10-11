# Decision: 자율주행 영상의 의미 재분류 스터디

**생성할 때 요청한 장면과 완성된 영상에서 실제로 보이는 장면을 구분하는 한국어 실습 프로젝트입니다.**

실제 영상·이미지를 **확신도 구간 → 상황** 순서로 폴더에 복사하고, 항목별 점수와 분포 보고서를 만들 수 있습니다. 여러 GPU에 파일을 자동 배정하는 추론 경로도 제공합니다.

“비 오는 밤에 차량이 끼어드는 영상”을 요청했더라도 비와 야간 조명만 있고 끼어드는 행동은 없을 수 있습니다. 이 프로젝트는 날씨·조명·사건·자료 품질을 나눠 기록하고, 재분류·사람 검수·생성 조건 수정을 돕습니다.

[실습 노트북](notebooks/S1_Semantic_Video_Verification.ipynb) · [기술 설명서](docs/ko/SEMANTIC_VIDEO_GUIDE.md) · [권장 판별 조건](docs/ko/SEMANTIC_CRITERIA.md) · [검증 기록](docs/ko/SEMANTIC_VALIDATION.md)

## 이론부터 조건 작성까지 읽기

새 [한국어 문서 모음](docs/ko/README.md)에 학습 순서를 정리했습니다.

| 알고 싶은 내용 | 문서 |
|---|---|
| Decision 모델·확률·보류·시간 정보의 의미 | [이론 설명서](docs/ko/THEORY.md) |
| 함수·입력·모델·보고서가 연결되는 구조 | [기술 구조 설명서](docs/ko/TECHNICAL_DESIGN.md) |
| 새 질문과 목표를 JSON에 넣는 방법 | [판별 조건 작성 안내](docs/ko/CRITERIA_AUTHORING.md) |
| 모델 선택·설정·새 모델 연결 방법 | [모델 설정과 교체](docs/ko/MODEL_CONFIGURATION.md) |
| GPU 환경 설치·장치 선택·노트북 실행 | [GPU 실행 안내](docs/ko/GPU_GUIDE.md) |
| GPU별 파일 자동 배정과 실패 처리 | [GPU 자동 분산](docs/ko/GPU_DISTRIBUTION.md) |
| 확신도 구간별 복사·상황 분류·보고서 | [확신도 분류 안내](docs/ko/CONFIDENCE_CLASSIFICATION.md) |
| 임계값에 따라 분기가 달라지는 과정 | [조작형 분기 실험](docs/assets/decision-explorer.html) |

문서에는 SVG 그림과 Mermaid 도식을 넣었습니다. 분기 실험은 저장소를 내려받은 뒤 HTML 파일을 브라우저에서 열면 동작합니다. 모든 숫자는 가상 관측의 예이며 실제 모델 성능이 아닙니다.

## 무엇을 할 수 있나요?

- 날씨, 조명, 끼어들기, 보행자 횡단, 선행차 감속 단서, 간격 감소, 장면 연속성, 관측 가능성을 따로 기록합니다.
- 생성 목표와 관측 라벨을 비교해 학습 후보·재분류 후보·판단 보류·생성 결함 검토로 나눕니다.
- 검수용 CSV, 판정 근거, 모델·기준 버전, 제한된 재생성 수정 제안을 저장합니다.
- 기존 Transfer/Predict 결과의 `manifest.json`을 읽거나 직접 작성한 입력 목록을 사용합니다.

CPU 기본 예제는 **가상 관측값으로 처리 절차를 배우는 실습**입니다. 실제 영상 인식에는 별도로 준비한 Valen 모델 또는 멀티이미지 입력을 지원하는 VLM HTTP 서버가 필요합니다. Cosmos도 이 HTTP 경로에 연결할 수 있습니다. GPU 정확도·속도는 아직 검증하지 않았습니다.

## 어떤 모델을 사용하나요?

**현재 기본 실행에서 실제 추론 모델을 사용하지는 않습니다.** `demo`와 기본 노트북은 `FixtureProvider`가 미리 만든 답을 반환합니다. 실제 영상 분석용으로는 **Valen-2B를 염두에 둔 로컬 연결**과 **Cosmos 또는 다른 영상 언어 모델(VLM)의 HTTP 연결**을 구현했습니다. 이 저장소에는 모델 가중치가 없으며 Jev 자체의 모델 코드도 포함하지 않습니다.

| 실행 경로 | 모델 또는 연결 클래스 | 역할과 현재 확인 범위 |
|---|---|---|
| `demo` | `FixtureProvider` | 학습된 모델이 아닙니다. 가상 답으로 분류 규칙과 검수 과정을 실행해 보았습니다. |
| `valen` | `ValenProvider` / 로컬 Valen 체크포인트 | 선택형 질문마다 답과 확률을 받습니다. API 요청 형식은 모의 응답으로 검사했으며 실제 GPU 추론은 아직 검증하지 않았습니다. |
| `cosmos-http` | `CosmosHTTPProvider` / 사용자가 준비한 Cosmos 서버 | 영상 이해 결과를 비교할 경로입니다. 특정 Cosmos 세대나 가중치가 코드에 고정되어 있지 않습니다. |
| `vlm-http` | `VLMHTTPProvider` / 다른 호환 VLM 서버 | 멀티이미지 입력과 JSON 답변을 지원하는 서버로 교체할 수 있습니다. 서버별 실제 호환성은 별도로 확인해야 합니다. |

Valen-2B는 Qwen3.5 기반 모델에 판정을 위한 별도 헤드를 붙인 모델입니다. 공식 설명에 따르면 하나의 입력 표현을 여러 질문에서 공유하는 `shared_state` 추론을 지원합니다. 이 특성 때문에 같은 영상에 날씨·조명·사건을 함께 묻는 실습의 후보로 선택했습니다. 모델의 구조와 지원 API는 [Valen 공식 모델 카드](https://huggingface.co/Valen-Team/Valen-2B)에서 확인할 수 있습니다.

Cosmos Reason은 물리적 장면 이해와 추론을 다루는 모델 계열이라 비교 경로로 두었습니다. 모델의 성격은 [NVIDIA 공식 저장소](https://github.com/nvidia-cosmos/cosmos-reason1)를 참고하세요. 이 프로젝트의 HTTP 연결은 사용자가 준비한 서버를 호출하며, Cosmos를 설치하거나 서버를 띄우지는 않습니다.

## 왜 모델과 판정 규칙을 나눴나요?

영상에서 무엇이 보이는지 알아내는 작업과, 그 관측을 어떤 자료로 분류할지 결정하는 작업은 성격이 다릅니다.

1. **모델은 관측을 담당합니다.** 예를 들어 “비가 보임”, “끼어들기는 판단 불가”처럼 정해 둔 선택지로 답합니다. 생성 목표는 모델에 보내지 않습니다.
2. **Python 규칙은 처리 방향을 결정합니다.** 모델 응답을 검사하고, 점수·시간 간격·목표 일치 여부를 비교해 후보·재분류 후보·보류·결함 검토로 나눕니다.
3. **사람은 최종 라벨을 검수합니다.** 후보라는 결과가 실제 정확도를 보증하지는 않습니다.

이렇게 나누면 모델을 바꿔도 같은 규칙과 보고서 형식으로 비교할 수 있습니다. 규칙 계산은 작고 단순하지만, 전체 처리 시간은 영상 읽기·모델 추론·서버 통신에 크게 좌우됩니다. **Decision 방식이라고 해서 전체 영상 분석이 빠르거나 정확하다고 이미 입증된 것은 아닙니다.** 같은 영상과 질문으로 처리 시간, 메모리, 사람 정답 대비 오류율을 측정해야 합니다.

현재 모델 입력은 시간순으로 추출한 여러 이미지입니다. 기본은 초당 4장, 최대 8초, 최대 32장이며, 네이티브 비디오 모델의 공식 벤치마크를 재현하는 경로는 아닙니다. 모델이 생성한 “확신도 0.95”를 실제 확률로 쓰지 않기 위해 HTTP 경로의 확률은 비워 둡니다. 따라서 현재 HTTP 결과는 모두 `review`로 남고 사람이 선택지를 확인합니다.

## 판별 조건은 어디에 어떻게 넣나요?

**모델을 고르는 설정과 판별 조건은 서로 다른 파일입니다.** 질문을 추가하는 것은 추론할 때 묻는 내용을 바꾸는 일이며, 모델을 재학습하는 일은 아닙니다.

| 바꾸려는 내용 | 정의하는 위치 | 예 |
|---|---|---|
| 무엇을 보고 어떤 답을 고를지 | 기준 JSON의 `questions` | `weather`, `cut_in`, 새 질문 `road_wet` |
| 확률·답의 차이·시간 공백 허용치 | 기준 JSON의 `thresholds` | `min_probability`, `min_margin`, `max_gap_s` |
| 생성 이름을 목표로 옮기는 기본 매핑 | 기준 JSON의 `target_presets` | `lane_cut_in` → `cut_in=yes` |
| 불일치 시 생성 조건 수정 제안 | 기준 JSON의 `repair_hints` | 사건 전후 구간을 더 포함하라는 안내 |
| 특정 영상에서 원하는 결과 | 입력 `records.json`의 각 항목에 있는 `target` | `{"weather": "rain", "cut_in": "yes"}` |
| 관측 가능한 시간 정보 | 각 입력 항목의 `window`, `source_observation_hz` | 분석 구간, 원본 관측 주기 |
| 어떤 모델을 호출할지 | 별도 모델 설정 JSON | `provider`, `model_dir` 또는 `endpoint`·`model` |

기본 기준은 [src/av_decision/configs/criteria.ko.json](src/av_decision/configs/criteria.ko.json)에 있습니다. 날씨·조명·끼어들기·보행자 횡단·선행차 감속 단서·간격 감소·연속성·관측 가능성의 여덟 질문이 들어 있습니다. 실험할 때는 이 파일을 복사하고 `rubric_id`를 바꿔 사용하는 편이 비교하기 좋습니다.

예를 들어 “도로가 젖어 보이는가?”를 추가하려면, 복사한 기준 파일의 `questions` 안에 다음 항목을 넣습니다. 아래는 **전체 기준 파일 중 추가할 부분**입니다.

```json
"road_wet": {
  "title": "도로 표면의 젖음",
  "instructions": "주행 도로가 젖어 보이나요? 젖음만으로 현재 비가 온다고 판단하지 마세요.",
  "criteria": {
    "yes": "도로 표면의 젖음을 뒷받침하는 단서가 보입니다.",
    "no": "표면이 충분히 보이며 젖음의 단서가 없습니다.",
    "unknown": "가림, 반사, 낮은 해상도 등으로 판단하기 어렵습니다."
  },
  "temporal": false
}
```

모든 질문에는 `unknown`이 필요합니다. `instructions`에는 판단 근거와 혼동하기 쉬운 반례를 적고, `criteria`에는 각 답의 의미를 적습니다. 전후 움직임을 봐야 하는 사건이면 `temporal`을 `true`로 둡니다. `title`은 사람이 읽는 이름이고, 모델에는 `instructions`와 선택지 설명을 보냅니다.

추가한 질문을 특정 영상의 목표 일치 검사에도 쓰려면 그 입력의 `target`에 `"road_wet": "yes"`를 넣습니다. **질문만 추가하면 관측 결과는 저장되지만 그 항목을 목표와 비교하지는 않습니다.** 폴더의 상황 기준 `scene_criteria`에 넣은 질문도 점수 검사에 포함합니다. 공통 품질 검사는 영상의 경우 `continuity`, `visibility`, 정지 이미지는 `visibility`입니다. 기준 파일의 두 품질 질문은 유지하세요.

전체 파일과 동작 예제는 이미 [criteria.road-wet.ko.json](examples/criteria.road-wet.ko.json), [custom_criteria_demo.py](examples/custom_criteria_demo.py)에 있습니다.

```bash
# 실제 모델 없이 새 조건이 분류에 반영되는 과정을 확인합니다.
python examples/custom_criteria_demo.py --out .study-runs/road-wet-001

# 실제 모델 설정을 먼저 수정한 뒤 실행합니다. --rubric은 run 앞에 둡니다.
av-decision --rubric examples/criteria.road-wet.ko.json run \
  --records .study-runs/records.json \
  --provider-config examples/provider.valen.json \
  --out .study-runs/road-wet-real-001
```

`records.json`은 실제 영상 목록을 준비하고 `target`을 검토한 파일이어야 합니다. 기본 `demo`의 가상 답은 여덟 질문에 맞춰져 있어 새 질문을 자동으로 만들어 주지 않습니다. 조건을 추가할 때는 위 전용 예제처럼 가상 답도 확장해야 합니다.

기본 임계값은 확률 `0.80`, 1·2위 확률 차이 `0.20`, 관측 공백 `0.30초`, 생성 횟수 한도 `2회`입니다. 이는 실습 시작값입니다. 확률 합이 1일 때 최대 확률이 0.80 이상이면 차이는 이미 0.60 이상이므로, 기본 차이 기준 0.20은 추가 필터 역할을 하지 않습니다. 질문별 임계값과 OR 조건은 아직 지원하지 않습니다. 자세한 설정·검증 순서는 [판별 조건 작성 안내](docs/ko/CRITERIA_AUTHORING.md)를 참고하세요.

## 모델을 변경할 수 있나요?

**가능합니다. 노트북과 명령행이 같은 모델 생성 함수를 사용하도록 구성했습니다.** [provider_factory.py](src/av_decision/provider_factory.py)가 모델을 선택하고, [semantic_providers.py](src/av_decision/semantic_providers.py)가 실제 추론을 연결합니다. 공통 판정 함수 `run_batch()`는 모델 종류에 따라 바뀌지 않습니다.

| 교체하려는 대상 | 바꾸는 부분 | 가능한 범위 |
|---|---|---|
| 다른 Valen snapshot | `model_dir`, `revision` | 현재의 `predict(..., execution="shared_state")`와 응답 형식을 지원하는 체크포인트 |
| 다른 Cosmos 서버·모델 | `endpoint`, `model`, `revision` | 시간순 멀티이미지와 JSON 답변을 지원하는 호환 서버 |
| 다른 영상 언어 모델 서버 | `provider="vlm-http"`, 서버·모델 정보 | 아래 공통 HTTP 계약을 지원하는 서버 |
| 다른 로컬 SDK·응답 형식 | `ObservationProvider` 계약을 구현한 연결 클래스 | 변환 코드를 작성해 `run_batch()`에 전달; 이름만 바꿔 연결되지는 않음 |

설정 예제는 [Valen](examples/provider.valen.json), [Cosmos HTTP](examples/provider.cosmos-http.json), [범용 VLM HTTP](examples/provider.vlm-http.json) 세 가지입니다. 파일의 경로·모델 이름·`revision` 자리표시자를 실제 값으로 바꿉니다. 예를 들어 다른 서버를 사용하려면 다음처럼 설정합니다.

```json
{
  "schema_version": 1,
  "provider": "vlm-http",
  "endpoint": "http://127.0.0.1:8000/v1",
  "model": "서버가_제공하는_실제_모델_이름",
  "revision": "실제로_배포한_모델_commit",
  "timeout_s": 180,
  "sampling": {"sample_fps": 4.0, "max_frames": 32, "max_duration_s": 8.0}
}
```

이 서버는 `/chat/completions` 요청, 여러 `image_url` 이미지와 텍스트, 순수 JSON 응답을 지원해야 합니다. 일반 텍스트 전용 모델은 사용할 수 없습니다. 여기서 HTTP 호환 형식은 통신 규약을 가리키며 OpenAI 모델을 호출한다는 뜻은 아닙니다. API 키가 필요하면 `SEMANTIC_VLM_API_KEY` 환경변수로 전달합니다.

```bash
av-decision run --records .study-runs/records.json \
  --provider-config examples/provider.vlm-http.json \
  --out .study-runs/other-model-001
```

노트북에서는 실제 모델 설정 셀의 `PROVIDER_CONFIG`만 바꿉니다. 같은 기능을 Python에서 직접 쓰려면 다음과 같습니다.

```python
from av_decision.provider_factory import load_provider
from av_decision.semantic import load_rubric, read_json, run_batch

provider = load_provider("examples/provider.vlm-http.json")
rubric = load_rubric("examples/criteria.road-wet.ko.json")
records = read_json(".study-runs/records.json")
report = run_batch(records, provider, rubric, ".study-runs/other-model-python-001")
```

기존 `--provider valen`과 `--provider cosmos-http` 방식도 유지합니다. 설정 파일과 개별 모델·샘플링 옵션은 함께 사용할 수 없습니다. 잘못된 키와 값은 실행 전에 오류로 알려 줍니다. 모델 이름·revision·샘플링 조건은 보고서에 기록하지만, 입력한 revision이 실제 가중치와 일치하는지는 자동으로 검증하지 않습니다.

교체 후에는 같은 검증 영상으로 라벨 정확도·보류율·처리 시간과 점수 분포를 다시 확인하세요. 모델 간 확률 0.80이 같은 신뢰도를 뜻하지는 않습니다. 새로운 연결 클래스 작성법과 입력·출력 계약은 [모델 설정과 교체 안내](docs/ko/MODEL_CONFIGURATION.md)에 정리했습니다.

## 1. 빠르게 시작하기

실제 NVIDIA GPU로 실행하려면 [GPU 실행 안내](docs/ko/GPU_GUIDE.md)를 함께 보세요. Valen 경로는 이미 GPU용으로 작성되어 있어, CUDA용 PyTorch 환경과 로컬 모델을 준비한 뒤 `examples/provider.valen.json`의 `model_dir`, `revision`, `device`를 지정하면 됩니다. `device="cuda:0"`은 프로세스에 보이는 첫 GPU이며, 노트북에서는 GPU 서버의 **Decision GPU** 커널과 `PROVIDER_CONFIG`를 선택합니다.

Cosmos·범용 VLM HTTP 방식은 원격 모델 서버가 GPU를 사용하므로 클라이언트의 `device`를 바꾸지 않습니다. 로컬 Valen은 `--gpus auto`로 GPU마다 모델 한 개를 올리고 파일을 자동 배정할 수 있습니다. 모델 하나를 여러 GPU 메모리에 나누는 기능과는 다릅니다. [분산 구현 안내](docs/ko/GPU_DISTRIBUTION.md)에 구조와 한계를 정리했습니다.

Python 3.10 이상에서, 이 README와 `pyproject.toml`이 있는 폴더를 기준으로 실행합니다. CPU 기본 설치에는 모델 가중치나 AWS 계정이 필요 없습니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
av-decision demo --out .study-runs/demo-001
```

Windows PowerShell에서는 가상환경 활성화 명령을 `.venv\Scripts\Activate.ps1`로 바꿉니다. 환경에 따라 `python` 대신 `python3`을 사용하세요.

동일한 실행은 다음 명령으로도 가능합니다.

```bash
python -m av_decision demo --out .study-runs/demo-002
```

예제 결과는 학습 후보 2개, 재분류 후보 1개, 보류 2개, 결함 검토 1개입니다. 출력 폴더에 `report.json`, `records.json`, `human_review.csv`, `regeneration_plans.json`이 생깁니다. 같은 폴더를 다시 지정하면 기존 결과를 보호하기 위해 오류가 나므로 새 이름을 사용하세요.

## 2. Jupyter 노트북으로 배우기

```bash
python -m pip install -e '.[notebook]'
python -m ipykernel install --user --name av-decision-study --display-name 'Decision 스터디'
jupyter lab notebooks/S1_Semantic_Video_Verification.ipynb
```

Jupyter에서 **Decision 스터디** 커널을 선택하고 위에서 아래로 실행합니다. 기본값인 `IMPORT_REAL=False`, `RUN_REAL=False`에서는 실제 영상이나 모델을 호출하지 않습니다.

노트북은 GitHub에서 읽기 좋도록 설명을 포함하고, 실행 출력은 비운 상태로 관리합니다. 전체 30개 셀 중 코드 셀은 13개입니다. 각 셀의 한국어 설명과 주석을 보며 가상 예제 → 목표 비교 → 검수 → 실제 모델 연결 순서로 진행하세요.

## 3. 실제 영상으로 확장하기

```bash
python -m pip install -e '.[video]'
```

실제 생성 출력 폴더의 manifest를 변환합니다. 아래 경로와 원본 주행 이름은 자신의 자료에 맞게 바꿉니다.

```bash
av-decision import-manifest \
  --manifest /data/cosmos-run/manifest.json \
  --video-root /data/cosmos-run \
  --source-group original-drive-001 \
  --split train \
  --out .study-runs/records.json
```

변환한 입력의 목표, 사건 구간, 원본 관측 주기를 먼저 검토합니다. Valen 설치와 모델 반입, Cosmos 서버 설정, 실제 실행 명령은 [기술 설명서](docs/ko/SEMANTIC_VIDEO_GUIDE.md#5-모델-연결-방식과-기술적-선택)에 정리했습니다. GPU 환경은 모델의 공식 요구사항에 맞춰 별도로 준비합니다.

이 프로젝트는 원본 AV 3.0 저장소 없이 CPU 실습을 실행할 수 있습니다. 원본의 로컬 M4(Transfer)·M5(Predict)와 개편 버전 M5(Transfer)·M6(Predict)는 생성 결과 파일을 통해 연결합니다. 원본 저장소의 클라우드 배포·영상 생성 코드는 이 패키지에 포함하지 않습니다.

## 4. 확신도별로 파일을 분류하고 보고서 만들기

기본 구간은 **높음 0.90 이상, 중간 0.80 이상, 낮음 0.80 미만**입니다. 판단 불가·점수 없음·처리 실패는 각각 별도 구간으로 보관합니다. 먼저 확신도를 나누고 그 안을 날씨·조명 순서로 나눕니다. 경계와 상황 순서는 [분류 설정 JSON](examples/classification.ko.json)에서 수정합니다.

대표 점수는 검사한 질문들의 **선택 확률 중 최솟값**입니다. 실제 정답률이나 이미지 유사도가 아닙니다. 항목별 선택 확률, 목표 선택지의 확률, 모든 선택지 확률, 상위 두 선택지의 차이를 `scores.csv`에 함께 저장합니다. HTTP 모델처럼 확률이 없는 경우 숫자를 임의로 만들지 않습니다.

```bash
# 실제 영상·이미지 폴더를 입력 목록으로 만듭니다.
av-decision prepare-inputs --input /data/driving-media \
  --source-group drive-001 --split unassigned \
  --out .study-runs/media-records.json

# 원본 관측 주기·분석 구간과 모델 설정을 검토한 뒤 실행합니다.
# 단일 GPU 실행은 --gpus auto를 생략합니다.
av-decision run --records .study-runs/media-records.json \
  --provider-config examples/provider.valen.json \
  --classification-config examples/classification.ko.json \
  --gpus auto --out .study-runs/classification-001
```

생성 목표가 없는 일반 분류는 `target={}`로 처리할 수 있습니다. 이미지는 `image_path`, 영상은 `video_path`를 사용합니다. 단일 이미지로 끼어들기·감속 같은 시간 사건을 확정하지 않습니다. 긴 영상에는 실제 분석할 `window`를 지정하고, 알려진 원본 관측 주기만 기록하세요.

```text
classification-001/
├── classified/high/weather=rain/lighting=night/candidate/unassigned/...
├── classified/medium/...
├── classified/low/...
├── classified/unknown/...       # 판단 불가
├── classified/unscored/...      # 점수 없음
├── classified/error/...         # 처리 실패
├── summary.html                # 분포 그림·파일별 점수·폴더 집계
├── summary.md                  # Markdown 요약
├── scores.csv                  # 질문별 상세 확률
├── classification.csv          # 파일별 분류·점수·복사 경로
├── classification.json         # 원본 대응과 폴더·실패 집계
└── report.json                 # 전체 판정 근거와 모델 정보
```

원본을 보존하고 복사본과 해시·분석 구간을 담은 JSON을 함께 저장합니다. 영상 일부를 판정해도 복사본은 원본 전체이므로 구간 정보도 확인해야 합니다. 점수가 높아도 시간 정보가 부족하면 해당 확신도 구간 안의 `review`로 나뉩니다. `--no-copy`는 동일한 기준으로 보고서만 만드는 옵션입니다.

파일 복사까지 모델 없이 먼저 연습하려면 다음 예제를 실행하세요. 색상 이미지 8개와 가상 답으로 높은 구간 2개·중간 2개·나머지 네 구간 각각 1개를 만듭니다.

```bash
python examples/confidence_sort_demo.py --out .study-runs/confidence-demo-001
```

노트북에서는 `CLASSIFICATION_CONFIG`, `COPY_CLASSIFIED`, `GPUS`를 지정합니다. 자세한 실행·점수 해석·결과 보고서 설명은 [확신도 분류 안내](docs/ko/CONFIDENCE_CLASSIFICATION.md)를 참고하세요.

## 5. 폴더 구성

```text
Decision/
├── README.md                       # 시작 안내
├── pyproject.toml                  # 설치·선택 패키지·명령행 설정
├── LICENSE / NOTICE                # 기존 라이선스와 출처
├── CONTRIBUTING.md                 # 수정·검증 방법
├── src/av_decision/                 # 독립 실행 패키지
│   ├── semantic.py                 # 판정·입력 변환·검수·평가
│   ├── semantic_providers.py       # 프레임 추출·모델 연결
│   ├── provider_factory.py         # 모델 설정·공통 연결 계약
│   ├── distributed.py              # GPU별 작업 자동 배정·실패 처리
│   ├── classification.py           # 확신도 분류·원본 복사·시각 보고서
│   ├── inputs.py                   # 영상·이미지 입력 목록 준비
│   ├── semantic_cli.py             # 명령행 도구
│   └── configs/                    # 판별 기준·확신도 분류 기준
├── notebooks/                      # 한국어 S1 실습
├── docs/                           # 한국어 설명서와 시각화 자료
├── examples/                       # 새로운 판별 조건 JSON과 실행 예제
├── tests/                          # 의미 판정과 어댑터 계약 테스트
├── scripts/                        # 문서·노트북 검사 도구
└── .github/                        # 자동 테스트와 이슈·PR 양식
```

기준 파일은 패키지에 포함됩니다. 설치 후 다른 폴더에서 실행해도 기본 기준을 찾을 수 있습니다. 기준을 수정하는 경우 `rubric_id`도 바꾸고, 비교할 실행에 `--rubric /경로/기준.json`을 명시할 수 있습니다.

## 6. 테스트와 GitHub 게시 준비

```bash
python -m unittest discover -s tests -v
python scripts/check_repository.py
python scripts/verify_notebook.py
```

노트북 검사는 `.[notebook]` 설치가 필요합니다. GitHub Actions에는 Python 3.10·3.12·3.14의 단위 테스트와 별도 노트북·배포 패키지 검사를 구성했습니다. 실제 GitHub 실행 결과는 저장소에 올린 뒤 Actions 화면에서 확인합니다.

[GitHub에 올리는 순서](docs/ko/GITHUB_GUIDE.md)에 초기화·파일 확인·원격 저장소 연결 방법을 적었습니다. 모델 가중치, 영상, `.env`, 실행 결과는 `.gitignore`에서 제외합니다. 프로젝트 수정 방법은 [기여 안내](CONTRIBUTING.md)를 참고하세요.

## 해석할 때 기억할 점

`candidate`도 사람 검수 전의 후보입니다. 확률 0.80, 선택지 차이 0.20, 관측 공백 0.30초는 교육용 시작값이며 보정된 운영 기준이 아닙니다. `unknown`은 사건이 없다는 뜻이 아닙니다. 원본 2Hz를 10fps로 저장했다고 독립 관측이 늘어나는 것도 아닙니다.

급제동·충돌 직전 여부에는 속도·궤적 등 추가 근거가 필요합니다. 재분류가 주행 정책의 성능을 높였는지는 별도의 학습 비교와 시뮬레이션으로 확인해야 합니다.

코드는 기존 워크숍의 [MIT-0 라이선스](LICENSE)와 [출처 안내](NOTICE)를 보존합니다. 외부 모델·데이터의 권한은 각각 별도로 확인합니다.

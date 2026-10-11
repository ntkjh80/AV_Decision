# 모델 설정과 교체 안내

이 문서는 **판별 기준은 유지하면서 관측 모델을 바꾸는 방법**을 설명합니다. 모델 가중치를 학습하거나 서버를 배포하는 기능은 포함하지 않습니다. 모델의 역할과 선택 이유는 [README](../../README.md)에, 조건 작성법은 [판별 조건 안내](CRITERIA_AUTHORING.md)에 정리했습니다.

GPU 설치와 실제 실행 전환은 [GPU 실행 안내](GPU_GUIDE.md)를 따라가세요. Valen의 `device` 지정, GPU 서버에서의 Jupyter 커널 선택, 메모리 부족 시 조정 방법을 설명합니다.

여러 GPU에 파일을 자동 배정하려면 [GPU 자동 분산](GPU_DISTRIBUTION.md), 점수에 따른 폴더 복사와 결과 보고서는 [확신도 분류](CONFIDENCE_CLASSIFICATION.md)를 참고하세요. 분산 실행에서는 작업별 GPU 번호가 모델 설정의 `device`를 대체합니다.

## 1. 세 가지 설정을 구분하세요

| 설정 | 핵심 필드 | 누가 읽나요? |
|---|---|---|
| 모델 설정 JSON | `provider`, `revision`, 모델 경로 또는 서버 정보, `sampling` | `load_provider()`가 읽고 모델 연결 객체를 만듭니다. |
| 판별 기준 JSON | `questions`, `thresholds`, `target_presets` | 질문 생성과 공통 판정 코드가 읽습니다. |
| 영상 목록 JSON | `video_path`, `target`, `source_observation_hz`, `window` | 영상 추출과 목표 비교에 사용합니다. |

예를 들어 날씨 질문을 고치는 일은 기준 JSON 수정이고, Valen을 다른 서버 모델로 바꾸는 일은 모델 설정 JSON 수정입니다. 같은 파일에 모두 넣지 않습니다. 생성 목표 `target`과 생성 프롬프트는 기본 어댑터의 모델 요청에 포함되지 않습니다.

## 2. 설정 파일로 교체하기

예제 파일을 복사하고 실제 값을 넣습니다. `REPLACE_WITH_...`는 설명을 위한 자리표시자이며 실제 모델 이름이나 commit이 아닙니다. 코드는 revision 문자열을 기록하지만 실제 가중치와 자동 대조하지 않으므로, 배포·반입 기록을 같이 보관하세요.

| 예제 | 수정할 필드 |
|---|---|
| [provider.valen.json](../../examples/provider.valen.json) | `model_dir`, `revision`, 필요하면 `device` |
| [provider.cosmos-http.json](../../examples/provider.cosmos-http.json) | `endpoint`, `model`, `revision` |
| [provider.vlm-http.json](../../examples/provider.vlm-http.json) | `endpoint`, `model`, `revision` |

공통 필드 `schema_version`은 `1`입니다. `sampling`을 생략하면 초당 4장, 최대 32장, 최대 8초를 사용합니다. `sample_fps`와 `max_duration_s`는 양수, `max_frames`는 2 이상의 정수입니다. 기본 구간을 늘리면 프레임 한도도 같이 검토하세요. 실제 추출 중 한도를 넘으면 샘플을 몰래 줄이지 않고 오류가 납니다.

Valen의 `device` 기본값은 `cuda`입니다. 경로는 로컬 모델 폴더여야 합니다. 상대 `model_dir`는 **JSON 파일이 있는 폴더**를 기준으로 해석합니다. `create_provider(dict)`를 직접 쓸 때만 현재 작업 폴더를 기준으로 하며, `base_dir`로 바꿀 수 있습니다. 모델 설정을 읽는 단계에서는 가중치를 로드하지 않고, 첫 `observe()`에서 로드합니다.

HTTP의 `timeout_s` 기본값은 180초입니다. `endpoint`는 예를 들어 `http://127.0.0.1:8000/v1`처럼 마지막에 `/chat/completions`를 빼고 씁니다. 인증키는 JSON에 넣지 않고 `SEMANTIC_VLM_API_KEY` 환경변수로 전달합니다. 인증정보가 섞인 URL과 알 수 없는 설정 키는 거부합니다. 이미지 바이트가 이 서버에 전달되므로 사용할 서버를 명시적으로 선택하세요.

```bash
av-decision --rubric examples/criteria.road-wet.ko.json run \
  --records .study-runs/records.json \
  --provider-config examples/provider.vlm-http.json \
  --out .study-runs/model-comparison-001
```

영상 목록은 먼저 준비해야 합니다. `--rubric`은 `run` 앞에 적습니다. 모델 설정 파일을 사용할 때는 `--model`, `--revision`, `--sample-fps` 같은 개별 옵션을 함께 적을 수 없습니다. 설정이 중복되면 오류로 알려 주므로 적용값을 추측할 필요가 없습니다.

기존 명령도 사용할 수 있습니다. 개별 옵션 역시 내부에서 같은 `create_provider()`를 거칩니다.

```bash
av-decision run --records .study-runs/records.json \
  --provider vlm-http --endpoint http://127.0.0.1:8000/v1 \
  --model ACTUAL_SERVED_MODEL --revision ACTUAL_MODEL_COMMIT \
  --sample-fps 4 --max-frames 32 --max-duration-s 8 --timeout-s 180 \
  --out .study-runs/model-comparison-002
```

모델 종류에 맞지 않는 옵션도 거부합니다. 예를 들어 HTTP 모델에 `device`를 넣어도 서버의 GPU를 바꾸는 기능이 되지는 않습니다.

## 3. 노트북과 Python에서 사용하기

S1 노트북의 실제 입력 설정 셀에는 `PROVIDER_CONFIG`가 있습니다. 다른 JSON 경로로 바꾸면 다음 실제 모델 실행 셀이 `load_provider()`로 해당 모델을 만듭니다. `IMPORT_REAL`과 `RUN_REAL`의 기본값은 계속 `False`이므로 CPU 실습에서 외부 모델을 부르지 않습니다.

```python
from av_decision.provider_factory import load_provider
from av_decision.semantic import load_rubric, read_json, run_batch

provider = load_provider("examples/provider.valen.json")
rubric = load_rubric()  # 다른 기준은 load_rubric("기준.json")으로 읽습니다.
records = read_json(".study-runs/records.json")
report = run_batch(records, provider, rubric, ".study-runs/python-model-001")
```

모델 종류·모델 이름·revision·공통 입력 방식·샘플링 조건은 `report.json`의 `provider`에 기록됩니다. Valen의 모델 이름은 로컬 체크포인트 폴더 이름입니다. revision은 사람이 제공하는 식별자이므로 이 정보만으로 파일의 동일성이 보장되지는 않습니다. HTTP 주소나 인증키는 provider 메타데이터에 넣지 않습니다.

## 4. 어떤 모델까지 설정만으로 바꿀 수 있나요?

### Valen API를 유지하는 체크포인트

현재 로컬 어댑터는 `AutoModel`로 로드한 객체가 `predict(request, execution="shared_state")`를 제공한다고 가정합니다. 입력은 `state.messages`와 `questions`, 출력은 각 질문의 `choice`와 선택지별 `probabilities`입니다. 일반적인 다른 Transformers 모델 폴더로 경로만 바꾼다고 같은 메서드가 생기지는 않습니다. 같은 Valen 계열이라도 API가 바뀐 snapshot은 연결 코드를 확인해야 합니다.

### 공통 HTTP 규약을 지원하는 영상 언어 모델

범용 `vlm-http`와 기존 `cosmos-http`는 같은 통신 코드를 사용합니다. 기존 클래스 이름과 보고서의 `cosmos_http` 표시는 유지했습니다. 서버가 다음 조건을 모두 지원하면 모델 이름과 주소를 바꿔 시험할 수 있습니다.

1. `POST {endpoint}/chat/completions`를 받습니다.
2. 한 메시지에서 시간표시 텍스트와 여러 `image_url` 항목을 받습니다. 이미지는 JPEG data URL로 보냅니다.
3. `model`, `temperature=0`, `max_tokens=2048` 요청 필드를 받습니다. 이 버전은 서버별 임의 생성 옵션을 설정 JSON으로 받지 않습니다.
4. `choices[0].message.content`에 문자열을 반환합니다. 문자열 안에는 코드 블록이나 추론 문구 없이 JSON 객체 하나가 있어야 합니다.

예를 들어 다음은 질문 두 개만 보인 응답 구조입니다. 실제로는 기준 JSON에 있는 모든 질문이 정확히 한 번씩 있어야 합니다.

```json
{"answers": {"weather": {"choice": "rain"}, "lighting": {"choice": "night"}}}
```

서버가 임의 확률이나 확신도를 써도 현재 HTTP 어댑터는 그것을 사용하지 않고 `probabilities=null`로 바꿉니다. 모델을 바꿨다는 이유로 보류를 없애거나 높은 확률을 임의로 채우지 않습니다. **현재의 HTTP 결과는 관측 라벨 검수를 돕는 자료이며 자동 통과 경로가 아닙니다.**

텍스트 전용 모델, 네이티브 비디오 URL만 받는 서버, 다른 응답 봉투를 쓰는 API는 이 어댑터로 바로 연결할 수 없습니다. 한국어 이해·여러 프레임 처리·JSON 준수도 모델마다 실제 영상을 넣어 확인해야 합니다. 특정 모델명을 설정할 수 있다는 사실이 호환성 검증을 끝냈다는 뜻은 아닙니다.

## 5. API가 다른 모델을 추가하는 방법

[ObservationProvider](../../src/av_decision/provider_factory.py)는 타입으로 표현한 공통 계약입니다. 특정 클래스를 상속하도록 강제하지는 않습니다. 아래 두 요소를 제공하는 객체라면 공통 `run_batch()`에 전달할 수 있습니다.

| 요소 | 구현할 내용 |
|---|---|
| `metadata` | JSON으로 저장할 수 있는 모델 이름·revision·실제 추론 여부·입력 방식·점수 종류 |
| `observe(record, rubric, evidence_dir)` | 영상을 읽고 질문에 답한 뒤 `(raw_answers, evidence)` 튜플 반환 |

새 어댑터는 다음 순서로 작성하세요.

1. `sample_video()`로 실제 프레임과 시간 근거를 얻습니다. 다른 샘플러를 쓰면 시간 간격 계산과 근거 저장도 직접 구현해야 합니다.
2. `questions_for_model(rubric)`의 전체 질문을 새 모델의 요청 형식으로 변환합니다. 입력 record에 목표가 있어도 모델 프롬프트에는 넣지 않습니다.
3. 반환된 라벨을 기준 파일의 선택지 ID로 변환합니다. 지원하지 않는 결과를 억지로 `no`로 바꾸지 말고, 관측이 불확실하면 `unknown`, 형식 자체가 잘못되면 예외로 남깁니다.
4. 선택지별 실제 점수를 얻는 모델이라면 그 의미를 확인한 뒤 확률 계약에 맞춥니다. 자유 생성된 확신도·유사도·정규화하지 않은 logit을 확률처럼 넘기지 않습니다. 확률을 제공하지 않는다면 `None`을 씁니다.
5. 같은 입력에 대한 단위 테스트와 실제 영상 검증을 추가합니다.

정규화 전 응답 계약은 다음과 같습니다. 아래 역시 한 질문만 보인 부분 예입니다.

```python
raw_answers = {
    "answers": {
        "road_wet": {
            "choice": "yes",
            "probabilities": {"yes": 0.90, "no": 0.05, "unknown": 0.05},
        }
    }
}
```

전체 질문 ID와 각 질문의 모든 선택지 확률을 포함해야 하며 합은 1이어야 합니다. 선택한 답은 최대 확률과 일치해야 합니다. 확률을 얻지 못하면 질문마다 `"probabilities": None`으로 둡니다. 공통 `normalize_answers()`가 누락·잘못된 선택지·비정상 확률을 검사합니다.

`evidence`는 JSON으로 저장 가능한 관측 근거입니다. 시간 조건에 사용하는 `max_gap_s`를 실제 샘플링 결과에서 계산해 넣으세요. 이를 생략하면 시간 정보 부족으로 보류됩니다. 원본 관측 주기는 입력 record의 `source_observation_hz`를 사용합니다. 샘플링한 이미지 수만으로 원본 관측 주기를 만들어 넣지 않습니다.

Python에서는 새 객체를 `run_batch(records, 새_객체, rubric, output_dir)`에 전달하면 됩니다. CLI에서도 새 이름으로 선택하려면 `provider_factory.py`의 `PROVIDER_NAMES`와 `create_provider()`에 생성 분기·필수 필드 검사를 추가합니다. CLI 선택 목록은 이 상수를 함께 쓰므로 따로 문자열 목록을 복사할 필요가 없습니다. 현재는 설정 JSON에 모듈 경로나 실행 코드를 넣어 임의 클래스를 불러오는 기능이 없습니다.

## 6. 교체한 뒤 확인할 것

같은 원본 그룹·영상 구간·판별 기준으로 별도 출력 폴더에 실행합니다. 모델뿐 아니라 프레임 수나 질문도 바꾸면 차이가 어느 변경에서 왔는지 알기 어려우므로 하나씩 바꾸는 것이 좋습니다.

- 먼저 모든 질문을 반환하는지, `unknown`과 시간 정보를 지키는지 확인합니다.
- 사람이 검수한 정답을 기준으로 항목별 오류율과 보류율을 함께 봅니다.
- 모델을 처음 올리는 시간과 준비된 뒤 추론 시간을 구분합니다. 현재 `elapsed_s`에는 영상 추출·추론·일부 저장 시간이 함께 들어가므로 순수 GPU 연산 시간으로 해석하지 않습니다.
- 모델이 바뀌면 확률 분포도 달라집니다. 기존 임계값이 새 모델에도 적합한지 검증 자료에서 다시 확인합니다.

새 경로의 소프트웨어 검증 결과는 [검증 기록](SEMANTIC_VALIDATION.md)에 있습니다. 실제 GPU·서버 성능은 별도로 측정해야 합니다.

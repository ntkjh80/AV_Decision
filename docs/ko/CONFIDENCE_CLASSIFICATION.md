# 확신도 → 상황별 폴더 분류와 결과 보고서

**확신도를 먼저 나누고, 같은 확신도 구간 안에서 상황을 분류합니다.** 기본 예는 높은 확신도 → 비 → 야간 → 학습 후보 → validation 순서입니다. 모델의 선택 확률과 목표 선택지의 확률을 함께 저장하며, 원본 파일은 그대로 유지하고 분류 폴더에 복사합니다.

이 기능은 실제 영상·이미지 입력을 받습니다. 실제 결과를 얻으려면 Valen 또는 호환 VLM 서버가 준비되어 있어야 합니다. 함께 제공한 색상 이미지 실습은 가상 응답으로 파일 정리 과정을 확인하는 예제이며 모델 성능 시험은 아닙니다.

## 1. 여기서 말하는 확신도는 무엇인가요?

현재 Valen 연결에서 얻는 것은 **질문별 선택지 확률**입니다. 이미지와 문장 사이의 코사인 유사도를 계산한 값은 아닙니다. 또한 모델 확률 0.90이 실제 정답률 90%라는 뜻도 아닙니다. 코드와 보고서에는 이를 구분하기 위해 `score_kind`와 `calibrated_correctness_probability=false`를 기록합니다.

| 보고하는 값 | 예 | 해석 |
|---|---|---|
| `selected_probability` | 모델이 rain을 골랐고 P(rain)=0.94 | 선택한 답에 모델이 부여한 확률 |
| `target_probability` | 목표는 rain인데 모델은 clear를 선택; P(rain)=0.03 | 요청한 목표 답에 모델이 부여한 확률 |
| `margin` | 1위 0.94, 2위 0.03 → 0.91 | 상위 두 선택지 사이의 차이 |
| `probabilities_json` | 모든 날씨 선택지의 확률 | 다른 선택지와의 경쟁 상태 |
| `min_selected_probability` | 날씨 0.95, 조명 0.85, 품질 0.94 → 0.85 | 폴더를 나눌 대표 점수; 검사 항목 중 최저 선택 확률 |

여러 질문의 확률을 평균내면 한 항목의 약한 근거가 다른 항목의 높은 점수에 가려질 수 있습니다. 그래서 대표 점수는 최솟값으로 정했습니다. 이 값은 여러 조건이 모두 맞을 확률을 수학적으로 계산한 것이 아닙니다. 기준이 다른 실험끼리 대표 점수만 비교하지 말고 항목별 점수도 함께 보세요.

대표 점수에 포함하는 항목은 다음과 같습니다.

- 영상: `target`의 질문 + `scene_criteria`의 질문 + `continuity`, `visibility`
- 정지 이미지: `target`의 질문 + `scene_criteria`의 질문 + `visibility`

모델은 기준 파일에 있는 전체 질문에 답하지만, 대표 점수는 위 검사 항목으로 계산합니다. 상황별 폴더에 쓰는 질문도 반드시 검사하므로, 낮은 날씨 점수를 무시한 채 높은 확신도 날씨 폴더로 들어가지는 않습니다.

정지 이미지는 시간적 연속성을 판단할 수 없어 `continuity`를 공통 검사에서 뺍니다. 모든 시간 질문의 최종 관측은 `unknown`, 점수는 없음으로 바꿉니다. 원래 모델 응답은 `raw-*.json`에 남깁니다. 이미지 목표나 상황 기준에 `cut_in` 같은 시간 질문을 넣으면 그 입력은 판단 불가·검수 대기로 분류됩니다.

## 2. 확신도 구간과 상황 기준을 정하세요

기본 파일은 [classification.ko.json](../../src/av_decision/configs/classification.ko.json)이며, 수정용 예제는 [examples/classification.ko.json](../../examples/classification.ko.json)입니다.

```json
{
  "schema_version": 1,
  "high_min_probability": 0.9,
  "medium_min_probability": 0.8,
  "scene_criteria": ["weather", "lighting"]
}
```

| 최상위 구간 | 기본 조건 |
|---|---|
| `high` | 대표 점수 0.90 이상 |
| `medium` | 0.80 이상, 0.90 미만 |
| `low` | 0.80 미만 |
| `unknown` | 검사 항목에 `unknown`이 있음 |
| `unscored` | 검사 항목의 확률이 누락됨 |
| `error` | 모델·디코딩·입력 검증·분산 작업의 처리 실패 |

처리 실패 → 판단 불가 → 점수 없음 → 숫자 구간 순서로 판단합니다. 예를 들어 모델이 “판단 불가”를 확률 0.98로 골랐어도 높은 확신도 상황 폴더에 넣지 않고 `unknown`에 둡니다. 점수가 일부 없는 경우 대표 점수를 임의로 평균내거나 0으로 채우지 않습니다.

`scene_criteria`는 **각 확신도 구간 아래의 상황 폴더 순서**입니다. 사건별로 나누려면 `["cut_in", "pedestrian_crossing"]`, 상황 폴더가 필요 없으면 `[]`로 지정합니다. 질문 ID는 판별 기준의 `questions`에 이미 있어야 합니다.

두 설정 파일의 역할은 다릅니다.

| 파일 | 바꾸는 내용 |
|---|---|
| `criteria.ko.json` | 질문·선택지·검수 기준. 예: 확률 0.80, margin 0.20, 시간 공백 0.30초 |
| `classification.ko.json` | 확신도 폴더의 경계와 그 안의 상황 폴더 순서 |

예를 들어 높은 구간을 0.95로 바꾸어도 후보 판정 기준이 자동으로 0.95가 되지는 않습니다. 확신도 구간은 정리 기준이고 `candidate`, `review` 등은 근거·시간 정보까지 고려한 판정입니다. 점수가 높아도 시간 정보가 부족하면 `high/.../review/...`에 들어갑니다.

## 3. 영상·이미지 목록을 준비하세요

저장소를 설치하고 실제 영상·이미지만 모아 둔 입력 폴더를 지정합니다. 출력 폴더는 입력 폴더와 분리하면 이전 분류 결과가 다음 입력으로 섞이는 일을 줄일 수 있습니다.

```bash
python -m pip install -e '.[video]'

av-decision prepare-inputs \
  --input /data/driving-media \
  --source-group drive-001 \
  --split unassigned \
  --out .study-runs/media-records.json
```

`--input`에는 폴더 또는 파일 하나를 넣습니다. 폴더는 재귀 탐색합니다. 지원 확장자는 영상 `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, 이미지 `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tif`, `.tiff`입니다. 실제 디코딩 가능 여부는 추론 때 검사합니다. 움직이는 이미지나 여러 페이지 이미지는 정지 이미지 한 장으로 취급하지 않습니다.

`source_group`은 같은 원본 주행·수집 묶음을 가리킵니다. 서로 다른 그룹은 따로 준비한 뒤 목록을 합칠 수 있습니다. 같은 원본이 train과 test로 갈라지면 배치 실행 전에 거부합니다. 단순 폴더 분류를 시작할 때는 `unassigned`를 사용할 수 있지만 학습·시험 편입 전 원본 단위로 분할해야 합니다.

생성 목표와 비교할 필요가 없다면 `target={}`로 분류할 수 있습니다. `prepare-inputs`도 기본적으로 빈 목표를 만듭니다. 목표가 있으면 `{"weather": "rain"}` 같은 JSON 파일을 만들어 `--target-file /data/target.json`을 추가합니다. 목표는 모델에 보내지 않고 관측 후 비교합니다.

영상에는 원본 관측 주기와 분석 구간을 검토해 넣으세요. 아는 경우에만 `--source-observation-hz 10`, `--start-s 0 --end-s 5`처럼 지정할 수 있습니다. 이 값은 준비하는 영상 전체에 적용되므로 영상마다 다르면 만들어진 JSON을 개별 수정합니다. 모르는 주기를 추측하거나 인코딩 fps로 대신하지 않습니다. 기본 분석 길이 한도는 8초이며 긴 영상은 구간이 필요합니다.

입력 항목의 예는 다음과 같습니다.

```json
[
  {
    "clip_id": "image-001", "source_group": "drive-001", "split": "unassigned",
    "media_type": "image", "image_path": "/data/frame.png", "target": {}
  },
  {
    "clip_id": "video-001", "source_group": "drive-002", "split": "unassigned",
    "media_type": "video", "video_path": "/data/clip.mp4", "target": {"cut_in": "yes"},
    "source_observation_hz": 10, "window": {"start_s": 0, "end_s": 5}
  }
]
```

`source_observation_hz=10`은 실제 원본이 그 주기로 관측된 경우에만 사용합니다. 파일 경로는 실제 실행 컴퓨터 기준이며, 같은 영상을 여러 구간으로 분석하면 구간마다 다른 `clip_id`를 지정합니다.

## 4. 점수 출력·폴더 분류·보고서를 한 번에 실행하세요

먼저 [모델 설정](MODEL_CONFIGURATION.md)의 경로·revision을 실제 값으로 수정합니다.

```bash
av-decision run \
  --records .study-runs/media-records.json \
  --provider-config examples/provider.valen.json \
  --classification-config examples/classification.ko.json \
  --out .study-runs/classification-001
```

`run`은 기본적으로 복사합니다. `--no-copy`를 붙이면 같은 점수·판정 기준으로 보고서만 만들고 원본을 복사하지 않습니다. 명령행에는 파일별 대표 점수와 전체 질문의 선택 확률·margin을 출력합니다. 많은 파일을 처리할 때는 CSV와 HTML을 중심으로 확인하세요.

여러 GPU에 파일을 자동 배정하려면 위 명령에 `--gpus auto`를 추가합니다. 장치를 고르려면 `--gpus 0,1`을 씁니다. 실제 구조와 조건은 [GPU 자동 분산 안내](GPU_DISTRIBUTION.md)에 설명했습니다.

폴더 구조는 다음과 같습니다. 실제 입력에 해당하는 폴더만 생성됩니다.

```text
classification-001/
├── classified/
│   ├── high/
│   │   └── weather=rain/lighting=night/candidate/validation/
│   │       ├── 000000-식별해시.mp4
│   │       └── 000000-식별해시.mp4.json
│   ├── medium/
│   ├── low/
│   ├── unknown/
│   ├── unscored/
│   └── error/
├── summary.html                # 확신도 분포·파일별 점수·실제 폴더 집계
├── summary.md                  # Markdown 결과 요약
├── scores.csv                  # 질문별 선택 확률·목표 확률·margin·전체 분포
├── classification.csv          # 파일 한 개당 분류·점수·복사 결과 한 행
├── classification.json         # 폴더 집계·원본→복사본 대응·실패 상세
├── report.json                 # 판정·모델·근거·점수·분류 결과 전체
├── records.json                # 사용한 입력 목록
├── human_review.csv            # 사람 검수용 빈 정답 칸과 모델 점수
├── raw-0000.json               # 정규화하기 전 모델 응답
└── evidence-0000/              # 실제 모델 경로의 관측 이미지·샘플링 근거
```

원본을 이동하거나 삭제하지 않습니다. 복사본 이름에는 입력 순번과 ID 해시를 붙여 같은 원본 파일명 때문에 덮어쓰지 않도록 했습니다. 옆의 JSON에는 원본 경로·원본 그룹·점수·판정·분석 구간·복사본 해시가 들어 있습니다. 원본 영상의 일부분을 판정했어도 **복사본은 영상 전체**입니다. 전체 영상이 그 라벨로 검증되었다고 해석하지 말고 JSON의 `assessed_window`를 확인하세요.

추론 때 기록한 원본 해시와 복사 중 계산한 해시가 다르면 복사를 취소합니다. 파일 없음·저장 실패 등은 `copy_failed_count`와 `export_error`에 기록합니다. 입력 수 = 복사 성공 수 + 복사 실패 수는 복사를 켠 실행에서 성립합니다. 점수만 저장하는 실행에서는 두 복사 수가 모두 0입니다. 추론 실패 자료도 원본이 읽히면 `error` 아래로 복사해 검토할 수 있습니다.

처리 오류·분산 작업 오류·복사 실패가 있으면 보고서를 남긴 뒤 CLI 종료 코드 2를 반환합니다. 단순히 확신도가 낮거나 점수가 없어 보류된 것은 처리 오류와 다릅니다. 충분한 출력 저장 공간은 필요합니다.

## 5. 노트북에서 사용하는 방법

S1 노트북의 실제 입력 설정 셀에서 다음을 지정합니다.

```python
CLASSIFICATION_CONFIG = REPO / "examples/classification.ko.json"
COPY_CLASSIFIED = True
GPUS = "auto"  # 단일 실행이면 None
```

기존 manifest를 읽는 흐름 대신 직접 준비한 이미지·영상 목록을 쓸 때는 입력 준비 셀 다음에 `real_records = read_json("입력목록.json")`으로 지정하고 내용을 확인합니다. 실제 모델 실행 셀은 점수 표와 `summary.html`, `scores.csv` 위치를 출력합니다. GPU 준비 방법은 [GPU 실행 안내](GPU_GUIDE.md)를 참고하세요.

Python에서 직접 호출할 때는 명시적으로 분류 설정을 전달합니다. 기존 CPU 실습의 `run_batch()` 호출은 분류 설정을 생략하므로 파일 복사를 시도하지 않지만 점수 보고서는 생성합니다.

```python
from av_decision.classification import load_classification
from av_decision.provider_factory import load_provider
from av_decision.semantic import load_rubric, read_json, run_batch

report = run_batch(
    read_json(".study-runs/media-records.json"),
    load_provider("examples/provider.valen.json"),
    load_rubric(),
    ".study-runs/python-classification-001",
    classification=load_classification("examples/classification.ko.json"),
)
```

## 6. 모델 없이 폴더와 보고서부터 확인하기

```bash
python examples/confidence_sort_demo.py --out .study-runs/confidence-demo-001
```

작은 색상 PNG 8개를 만들고 가상 응답을 적용합니다. 결과는 높은 구간 2개, 중간 2개, 낮음·판단 불가·점수 없음·처리 실패 각각 1개입니다. 처리 실패도 원본이 있으므로 8개 모두 각 폴더에 복사됩니다. 예제에는 의도한 오류 한 개가 있어 실패 구간과 보고서 표시도 확인할 수 있습니다.

`result/summary.html`을 브라우저에서 열어 분포와 파일별 결과를 확인하세요. 이 색상 이미지의 날씨 라벨은 가상 답이며 영상 인식 성능을 의미하지 않습니다.

## 7. HTTP 모델의 점수가 없는 경우

현재 Cosmos·범용 VLM HTTP 어댑터는 자유 생성한 확신도를 버리고 선택지만 보관합니다. 일반적으로 `unscored`로 나뉘며, 검사 항목이 `unknown`이면 `unknown`, 호출이나 형식 오류면 `error`입니다. 숫자 점수를 반드시 얻고 싶다면 선택지별 확률을 반환하는 모델 경로를 쓰거나 실제 점수 계약을 검증한 새 어댑터가 필요합니다. 임의 유사도를 정답 확률로 바꾸거나 누락 점수에 1.0을 넣는 기능은 제공하지 않습니다.

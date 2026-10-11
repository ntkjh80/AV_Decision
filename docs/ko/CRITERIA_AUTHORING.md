# 판별 조건 작성 안내: 질문·목표·임계값 넣는 방법

[문서 모음](README.md) · [이론](THEORY.md) · [기술 구조](TECHNICAL_DESIGN.md)

“조건을 넣는다”는 말은 여러 작업을 뜻할 수 있습니다. 먼저 **새로 물어볼 질문을 만드는 것인지, 이번 영상의 목표를 정하는 것인지, 분기 임계값을 바꾸는 것인지** 구분하세요. 아래 순서대로 따라 하면 실제 파일을 만들고 실행 결과까지 확인할 수 있습니다.

![질문·목표·임계값의 서로 다른 역할](../assets/condition-layers.svg)

## 1. 어디에 무엇을 넣나요?

| 바꾸고 싶은 내용 | 위치 | 예 |
|---|---|---|
| 모델에게 물어볼 것 | 기준 JSON의 `questions` | “도로가 젖어 보이나요?” |
| 이번 영상이 갖춰야 할 조건 | 입력 기록의 `target` | `{"road_wet": "yes"}` |
| 자동 분기에 필요한 점수와 시간 근거 | 기준 JSON의 `thresholds` | 확률 0.85 이상 |
| 기존 생성 이름을 목표로 바꾸기 | 기준 JSON의 `target_presets` | `wet_road` → `road_wet=yes` |
| 다시 생성할 때 검토할 문장 | 기준 JSON의 `repair_hints` | 젖은 노면 단서를 포함하도록 제안 |

`questions`에 추가하면 모든 영상에 그 질문을 보냅니다. 하지만 **그 질문이 자동 통과 조건이 되려면 `target`에도 들어가야 합니다.** 예외는 항상 검사하는 `continuity`, `visibility`입니다.

## 2. 기본 기준을 복사해서 시작하세요

저장소 루트에서 실행합니다. 패키지를 수정할 때는 editable 설치를 사용하면 작업 중인 코드를 그대로 실행하기 편합니다.

```bash
python -m pip install -e .
```

다음 Python 예제는 기본 기준을 작업용 폴더에 저장합니다. 이 코드와 이어지는 예제는 같은 Python/Jupyter 세션에서 실행합니다.

```python
from pathlib import Path
from av_decision.semantic import load_rubric, write_json

rubric = load_rubric()
rubric["rubric_id"] = "my-driving-criteria-v1"
custom_path = Path("configs/custom/criteria.ko.json")

# 아직 저장하지 않습니다. 아래 단계에서 질문과 목표 매핑을 추가합니다.
```

기본 여덟 질문은 우선 유지하세요. 현재 코드는 공통 질문 `continuity`, `visibility`와 그 선택지를 직접 사용합니다. 이 ID를 지우거나 바꾸는 것은 단순한 설정 변경으로 끝나지 않습니다.

## 3. 질문 하나를 작게 정의합니다

여기서는 “도로가 젖어 보이는가?”를 추가합니다. 이것은 “현재 비가 오는가?”와 다른 질문입니다. 비가 그친 뒤에도 도로는 젖어 있을 수 있으므로 분리해서 묻는 편이 의미가 명확합니다.

```python
rubric["questions"]["road_wet"] = {
    "title": "젖은 노면의 시각적 단서",
    "instructions": (
        "주행 노면에 물막, 물웅덩이, 젖은 표면의 연속적인 반사 등 "
        "젖은 노면을 뒷받침하는 단서가 보이나요? "
        "현재 비가 내리는지와는 별개로 판단하세요. "
        "조명 반사만으로 확정하지 말고 근거가 부족하면 unknown을 고르세요."
    ),
    "criteria": {
        "yes": "노면이 보이며 물막이나 물웅덩이 등 젖은 상태의 단서가 있습니다.",
        "no": "노면을 충분히 관측할 수 있고 젖은 상태의 단서는 보이지 않습니다.",
        "unknown": "노면이 가려지거나 반사·해상도 때문에 젖은 상태를 구분하기 어렵습니다."
    },
    "temporal": False
}
```

| 필드 | 작성 방법 |
|---|---|
| 질문 ID `road_wet` | 보고서와 목표에서 쓰는 키입니다. 짧고 일관된 영문 ID를 권장합니다. |
| `title` | 사람이 표에서 읽을 한국어 이름입니다. |
| `instructions` | 관측할 사실, 혼동할 반례, 판단 불가 조건을 적습니다. |
| `criteria` | 가능한 답과 각 답의 의미를 적습니다. `unknown`이 필요합니다. |
| `temporal` | 전후 과정이 반드시 필요한지 지정합니다. 현재 코드에서는 시간 검사에 사용합니다. |

선택지들은 가능한 한 서로 겹치지 않게 작성하세요. “위험함”처럼 범위가 넓은 질문보다 “옆 차로 차량이 자차 앞으로 들어오는 과정이 보이는가?”처럼 관측 대상이 분명한 질문이 좋습니다.

## 4. 생성 이름의 매핑과 수정 제안도 적습니다

```python
rubric["target_presets"]["wet_road"] = {"road_wet": "yes"}
rubric["repair_hints"]["road_wet"] = (
    "젖은 노면이 목표라면 물막이나 물웅덩이 등 관측 가능한 단서를 포함해 주세요. "
    "조명 반사만 과장하지 말고 원래 도로와 차량의 구조를 함께 확인해 주세요."
)

# 최초 생성이 포함된 총 횟수입니다. GPU 생성 작업을 실행하는 설정은 아닙니다.
rubric["thresholds"]["max_attempts"] = 2
write_json(custom_path, rubric)
print(custom_path)
```

`write_json()`은 기존 파일을 덮어쓰지 않습니다. 두 번째 실험은 새 파일 이름을 쓰거나 기존 작업 파일을 편집기로 수정하세요.

`target_presets`는 manifest 변환에만 사용합니다. `condition` 또는 `scenario` 값이 정확히 `wet_road`인 경우에만 목표가 매핑됩니다. 문장에 비슷한 단어가 들어 있다고 추측해서 매핑하지 않습니다.

`repair_hints`는 수정 제안 문장입니다. 모델 질문에 전달하지 않으며, 실제 생성 작업도 실행하지 않습니다. 새로운 기준에 이 항목을 빠뜨리면 불일치를 찾고도 수정 문장이 비어 있을 수 있습니다.

## 5. 이번 영상의 목표를 넣습니다

기존 입력 기록의 `target`을 다음처럼 설정합니다. 입력 파일은 이와 같은 객체들의 JSON 목록입니다.

```json
[
  {
    "clip_id": "wet-road-001",
    "video_path": "/data/clips/wet-road-001.mp4",
    "source_group": "drive-001",
    "split": "train",
    "target": {"road_wet": "yes"},
    "target_needs_review": false,
    "source_observation_hz": null,
    "window": {"start_s": 0.0, "end_s": 5.0},
    "attempt": 1
  }
]
```

위 경로는 예시입니다. 실제 영상과 원본 주행 정보를 넣어야 합니다. 원본 관측 주기는 확인하기 전까지 `null`을 유지합니다. `road_wet` 자체는 시간 질문이 아니지만, 공통 `continuity` 검사가 시간 근거를 요구하므로 현재 규칙에서 자동 후보가 되려면 원본 주기도 확인되어야 합니다.

여러 목표를 동시에 요구하면 `target`에 함께 넣습니다.

```json
{"weather": "rain", "lighting": "night", "road_wet": "yes", "cut_in": "yes"}
```

이는 모든 목표가 각각 일치해야 한다는 뜻입니다. `{"weather": ["rain", "snow"]}`처럼 목록을 넣는 OR 조건은 현재 지원하지 않습니다. 목록을 넣지 말고 질문 설계 또는 코드 확장부터 검토하세요.

## 6. 설정 파일을 실제 실행에 지정합니다

중요한 점은 **`--rubric`을 `run`이나 `import-manifest` 앞에 둬야 한다는 것**입니다. 현재 명령행에서는 전체 도구의 옵션이기 때문입니다.

기존 생성 결과에서 입력 목록을 만드는 예:

```bash
av-decision --rubric configs/custom/criteria.ko.json import-manifest \
  --manifest /data/generated/manifest.json \
  --video-root /data/generated \
  --source-group drive-001 \
  --split train \
  --out .study-runs/wet-road-records.json
```

로컬 Valen 실행 예:

```bash
av-decision --rubric configs/custom/criteria.ko.json run \
  --records .study-runs/wet-road-records.json \
  --provider valen \
  --model-dir /models/reviewed-valen-snapshot \
  --revision ACTUAL_MODEL_COMMIT \
  --sample-fps 4 --max-frames 32 \
  --out .study-runs/wet-road-valen-001
```

모델 경로와 revision은 실제 값으로 바꿉니다. 모델 환경 준비는 [설치·실행 설명](SEMANTIC_VIDEO_GUIDE.md)을 참고하세요. JSON 파일을 만들었다고 GPU 모델이 자동 설치되지는 않습니다.

### S1 노트북에서 지정하기

처음 설정 셀의 `rubric = load_rubric()`를 다음처럼 바꿉니다.

```python
rubric = load_rubric(REPO / "configs/custom/criteria.ko.json")
```

다만 S1 앞부분의 `demo_data()`는 기본 여덟 질문만 만드는 예제입니다. 새 질문을 추가하면 그 CPU 예제의 응답에도 새 질문을 넣어야 하므로, 먼저 아래 전용 예제로 검사하는 것이 좋습니다. 실제 모델 경로는 추가한 질문을 요청에 포함합니다.

## 7. GPU 없이 새 조건을 시험하는 완성 예제

복사해서 사용할 수 있는 전체 기준과 실행 코드를 준비했습니다.

- [완성 기준 JSON](../../examples/criteria.road-wet.ko.json)
- [새 조건의 CPU 실행 예제](../../examples/custom_criteria_demo.py)

```bash
python examples/custom_criteria_demo.py --out .study-runs/road-wet-001
```

예제는 다음 세 가지 가상 관측을 만듭니다. 점수는 모두 설명용이며 실제 모델 성능이 아닙니다.

| 가상 관측 | 목표 | 예상 결과 |
|---|---|---|
| 젖은 노면 단서가 있음 | `road_wet=yes` | `candidate` |
| 젖은 노면 단서가 없음 | `road_wet=yes` | `relabel_candidate` |
| 노면을 구분하기 어려움 | `road_wet=yes` | `review` |

결과 폴더에는 사용한 `criteria.ko.json`, `report.json`, 검수 CSV, 재생성 제안이 남습니다. `observed.road_wet`, `checked_criteria`, `mismatched_criteria`를 확인해 새 질문이 실제 비교에 쓰였는지 읽어 보세요.

기본 `av-decision demo`는 새 질문에 대한 가상 응답을 자동 생성하지 않습니다. 새 기준을 기본 demo에 그대로 지정하면 응답 질문 수가 맞지 않아 오류가 기록될 수 있습니다. 이는 모델 성능 저하가 아니라 실습 데이터 계약의 불일치입니다.

## 8. 임계값을 바꿀 때의 순서

```json
{
  "min_probability": 0.85,
  "min_margin": 0.20,
  "max_gap_s": 0.30,
  "max_attempts": 2
}
```

위 객체는 `thresholds` 부분만 보여 줍니다. 이 수치를 권장 운영값이라고 해석하지 마세요.

1. 검수자가 라벨을 붙인 validation 자료를 준비합니다.
2. 질문 정의를 먼저 고정합니다.
3. 임계값을 바꾼 별도 실행을 만들고 정확도·보류율·처리 실패율을 비교합니다.
4. 안개·야간·희귀 사건이 지나치게 사라지는지도 확인합니다.
5. 설정을 고정한 다음 최종 test를 평가합니다.

[분기 실험](../assets/decision-explorer.html)으로 기준 변경의 효과를 미리 볼 수 있습니다. 확률 0.80 이상이면 점수 차이는 최소 0.60이므로, 기본 차이 기준 0.20은 추가 필터 역할을 하지 않는다는 점도 [이론 설명](THEORY.md)에서 확인하세요.

`max_gap_s`를 늘리면 시간 관련 보류가 줄 수 있지만, 관측 정보가 늘어난 것은 아닙니다. 특히 2Hz 자료를 통과시키려고 실제 주기를 10Hz로 바꾸는 것은 설정 조정이 아니라 입력 정보의 왜곡입니다.

## 9. 작성한 조건을 확인하는 질문

| 확인할 질문 | 좋은 예 |
|---|---|
| 관측 가능한 사실인가요? | “횡단보도를 가로질러 이동한다”처럼 전후를 확인할 수 있습니다. |
| 비슷한 조건과 구분되나요? | 젖은 도로와 현재 우천을 별도 질문으로 다룹니다. |
| 음성과 판단 불가가 나뉘나요? | 필요한 장면이 보였는지부터 구분합니다. |
| 시간 정보가 필요한가요? | 끼어들기는 `temporal=true`이며 지시문에도 전후 과정이 필요하다고 씁니다. |
| 실제로 통과 조건에 들어갔나요? | `target`과 결과의 `checked_criteria`를 확인합니다. |
| 강한 주장을 몰래 대신하지 않나요? | 간격 감소를 충돌 직전의 정답이라고 부르지 않습니다. |
| 새 질문에 대한 응답이 있나요? | 모델·가상 예제 모두 모든 질문에 답해야 합니다. |

## 10. 현재 설정만으로 할 수 없는 것

조건별로 다른 점수 임계값, 다중 선택, 복잡한 AND/OR 식, 신호등 인식의 보장, 새 질문에 대한 자동 정답 생성, 모델 재학습은 구현되어 있지 않습니다. JSON에 새 필드를 적는 것만으로 이런 기능이 생기지 않습니다.

특히 `schema_version=1`은 완전한 JSON Schema 검증기를 뜻하지 않습니다. `load_rubric()`는 일부 필수 조건과 수치를 검사하지만 모든 오타와 참조 오류를 사전에 잡지는 않습니다. 기본 구조를 유지하고, 전용 예제·실제 소량 영상·사람 검수의 순서로 확인하세요.

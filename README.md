# Decision: 자율주행 영상의 의미 재분류 스터디

**생성할 때 요청한 장면과 완성된 영상에서 실제로 보이는 장면을 구분하는 한국어 실습 프로젝트입니다.**

“비 오는 밤에 차량이 끼어드는 영상”을 요청했더라도 비와 야간 조명만 있고 끼어드는 행동은 없을 수 있습니다. 이 프로젝트는 날씨·조명·사건·자료 품질을 나눠 기록하고, 재분류·사람 검수·생성 조건 수정을 돕습니다.

[실습 노트북](notebooks/S1_Semantic_Video_Verification.ipynb) · [기술 설명서](docs/ko/SEMANTIC_VIDEO_GUIDE.md) · [권장 판별 조건](docs/ko/SEMANTIC_CRITERIA.md) · [검증 기록](docs/ko/SEMANTIC_VALIDATION.md)

## 이론부터 조건 작성까지 읽기

새 [한국어 문서 모음](docs/ko/README.md)에 학습 순서를 정리했습니다.

| 알고 싶은 내용 | 문서 |
|---|---|
| Decision 모델·확률·보류·시간 정보의 의미 | [이론 설명서](docs/ko/THEORY.md) |
| 함수·입력·모델·보고서가 연결되는 구조 | [기술 구조 설명서](docs/ko/TECHNICAL_DESIGN.md) |
| 새 질문과 목표를 JSON에 넣는 방법 | [판별 조건 작성 안내](docs/ko/CRITERIA_AUTHORING.md) |
| 임계값에 따라 분기가 달라지는 과정 | [조작형 분기 실험](docs/assets/decision-explorer.html) |

문서에는 SVG 그림과 Mermaid 도식을 넣었습니다. 분기 실험은 저장소를 내려받은 뒤 HTML 파일을 브라우저에서 열면 동작합니다. 모든 숫자는 가상 관측의 예이며 실제 모델 성능이 아닙니다.

## 무엇을 할 수 있나요?

- 날씨, 조명, 끼어들기, 보행자 횡단, 선행차 감속 단서, 간격 감소, 장면 연속성, 관측 가능성을 따로 기록합니다.
- 생성 목표와 관측 라벨을 비교해 학습 후보·재분류 후보·판단 보류·생성 결함 검토로 나눕니다.
- 검수용 CSV, 판정 근거, 모델·기준 버전, 제한된 재생성 수정 제안을 저장합니다.
- 기존 Transfer/Predict 결과의 `manifest.json`을 읽거나 직접 작성한 입력 목록을 사용합니다.

CPU 기본 예제는 **가상 관측값으로 처리 절차를 배우는 실습**입니다. 실제 영상 인식에는 별도로 준비한 Valen 모델 또는 멀티이미지 입력을 지원하는 Cosmos HTTP 서버가 필요합니다. GPU 정확도·속도는 아직 검증하지 않았습니다.

## 1. 빠르게 시작하기

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

## 4. 폴더 구성

```text
Decision/
├── README.md                       # 시작 안내
├── pyproject.toml                  # 설치·선택 패키지·명령행 설정
├── LICENSE / NOTICE                # 기존 라이선스와 출처
├── CONTRIBUTING.md                 # 수정·검증 방법
├── src/av_decision/                 # 독립 실행 패키지
│   ├── semantic.py                 # 판정·입력 변환·검수·평가
│   ├── semantic_providers.py       # 프레임 추출·모델 연결
│   ├── semantic_cli.py             # 명령행 도구
│   └── configs/criteria.ko.json    # 실제로 사용하는 판별 기준
├── notebooks/                      # 한국어 S1 실습
├── docs/                           # 한국어 설명서와 시각화 자료
├── examples/                       # 새로운 판별 조건 JSON과 실행 예제
├── tests/                          # 의미 판정과 어댑터 계약 테스트
├── scripts/                        # 문서·노트북 검사 도구
└── .github/                        # 자동 테스트와 이슈·PR 양식
```

기준 파일은 패키지에 포함됩니다. 설치 후 다른 폴더에서 실행해도 기본 기준을 찾을 수 있습니다. 기준을 수정하는 경우 `rubric_id`도 바꾸고, 비교할 실행에 `--rubric /경로/기준.json`을 명시할 수 있습니다.

## 5. 테스트와 GitHub 게시 준비

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

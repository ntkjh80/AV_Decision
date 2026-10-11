# 수정하고 함께 검증하는 방법

이 프로젝트는 한국어로 판별 기준을 이해하고 실험하는 스터디 자료입니다. 설명과 주석은 가능한 한 쉬운 한국어로 작성해 주세요. 코드의 식별자와 모델 API 이름은 원래 표기를 유지합니다.

## 개발 환경

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[notebook,dev]'
```

CPU 테스트에는 GPU·영상·API 키가 필요하지 않습니다. 실제 모델을 바꾸는 실험은 별도 환경에서 수행하고, 모델 버전과 실행한 범위를 기록합니다.

## 변경할 때 지켜야 할 의미

- 생성 목표 `target`과 모델 관측 `observed`를 구분합니다.
- 판단할 수 없는 경우는 `unknown`으로 남깁니다.
- 원본 주행과 변형 영상은 같은 데이터 분할을 유지합니다.
- 처리 실패도 보고서와 검수 목록에 남깁니다.
- 실제 추론과 가상 예제를 혼동할 만한 출력이나 설명을 만들지 않습니다.
- 판별 기준을 바꾸면 `rubric_id`와 한국어 기준 설명도 함께 바꿉니다.

## 확인할 명령

```bash
python -m unittest discover -s tests -v
python scripts/check_repository.py
python scripts/verify_notebook.py
python -m build
```

`check_repository.py`는 Python 구문, 로컬 문서 링크, 노트북 출력 제거 상태를 검사합니다. `verify_notebook.py`는 기본 CPU 셀을 실제 Jupyter 커널로 실행하되 원본 노트북에 출력을 덮어쓰지 않습니다. 이 검사는 GPU 모델의 정확도를 측정하지 않습니다.

노트북을 직접 실행했다면 Jupyter에서 전체 출력 지우기(Clear All Outputs)를 하고 저장한 뒤 커밋하세요. 개인 경로와 영상이 노트북 출력에 포함되지 않도록 하기 위한 절차입니다.

PR에는 무엇을 바꿨는지, 왜 필요한지, 어떤 확인을 했는지 적어 주세요. 판별 결과가 달라지는 수정은 해당 반례를 다루는 테스트를 함께 추가하면 좋습니다. 모델 성능을 주장할 때는 영상 수, 정답 작성 방법, 분할, 보류율도 함께 제시해 주세요.

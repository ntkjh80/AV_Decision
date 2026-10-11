# GitHub에 올리는 순서

이 폴더는 자체 `pyproject.toml`, 코드, 노트북, 문서, 테스트를 가진 독립 프로젝트입니다. `Decision` 폴더 자체를 저장소 루트로 올리면 됩니다. 상위의 다른 스터디 폴더를 포함할 필요는 없습니다.

## 1. 올릴 내용 확인하기

코드와 한국어 설명, 기준 파일, 테스트, GitHub 설정을 올립니다. 모델 가중치·실제 영상·실행 결과·인증정보는 `.gitignore`에서 제외합니다. `.gitignore`는 이미 추적 중인 파일을 자동으로 제거하지는 않으므로 첫 커밋 전에 실제 목록을 확인하세요.

노트북은 설명과 코드만 저장하며 출력은 비웁니다. 실행했던 개인 경로나 영상이 GitHub 미리보기에 남는 것을 피할 수 있습니다.

```bash
python scripts/check_repository.py
python -m unittest discover -s tests -v
```

## 2. 로컬 저장소 준비하기

`Decision` 폴더에서 실행합니다. 이미 초기화되어 있다면 `git init`은 생략해도 됩니다.

```bash
git init -b main
git status --short
git add .
git diff --cached --stat
git diff --cached --name-only
git commit -m "자율주행 영상 의미 재분류 스터디 초기 구성"
```

Git 작성자 정보가 없다면 본인의 이름과 이메일을 이 저장소에 설정한 뒤 커밋합니다. 다른 사람이 사용하는 전역 설정을 임의로 바꿀 필요는 없습니다.

## 3. GitHub 원격 저장소 연결하기

GitHub에서 빈 저장소를 하나 만듭니다. README·라이선스·gitignore는 이 폴더에 있으므로 원격에서 새로 생성하지 않아도 됩니다. 저장소 공개 범위는 용도에 맞게 선택하세요.

아래 URL의 계정과 저장소 이름을 실제 값으로 바꿉니다.

```bash
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

이미 `origin`이 있다면 `git remote -v`로 주소부터 확인합니다. 기존 원격 주소나 이력을 강제로 덮어쓰지 마세요. 이 자료를 준비하는 작업에서는 원격 저장소 생성이나 push를 실행하지 않았습니다.

## 4. 업로드 후 확인하기

- README에서 실습 노트북·기술 설명서·판별 조건 링크가 열리는지 확인합니다.
- Actions의 **CPU 검사**와 **노트북과 배포 패키지 검사** 작업 결과를 확인합니다.
- 다른 폴더에 새로 clone하고 기본 설치 후 `av-decision demo`를 실행해 봅니다.

자동 검사는 모델 가중치나 실제 영상을 내려받지 않습니다. API 키를 GitHub에 등록할 필요도 없습니다. GPU 추론 검증은 별도 환경에서 수행하고 결과를 [검증 기록](SEMANTIC_VALIDATION.md)에 추가하세요.

## 5. 버전 배포 파일이 필요할 때

```bash
python -m pip install -e '.[dev]'
python -m build
```

`dist/`에 wheel과 소스 배포 파일이 생깁니다. 이 명령은 PyPI나 GitHub Releases에 자동 게시하지 않습니다. 설치용 wheel에는 실행 코드와 기본 기준 파일이, 소스 배포본에는 노트북·문서·테스트도 포함됩니다.

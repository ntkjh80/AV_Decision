"""모델 선택을 판별 조건과 분리합니다. JSON에는 실행 코드나 인증키를 넣지 않습니다.

CLI와 노트북이 같은 생성 함수를 사용하므로 모델을 바꿔도 판정·보고서 코드는
그대로 유지됩니다. 전혀 다른 API의 모델은 ObservationProvider 계약에 맞춰
별도 연결 클래스를 작성하고 run_batch에 전달할 수 있습니다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .semantic import finite_number, read_json
from .semantic_providers import CosmosHTTPProvider, ValenProvider, VLMHTTPProvider


class ObservationProvider(Protocol):
    """공통 계약: 모델 정보와 (답변, 관측 근거)를 제공합니다.

    answers의 모든 질문 ID·선택지·확률은 semantic.normalize_answers가 검사합니다.
    확률을 실제로 제공하지 않는 모델은 probabilities=None을 반환해야 합니다.
    """

    metadata: dict[str, Any]

    def observe(self, record: dict, rubric: dict, evidence_dir: Path) -> tuple[dict, dict]:
        ...


PROVIDER_NAMES = ('valen', 'cosmos-http', 'vlm-http')
SAMPLING_DEFAULTS = {'sample_fps': 4.0, 'max_frames': 32, 'max_duration_s': 8.0}


def _text(config, key):
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'모델 설정의 {key}에는 비어 있지 않은 문자열이 필요합니다.')
    return value


def create_provider(config: dict, *, base_dir=None) -> ObservationProvider:
    """설정을 검사하고 연결 객체만 만듭니다. 이 단계에서는 추론을 실행하지 않습니다.

    상대 model_dir는 base_dir 기준이며, 생략하면 현재 작업 폴더 기준입니다.
    알 수 없는 키를 거부하여 오타가 기본값으로 조용히 대체되는 일을 막습니다.
    """
    if not isinstance(config, dict) or type(config.get('schema_version')) is not int or config['schema_version'] != 1:
        raise ValueError('모델 설정은 schema_version=1인 JSON 객체여야 합니다.')
    name = _text(config, 'provider')
    if name not in PROVIDER_NAMES:
        raise ValueError(f'지원하는 provider: {", ".join(PROVIDER_NAMES)}')
    specific = {'model_dir', 'device'} if name == 'valen' else {'endpoint', 'model', 'timeout_s'}
    unknown = set(config) - {'schema_version', 'provider', 'revision', 'sampling'} - specific
    if unknown:
        raise ValueError(f'지원하지 않는 모델 설정 키: {", ".join(sorted(unknown))}')
    revision = _text(config, 'revision')
    supplied = config.get('sampling', {})
    if not isinstance(supplied, dict) or set(supplied) - set(SAMPLING_DEFAULTS):
        raise ValueError('sampling에는 sample_fps, max_frames, max_duration_s만 넣을 수 있습니다.')
    sampling = {**SAMPLING_DEFAULTS, **supplied}
    for key in ('sample_fps', 'max_duration_s'):
        if not finite_number(sampling[key]) or sampling[key] <= 0:
            raise ValueError(f'{key}는 양수여야 합니다.')
    if type(sampling['max_frames']) is not int or sampling['max_frames'] < 2:
        raise ValueError('max_frames는 2 이상의 정수여야 합니다.')
    if name == 'valen':
        model_dir = Path(_text(config, 'model_dir')).expanduser()
        if not model_dir.is_absolute():
            model_dir = Path(base_dir or Path.cwd()) / model_dir
        device = _text(config, 'device') if 'device' in config else 'cuda'
        provider = ValenProvider(model_dir, revision, device, **sampling)
    else:
        timeout = config.get('timeout_s', 180)
        if not finite_number(timeout) or timeout <= 0:
            raise ValueError('timeout_s는 양수여야 합니다.')
        cls = CosmosHTTPProvider if name == 'cosmos-http' else VLMHTTPProvider
        provider = cls(_text(config, 'endpoint'), _text(config, 'model'), revision,
                       timeout_s=timeout, **sampling)
    # 보고서에는 모델 식별 정보와 입력 샘플링 조건만 추가합니다. 인증키는 기록하지 않습니다.
    provider.metadata['sampling'] = dict(sampling)
    return provider


def load_provider(path) -> ObservationProvider:
    """JSON 파일로 모델을 선택합니다. 상대 모델 경로는 설정 파일 폴더 기준입니다."""
    return create_provider(load_provider_config(path))


def load_provider_config(path):
    """여러 프로세스로 보내기 전에 상대 모델 경로를 절대 경로로 고정합니다."""
    path = Path(path).expanduser().resolve()
    config = read_json(path)
    if isinstance(config, dict) and isinstance(config.get('model_dir'), str):
        model = Path(config['model_dir']).expanduser()
        config['model_dir'] = str((path.parent / model).resolve())
    return config

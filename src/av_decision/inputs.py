"""지정한 파일·폴더를 영상과 이미지 입력 목록으로 만듭니다. 모델은 호출하지 않습니다."""
from pathlib import Path

from .semantic import finite_number, validate_record

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}


def prepare_inputs(input_path, rubric, *, source_group, split='unassigned', target=None,
                   source_observation_hz=None, start_s=0.0, end_s=None):
    """폴더를 재귀 탐색하되 그 밖을 가리키는 심볼릭 링크는 입력으로 삼지 않습니다.

    source_group은 원본 주행/수집 묶음 단위로 지정합니다. 서로 다른 원본 그룹은
    따로 준비한 뒤 JSON 목록을 합칠 수 있습니다. 인코딩 fps에서 실제 관측 주기를
    추정하지 않으며, 모르면 source_observation_hz를 비워 둡니다.
    """
    root = Path(input_path).expanduser().resolve()
    if not root.exists():
        raise ValueError('입력 경로가 없습니다.')
    if not finite_number(start_s) or start_s < 0 or (end_s is not None and (not finite_number(end_s) or end_s <= start_s)):
        raise ValueError('분석 시작·종료 시각을 확인해 주세요.')
    if source_observation_hz is not None and (not finite_number(source_observation_hz) or source_observation_hz <= 0):
        raise ValueError('원본 관측 주기는 양수 또는 미지정이어야 합니다.')
    base = root.parent if root.is_file() else root
    candidates = [root] if root.is_file() else sorted(root.rglob('*'))
    records = []
    import hashlib
    for path in candidates:
        suffix = path.suffix.lower()
        if not path.is_file() or suffix not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
            continue
        if not path.resolve().is_relative_to(base):
            raise ValueError('입력 폴더 밖으로 연결된 미디어 파일이 있습니다.')
        image = suffix in IMAGE_EXTENSIONS
        # 원본 그룹과 상대 경로로 식별합니다. 같은 파일 이름도 그룹이 다르면 구분합니다.
        identity = str(source_group) + '\0' + path.relative_to(base).as_posix()
        row = {'clip_id': 'media-' + hashlib.sha256(identity.encode()).hexdigest()[:20],
               'source_group': source_group, 'split': split, 'media_type': 'image' if image else 'video',
               'image_path' if image else 'video_path': str(path.resolve()),
               'target': {} if target is None else dict(target), 'attempt': 1}
        if not image:
            row.update(source_observation_hz=source_observation_hz,
                       window={'start_s': start_s, 'end_s': end_s})
        validate_record(row, rubric)
        records.append(row)
    if not records:
        raise ValueError('지원하는 영상 또는 이미지 파일이 없습니다.')
    return records

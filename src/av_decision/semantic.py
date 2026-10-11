"""영상 의미 재검증의 공통 규칙입니다. 이 파일 자체는 영상을 이해하지 않습니다.

모델은 관측 결과를 만들고, 여기서는 응답을 검증한 다음 생성 목표와 비교합니다.
생성 목표를 모델 입력에 섞지 않아 '요청한 대로 보인다'는 편향을 줄입니다.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import time


DEFAULT_RUBRIC = Path(__file__).resolve().parent / 'configs/criteria.ko.json'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    """NaN을 JSON에 흘려보내지 않고, 같은 경로의 기존 결과도 덮어쓰지 않습니다."""
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(data)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_rubric(path=DEFAULT_RUBRIC):
    rubric = read_json(path)
    if rubric.get('schema_version') != 1 or not rubric.get('questions'):
        raise ValueError('판별 기준의 버전 또는 질문 목록을 확인해 주세요.')
    for key, q in rubric['questions'].items():
        if not q.get('instructions') or 'unknown' not in q.get('criteria', {}):
            raise ValueError(f'{key}: 설명과 unknown 선택지가 필요합니다.')
    t = rubric['thresholds']
    for key in ('min_probability', 'min_margin'):
        if not finite_number(t[key]) or not 0 <= t[key] <= 1:
            raise ValueError(f'{key}: 0~1 사이의 값이 필요합니다.')
    if not finite_number(t['max_gap_s']) or t['max_gap_s'] <= 0:
        raise ValueError('허용 관측 간격은 양수여야 합니다.')
    if type(t['max_attempts']) is not int or t['max_attempts'] < 1:
        raise ValueError('최대 생성 횟수는 양의 정수여야 합니다.')
    return rubric


def questions_for_model(rubric):
    """모든 영상에 같은 질문을 줍니다. 목표 라벨과 원본 파일명은 전달하지 않습니다."""
    return {key: {'type': 'choice', 'instructions': q['instructions'],
                  'criteria': q['criteria']} for key, q in rubric['questions'].items()}


def normalize_answers(raw, rubric):
    """모델의 잘못된 형식을 조용히 보정하지 않습니다. 누락·NaN·가짜 확률은 실패입니다.

    Valen은 선택지별 확률을 반환합니다. Cosmos의 자유 생성 JSON에는 확률이 없을 수
    있습니다. 이 경우 확률을 만들어 넣지 않고 None으로 남깁니다.
    """
    answers = raw.get('answers', raw) if isinstance(raw, dict) else None
    if not isinstance(answers, dict) or set(answers) != set(rubric['questions']):
        raise ValueError('모델 응답은 모든 질문을 정확히 한 번씩 포함해야 합니다.')
    normalized = {}
    for key, q in rubric['questions'].items():
        answer = answers[key]
        if not isinstance(answer, dict):
            raise ValueError(f'{key}: 답변은 객체여야 합니다.')
        choice, probs = answer.get('choice'), answer.get('probabilities')
        if choice not in q['criteria']:
            raise ValueError(f'{key}: 정의하지 않은 선택지입니다.')
        confidence = margin = None
        if probs is not None:
            if not isinstance(probs, dict) or set(probs) != set(q['criteria']):
                raise ValueError(f'{key}: 선택지 전체의 확률이 필요합니다.')
            values = list(probs.values())
            if any(not finite_number(v) or not 0 <= v <= 1 for v in values):
                raise ValueError(f'{key}: 확률에 유효하지 않은 수가 있습니다.')
            if not math.isclose(sum(values), 1, abs_tol=1e-5):
                raise ValueError(f'{key}: 확률 합이 1이 아닙니다.')
            if probs[choice] < max(values) - 1e-8:
                raise ValueError(f'{key}: 선택한 답과 최대 확률이 다릅니다.')
            ordered = sorted(values, reverse=True)
            confidence, margin = probs[choice], ordered[0] - ordered[1]
        normalized[key] = {'choice': choice, 'probabilities': probs,
                           'confidence': confidence, 'margin': margin}
    return normalized


def validate_record(record, rubric):
    for key in ('clip_id', 'source_group', 'split'):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f'{key}가 필요합니다.')
    if record['split'] not in ('train', 'validation', 'test', 'unassigned'):
        raise ValueError('split은 train/validation/test/unassigned 중 하나입니다.')
    if record.get('target_needs_review'):
        raise ValueError('이 입력의 목표를 사람이 검토한 뒤 target_needs_review를 false로 바꿔 주세요.')
    if not isinstance(record.get('target'), dict) or not record['target']:
        raise ValueError('생성 목표 target을 확인해 주세요. 알 수 없는 목표는 직접 작성합니다.')
    for key, choice in record['target'].items():
        if key not in rubric['questions'] or choice not in rubric['questions'][key]['criteria'] or choice == 'unknown':
            raise ValueError(f'목표 {key}={choice}를 판별 기준과 맞춰 주세요.')
    hz = record.get('source_observation_hz')
    if hz is not None and (not finite_number(hz) or hz <= 0):
        raise ValueError('원본 관측 주기는 양수 또는 null이어야 합니다.')
    if type(record.get('attempt', 1)) is not int or record.get('attempt', 1) < 1:
        raise ValueError('attempt는 현재까지 생성한 횟수이며 1부터 시작합니다.')


def judge(record, raw_answers, evidence, rubric):
    """사실 재분류 → 목표 비교 → 처리 분기의 순서로 판정합니다.

    candidate는 학습 후보라는 뜻입니다. 검수 완료나 물리적으로 안전하다는 뜻이 아닙니다.
    원본 2Hz 영상을 10fps로 저장했다고 사건의 관측 해상도가 높아지지는 않습니다.
    """
    validate_record(record, rubric)
    answers = normalize_answers(raw_answers, rubric)
    t = rubric['thresholds']
    findings, mismatches = [], []
    checked = set(record['target']) | {'continuity', 'visibility'}
    for key in sorted(checked):
        answer = answers[key]
        if answer['choice'] == 'unknown':
            findings.append(f'{key}:판단_불가')
        if answer['confidence'] is None:
            findings.append(f'{key}:확률_없음_검수_필요')
        elif answer['confidence'] < t['min_probability'] or answer['margin'] < t['min_margin']:
            findings.append(f'{key}:낮은_확신_또는_선택지_경합')
        if rubric['questions'][key]['temporal']:
            gap = evidence.get('max_gap_s')
            hz = record.get('source_observation_hz')
            if not finite_number(gap) or gap < 0 or gap > t['max_gap_s']:
                findings.append(f'{key}:샘플링_간격_부족')
            if hz is None or 1 / hz > t['max_gap_s']:
                findings.append(f'{key}:원본_시간해상도_미확인_또는_부족')
        if key in record['target'] and answer['choice'] not in ('unknown', record['target'][key]):
            mismatches.append(key)
    # limited는 결함이 아닙니다. 어려운 장면을 별도로 보존합니다.
    artifact = answers['continuity']['choice'] == 'artifact'
    disposition = ('review' if findings else 'quarantine' if artifact else
                   'relabel_candidate' if mismatches else 'candidate')
    return {'clip_id': record['clip_id'], 'source_group': record['source_group'],
            'split': record['split'], 'target': record['target'], 'observed': answers,
            'checked_criteria': sorted(checked),
            'mismatched_criteria': mismatches, 'review_reasons': findings,
            'disposition': disposition, 'requires_human_approval': True,
            'difficult_scene': answers['visibility']['choice'] != 'clear',
            'evidence': evidence, 'rubric_id': rubric['rubric_id'],
            'rubric_sha256': fingerprint(rubric), 'attempt': record.get('attempt', 1)}


def regeneration_plan(record, result, rubric):
    """다음 생성에 쓸 제안만 만듭니다. 유료 생성 작업을 자동으로 실행하지 않습니다."""
    allowed = (result['disposition'] == 'relabel_candidate'
               and record.get('attempt', 1) < rubric['thresholds']['max_attempts']
               and record['split'] not in ('test', 'unassigned'))
    return {'clip_id': record['clip_id'], 'parent_source_group': record['source_group'],
            'split': record['split'], 'ready_for_generation': allowed,
            'next_attempt': record.get('attempt', 1) + 1,
            'corrections': [rubric['repair_hints'][k] for k in result['mismatched_criteria']
                            if k in rubric['repair_hints']],
            'target': record['target'], 'status': 'prepared_only' if allowed else 'held',
            'note': '기존 프롬프트와 함께 검토할 수정 제안입니다. 새 영상도 다시 판정합니다. 시험 자료에는 피드백 생성하지 않습니다.'}


def import_manifest(path, video_root, rubric, source_group=None, split='unassigned'):
    """구버전 M4/M5와 현재 M5/M6의 results 목록을 같은 계약으로 읽습니다.

    모듈 번호 대신 condition/scenario 필드를 사용합니다. 알려진 이름만 매핑하며,
    처음 보는 이름을 문자열 포함 검사로 추측하지 않습니다.
    """
    manifest, base = read_json(path), Path(video_root).resolve()
    records = []
    for index, row in enumerate(manifest.get('results', [])):
        rel = Path(row['generated_video'])
        video = (base / rel).resolve()
        if not video.is_relative_to(base) or not video.is_file():
            raise ValueError(f'영상 경로가 없거나 지정 폴더 밖입니다: {rel}')
        requested = row.get('condition') or row.get('scenario')
        target = copy.deepcopy(rubric['target_presets'].get(requested, {}))
        # 생성 제목은 출처로 남기되 모델에는 보내지 않습니다.
        records.append({'clip_id': f'clip-{index:04d}-{file_hash(video)[:10]}',
                        'video_path': str(video), 'target': target,
                        'requested_name': requested, 'original_prompt': row.get('prompt'),
                        'source_group': source_group or 'manifest-' + file_hash(path)[:12],
                        'split': split, 'source_observation_hz': None, 'attempt': 1,
                        'window': {'start_s': 0.0, 'end_s': None},
                        'source_manifest': str(Path(path).resolve()),
                        'target_needs_review': not bool(target)})
    if not records:
        raise ValueError('manifest의 results 목록에 생성 영상이 없습니다.')
    return records


def check_group_splits(records):
    seen = {}
    for row in records:
        group, split = row['source_group'], row['split']
        if group in seen and seen[group] != split:
            raise ValueError(f'{group}: 원본과 변형 영상이 서로 다른 데이터 분할에 있습니다.')
        seen[group] = split


def run_batch(records, provider, rubric, output_dir):
    """실패 영상도 review로 기록합니다. 일부 실패를 성공률 계산에서 숨기지 않습니다."""
    check_group_splits(records)
    ids = [r['clip_id'] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError('clip_id가 중복됩니다.')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    results, plans = [], []
    for i, record in enumerate(records):
        started = time.perf_counter()
        try:
            validate_record(record, rubric)
            raw, evidence = provider.observe(record, rubric, output / f'evidence-{i:04d}')
            result = judge(record, raw, evidence, rubric)
            write_json(output / f'raw-{i:04d}.json', raw)
            plans.append(regeneration_plan(record, result, rubric))
        except Exception as exc:
            # 모델·통신·디코더의 예외도 영상별로 남깁니다. KeyboardInterrupt는 잡지 않습니다.
            result = {'clip_id': record['clip_id'], 'source_group': record['source_group'],
                      'split': record['split'], 'target': record.get('target', {}),
                      'disposition': 'review', 'requires_human_approval': True,
                      'error_type': type(exc).__name__, 'error': str(exc)}
        result['elapsed_s'] = round(time.perf_counter() - started, 6)
        result['provider'] = provider.metadata
        results.append(result)
    report = {'schema_version': 1, 'rubric': rubric, 'provider': provider.metadata,
              'counts': {s: sum(r['disposition'] == s for r in results)
                         for s in ('candidate', 'relabel_candidate', 'review', 'quarantine')},
              'results': results}
    write_json(output / 'records.json', records)
    write_json(output / 'report.json', report)
    write_json(output / 'regeneration_plans.json', plans)
    # 검수용 CSV는 사람이 복사본을 작성합니다. 원본 정답을 모델이 스스로 채우지 않습니다.
    import csv
    with (output / 'human_review.csv').open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['clip_id', 'criterion', 'model_choice', 'human_choice', 'reviewer', 'note', 'disposition', 'error'])
        for result in results:
            for key in rubric['questions']:
                answer = result.get('observed', {}).get(key, {})
                writer.writerow([result['clip_id'], key, answer.get('choice', ''), '', '', '',
                                 result['disposition'], result.get('error', '')])
    return report


def evaluate_human_labels(results, labels, rubric=None):
    """labels는 (clip_id, criterion, human_choice) 목록입니다. 판단 보류도 분모에 남깁니다."""
    rubric = load_rubric() if rubric is None else rubric
    lookup = {r['clip_id']: r for r in results}
    if len(lookup) != len(results):
        raise ValueError('평가 결과의 clip_id가 중복됩니다.')
    seen, per_question = set(), {}
    for row in labels:
        pair = (row['clip_id'], row['criterion'])
        if pair in seen or row['clip_id'] not in lookup:
            raise ValueError('중복 정답 또는 결과에 없는 clip_id입니다.')
        seen.add(pair)
        actual = row['human_choice']
        key = row['criterion']
        if key not in rubric['questions'] or actual not in rubric['questions'][key]['criteria']:
            raise ValueError('사람 정답의 기준 ID와 선택지를 확인해 주세요. 빈 검수 행은 평가에서 제외합니다.')
        result = lookup[row['clip_id']]
        answer = result.get('observed', {}).get(row['criterion'])
        metric = per_question.setdefault(row['criterion'], {'n': 0, 'correct': 0, 'missing': 0,
                     'false_positive': 0, 'false_negative': 0, 'unknown_on_positive': 0,
                     'positive_n': 0, 'negative_n': 0})
        metric['n'] += 1
        pred = answer['choice'] if answer else None
        metric['missing'] += pred is None
        metric['correct'] += pred == actual
        metric['positive_n'] += actual == 'yes'
        metric['negative_n'] += actual == 'no'
        metric['false_positive'] += actual == 'no' and pred == 'yes'
        metric['false_negative'] += actual == 'yes' and pred == 'no'
        metric['unknown_on_positive'] += actual == 'yes' and pred in ('unknown', None)
    for m in per_question.values():
        m['accuracy'] = m['correct'] / m['n']
        m['false_positive_rate'] = m['false_positive'] / m['negative_n'] if m['negative_n'] else None
        m['false_negative_rate'] = m['false_negative'] / m['positive_n'] if m['positive_n'] else None
    return per_question


def demo_data(rubric):
    """명시적인 가상 관측값입니다. 이 수치로 모델 정확도나 속도를 주장하지 않습니다."""
    base = {'weather': 'rain', 'lighting': 'night', 'cut_in': 'yes',
            'pedestrian_crossing': 'no', 'lead_braking': 'no', 'gap_closing': 'no',
            'continuity': 'consistent', 'visibility': 'clear'}
    variants = [('목표일치', {}), ('끼어들기없음', {'cut_in': 'no'}),
                ('가림으로보류', {'cut_in': 'unknown', 'visibility': 'limited'}),
                ('생성결함', {'continuity': 'artifact'}), ('낮은원본주기', {}),
                ('어려워도유효', {'visibility': 'limited'})]
    records, observations = [], {}
    for i, (name, edits) in enumerate(variants):
        clip_id = f'demo-{i+1}-{name}'
        record = {'clip_id': clip_id, 'source_group': f'demo-drive-{i}', 'split': 'validation',
                  'target': {'weather': 'rain', 'lighting': 'night', 'cut_in': 'yes'},
                  'source_observation_hz': 2 if i == 4 else 10, 'attempt': 1}
        answers = {}
        for key, choice in dict(base, **edits).items():
            options = rubric['questions'][key]['criteria']
            probs = {option: (0.94 if option == choice else 0.06 / (len(options)-1)) for option in options}
            answers[key] = {'choice': choice, 'probabilities': probs}
        records.append(record)
        observations[clip_id] = {'answers': answers}
    return records, observations


class FixtureProvider:
    """CPU 학습용입니다. 영상 추론을 했다고 표시하지 않습니다."""
    metadata = {'name': 'fixture', 'revision': 'ko-demo-v1', 'real_inference': False}
    def __init__(self, observations):
        self.observations = observations
    def observe(self, record, rubric, evidence_dir):
        return self.observations[record['clip_id']], {'source_kind': 'synthetic_observations',
                  'window_start_s': 0, 'window_end_s': 4, 'max_gap_s': 0.1,
                  'note': '예제 관측값이며 실제 영상을 분석하지 않았습니다.'}

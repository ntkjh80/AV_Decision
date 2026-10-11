"""확신도 구간을 먼저 나누고, 그 안에서 상황·검수 상태별로 원본을 복사합니다.

선택지 확률은 보정된 정답률이나 임베딩 유사도가 아닙니다. 점수가 없거나
unknown인 자료는 별도 구간에 남기고, 복사 실패도 보고서에서 빠뜨리지 않습니다.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import re
from urllib.parse import quote

from .semantic import file_hash, finite_number, media_type, read_json, write_json

DEFAULT_CLASSIFICATION = Path(__file__).parent / 'configs/classification.ko.json'
BAND_NAMES = {'high': '높은 확신도', 'medium': '중간 확신도', 'low': '낮은 확신도',
              'unknown': '판단 불가', 'unscored': '점수 없음', 'error': '처리 실패'}
STATUS_NAMES = {'candidate': '학습 후보', 'relabel_candidate': '재분류 후보',
                'review': '검수 대기', 'quarantine': '결함 검토'}


def load_classification(path=DEFAULT_CLASSIFICATION):
    return read_json(path)


def validate_classification(config, rubric):
    keys = {'schema_version', 'high_min_probability', 'medium_min_probability', 'scene_criteria'}
    if not isinstance(config, dict) or set(config) != keys or type(config['schema_version']) is not int or config['schema_version'] != 1:
        raise ValueError('분류 설정의 버전과 필드를 확인해 주세요.')
    high, medium = config['high_min_probability'], config['medium_min_probability']
    if not all(finite_number(v) for v in (high, medium)) or not 0 <= medium < high <= 1:
        raise ValueError('분류 경계는 0 ≤ 중간 경계 < 높은 경계 ≤ 1이어야 합니다.')
    scenes = config['scene_criteria']
    if not isinstance(scenes, list) or any(not isinstance(k, str) for k in scenes) or len(scenes) != len(set(scenes)):
        raise ValueError('scene_criteria는 중복 없는 질문 ID 목록이어야 합니다.')
    if set(scenes) - set(rubric['questions']):
        raise ValueError('상황 분류의 질문 ID를 판별 기준에 추가해 주세요.')


def confidence_summary(result, config):
    """검사 항목의 최저 선택 확률을 대표 점수로 씁니다. 결합 정답 확률은 아닙니다."""
    keys = result.get('checked_criteria', [])
    answers = [result.get('observed', {}).get(k, {}) for k in keys]
    values = [a.get('confidence') for a in answers]
    margins = [a.get('margin') for a in answers]
    complete = bool(values) and all(finite_number(v) for v in values)
    score = min(values) if complete else None
    if 'error' in result:
        band = 'error'
    elif any(a.get('choice') == 'unknown' for a in answers):
        band = 'unknown'
    elif not complete:
        band = 'unscored'
    elif score >= config['high_min_probability']:
        band = 'high'
    elif score >= config['medium_min_probability']:
        band = 'medium'
    else:
        band = 'low'
    return {'band': band, 'min_selected_probability': score,
            'min_margin': min(margins) if margins and all(finite_number(v) for v in margins) else None,
            'criteria': keys, 'aggregation': 'minimum_of_checked_selected_probabilities',
            'score_kind': result['provider'].get('probability_kind', 'not_available'),
            'calibrated_correctness_probability': False}


def safe_component(value):
    """라벨 원문을 경로로 실행하지 않습니다. 긴 값·구분자·예약 이름은 해시로 구분합니다."""
    value = str(value)
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}', value) and value.upper() not in reserved:
        return value
    prefix = ''.join(c if c.isalnum() else '-' for c in value)[:24].strip('-') or 'label'
    return prefix + '-' + hashlib.sha256(value.encode()).hexdigest()[:12]


def _csv_value(value):
    if value is None:
        return ''
    if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')):
        return "'" + value
    return value


def _write_csv(path, fields, rows):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _copy_record(index, record, result, output, config):
    summary = result['confidence_summary']
    labels = {k: result.get('observed', {}).get(k, {}).get('choice', 'unavailable')
              for k in config['scene_criteria']}
    parts = ['classified', summary['band']]
    parts += [safe_component(key) + '=' + safe_component(label) for key, label in labels.items()]
    parts += [result['disposition'], safe_component(record['split'])]
    source_value = record.get('image_path') if media_type(record) == 'image' else record.get('video_path')
    item = {'clip_id': record['clip_id'], 'band': summary['band'], 'disposition': result['disposition'],
            'media_type': media_type(record), 'scene_labels': labels,
            'source_path': str(source_value) if source_value is not None else None,
            'source_group': record['source_group'], 'split': record['split'],
            'assessed_window': record.get('window'), 'copy_scope': 'entire_original_file',
            'confidence_summary': summary, 'export_status': 'failed', 'copied_path': None}
    destination = None
    created = False
    try:
        if not isinstance(source_value, str) or not source_value:
            raise ValueError('분류해서 복사할 image_path 또는 video_path가 없습니다.')
        source = Path(source_value).expanduser().resolve()
        if not source.is_file():
            raise ValueError('원본 파일이 없습니다.')
        suffix = source.suffix.lower()
        if suffix not in ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'):
            raise ValueError('복사할 파일의 영상·이미지 확장자를 확인해 주세요.')
        # 원본 파일명이 같거나 clip_id에 경로 문자가 있어도 충돌하지 않습니다.
        filename = f'{index:06d}-' + hashlib.sha256(record['clip_id'].encode()).hexdigest()[:12] + suffix
        relative = Path(*parts) / filename
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with source.open('rb') as reader, destination.open('xb') as writer:
            created = True
            for block in iter(lambda: reader.read(1024 * 1024), b''):
                digest.update(block)
                writer.write(block)
        copied_hash = digest.hexdigest()
        evidence = result.get('evidence', {})
        observed_hash = evidence.get('media_sha256') or evidence.get('video_sha256')
        if observed_hash is not None and observed_hash != copied_hash:
            raise ValueError('추론 후 원본 내용이 바뀌어 분류 복사를 취소했습니다.')
        if file_hash(destination) != copied_hash:
            raise OSError('복사본 해시 검증에 실패했습니다.')
        item.update(export_status='copied', copied_path=relative.as_posix(), source_path=str(source), sha256=copied_hash)
        # 원본 영상 전체를 복사하되 판정한 구간은 반드시 함께 기록합니다.
        write_json(destination.with_suffix(destination.suffix + '.json'), item)
    except Exception as exc:
        if created and destination is not None:
            destination.unlink(missing_ok=True)
            destination.with_suffix(destination.suffix + '.json').unlink(missing_ok=True)
        item.update(export_status='failed', copied_path=None, export_error=str(exc))
    return item


def create_classification_report(records, report, output_dir, classification=None, copy_files=True):
    output = Path(output_dir)
    config = classification if classification is not None else load_classification()
    # 점수 보고만 할 때 기본 상황 항목이 사용자 기준에 없더라도 문제가 되지 않습니다.
    if classification is not None:
        validate_classification(config, report['rubric'])
    items, scores = [], []
    for index, (record, result) in enumerate(zip(records, report['results'])):
        result['confidence_summary'] = confidence_summary(result, config)
        if classification is not None and copy_files:
            item = _copy_record(index, record, result, output, config)
        else:
            item = {'clip_id': record['clip_id'], 'band': result['confidence_summary']['band'],
                    'disposition': result['disposition'], 'media_type': media_type(record),
                    'scene_labels': {key: result.get('observed', {}).get(key, {}).get('choice', 'unavailable')
                                     for key in config['scene_criteria']}, 'split': record['split'], 'source_group': record['source_group'],
                    'confidence_summary': result['confidence_summary'], 'export_status': 'disabled', 'copied_path': None}
        result['classification'] = item
        items.append(item)
        for key in report['rubric']['questions']:
            answer = result.get('observed', {}).get(key, {})
            probs = answer.get('probabilities')
            target = record.get('target', {}).get(key)
            scores.append({'clip_id': record['clip_id'], 'criterion': key, 'choice': answer.get('choice'),
                           'selected_probability': answer.get('confidence'), 'margin': answer.get('margin'),
                           'target_choice': target, 'target_probability': probs.get(target) if probs and target else None,
                           'probabilities_json': json.dumps(probs, ensure_ascii=False),
                           'checked': key in result.get('checked_criteria', []),
                           'score_kind': result['confidence_summary']['score_kind'],
                           'band': item['band'], 'disposition': result['disposition'],
                           'device': result['provider'].get('device'), 'error': result.get('error')})
    bands = {band: sum(item['band'] == band for item in items) for band in BAND_NAMES}
    folders = Counter(str(Path(item['copied_path']).parent) for item in items if item['export_status'] == 'copied')
    report['classification_summary'] = {
        'enabled': classification is not None and copy_files, 'config': config, 'input_count': len(records),
        'confidence_band_counts': bands, 'copied_count': sum(v['export_status'] == 'copied' for v in items),
        'copy_failed_count': sum(v['export_status'] == 'failed' for v in items),
        'folder_counts': dict(sorted(folders.items())),
        'score_note': '대표 점수는 검사 항목의 선택 확률 중 최솟값입니다. 보정된 정답률이나 유사도가 아닙니다.'}
    write_json(output / 'classification.json', {'summary': report['classification_summary'], 'items': items})
    score_fields = ['clip_id', 'criterion', 'choice', 'selected_probability', 'margin', 'target_choice',
                    'target_probability', 'probabilities_json', 'checked', 'score_kind', 'band', 'disposition', 'device', 'error']
    _write_csv(output / 'scores.csv', score_fields, scores)
    _write_csv(output / 'classification.csv', ['clip_id', 'band', 'min_selected_probability', 'min_margin',
               'disposition', 'media_type', 'scene_labels', 'split', 'source_group', 'export_status', 'copied_path', 'export_error'],
               [{**item, **{k: item['confidence_summary'][k] for k in ('min_selected_probability', 'min_margin')},
                 'scene_labels': json.dumps(item['scene_labels'], ensure_ascii=False)} for item in items])
    _render_reports(report, items, output)


def _render_reports(report, items, output):
    summary = report['classification_summary']
    escape = lambda value: html.escape(str(value), quote=True)
    md = lambda value: escape(value).replace('|', '&#124;').replace('\n', ' ')
    number = lambda value: '—' if value is None else f'{value:.4f}'
    copy_note = ('원본을 유지하고 전체 파일을 복사합니다. 영상 판정 구간은 파일 옆 JSON을 확인하세요.'
                 if summary['enabled'] else '이번 실행은 파일 복사를 껐습니다. 점수와 판정 보고서만 저장합니다.')
    lines = ['# 확신도별 분류 결과', '', f"입력 {summary['input_count']}개 · 복사 {summary['copied_count']}개 · 복사 실패 {summary['copy_failed_count']}개", '',
             summary['score_note'], '', copy_note + ' 모든 분류 결과는 사람 검수 전 후보입니다.', '',
             '## 확신도 구간', '', '| 구간 | 개수 |', '|---|---:|']
    for band, count in summary['confidence_band_counts'].items():
        lines.append(f'| {BAND_NAMES[band]} (`{band}`) | {count} |')
    lines += ['', '## 실제 복사된 폴더', '', '| 폴더 | 파일 수 |', '|---|---:|']
    lines += [f'| {md(folder)} | {count} |' for folder, count in summary['folder_counts'].items()]
    lines += ['', '## 파일별 결과', '', '| ID | 확신도 구간 | 최저 선택 확률 | 판정 | 복사 |', '|---|---|---:|---|---|']
    rows = []
    for item, result in zip(items, report['results']):
        score = number(item['confidence_summary']['min_selected_probability'])
        lines.append(f"| {md(item['clip_id'])} | {item['band']} | {score} | {item['disposition']} | {md(item['export_status'])} |")
        link = (f'<a href="{escape(quote(item["copied_path"], safe="/"))}">복사본 열기</a>' if item['copied_path'] else escape(item['export_status']))
        reasons = '; '.join(result.get('review_reasons', [])) or result.get('error', '')
        if item.get('export_error'):
            reasons += ' / 복사: ' + item['export_error']
        labels = ', '.join(f'{key}={value}' for key, value in item['scene_labels'].items()) or '—'
        rows.append(f'<tr><td>{escape(item["clip_id"])}</td><td>{BAND_NAMES[item["band"]]}</td><td class="num">{score}</td>'
                    f'<td>{STATUS_NAMES[result["disposition"]]}</td><td>{escape(labels)}</td><td>{link}</td><td>{escape(reasons)}</td></tr>')
    lines += ['', '상세 선택지별 확률·목표 선택 확률·1·2위 차이는 scores.csv, 원본 경로·복사 실패·폴더 집계는 classification.json에서 확인합니다.', '']
    (output / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    total = max(summary['input_count'], 1)
    bars = ''.join(f'<div class="bar-row"><span>{BAND_NAMES[band]}</span><div class="track"><div class="fill {band}" style="width:{100*count/total:.3f}%"></div></div><b>{count}</b></div>'
                   for band, count in summary['confidence_band_counts'].items())
    folders = ''.join(f'<tr><td>{escape(folder)}</td><td class="num">{count}</td></tr>' for folder, count in summary['folder_counts'].items())
    provider = report['provider']
    real = provider.get('real_inference', False)
    config = summary['config']
    page = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Decision 확신도별 분류 보고서</title><style>
:root{{font-family:system-ui,-apple-system,sans-serif;color:#18243b;background:#f4f6fa}}body{{margin:0}}main{{max-width:1250px;margin:auto;padding:36px 24px}}h1{{font-size:32px;margin:10px 0}}h2{{font-size:20px}}p{{line-height:1.7}}.eyebrow{{color:#246552;font-weight:700;letter-spacing:.08em}}.muted{{color:#53627a}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}}.card,section{{background:#fff;border:1px solid #dfe5ee;border-radius:14px;padding:20px}}.card b{{display:block;font-size:34px;margin-top:6px}}section{{margin:18px 0}}.bar-row{{display:grid;grid-template-columns:110px 1fr 40px;gap:14px;align-items:center;margin:14px 0}}.track{{background:#edf1f7;border-radius:5px;height:18px}}.fill{{background:#4684c7;height:18px;border-radius:5px;min-width:0}}.high{{background:#207d63}}.medium{{background:#4684c7}}.low{{background:#d39a34}}.error{{background:#c85b5b}}.unknown,.unscored{{background:#929daf}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:13px 10px;border-bottom:1px solid #e6eaf1;text-align:left;vertical-align:top;overflow-wrap:anywhere}}th{{color:#53627a}}.num{{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}}a{{color:#236f9a}}.tag{{padding:6px 10px;background:#e9eef5;border-radius:7px;display:inline-block}}@media(max-width:650px){{main{{padding:22px 12px}}.cards{{gap:8px}}.card{{padding:12px}}.card b{{font-size:26px}}h1{{font-size:26px}}}}
</style></head><body><main><div class="eyebrow">DECISION · CLASSIFICATION REPORT</div><h1>확신도별 분류 결과</h1>
<span class="tag">{'실제 추론 경로' if real else '가상 응답 실습 · 실제 모델 추론 아님'}</span>
<p class="muted">높음 ≥ {config['high_min_probability']:.2f} · 중간 ≥ {config['medium_min_probability']:.2f} · 그 미만은 낮음<br>{escape(summary['score_note'])}</p>
<div class="cards"><div class="card">분석 입력<b>{summary['input_count']}</b></div><div class="card">분류 폴더에 복사<b>{summary['copied_count']}</b></div><div class="card">복사 실패<b>{summary['copy_failed_count']}</b></div></div>
<section><h2>확신도 분포</h2>{bars}<p class="muted">점수가 높아도 시간 근거가 부족하면 검수 대기입니다. 판단 불가·점수 없음·처리 실패는 별도 구간에 보관합니다.</p></section>
<section><h2>파일별 점수와 상황</h2><div class="scroll"><table><thead><tr><th>ID</th><th>확신도 구간</th><th>최저 선택 확률</th><th>판정</th><th>상황</th><th>파일</th><th>검토 이유</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section><h2>실제 복사된 폴더</h2><div class="scroll"><table><thead><tr><th>상대 경로</th><th>파일 수</th></tr></thead><tbody>{folders}</tbody></table></div></section>
<p class="muted">상세 점수: <a href="scores.csv">scores.csv</a> · 분류 목록: <a href="classification.csv">classification.csv</a> · 전체 근거: <a href="report.json">report.json</a><br>{escape(copy_note)} 모든 결과는 사람 검수 전 후보입니다.</p></main></body></html>'''
    (output / 'summary.html').write_text(page, encoding='utf-8')


def print_scores(report):
    """CLI에서 결과별 대표 점수와 각 질문의 선택 확률을 바로 확인합니다."""
    for result in report['results']:
        summary = result['confidence_summary']
        score = summary['min_selected_probability']
        shown = '점수 없음' if score is None else f'{score:.4f}'
        print(f"{result['clip_id']} | {summary['band']} | 최저 선택 확률={shown} | {result['disposition']}")
        for key, answer in result.get('observed', {}).items():
            probability = answer['confidence']
            value = '없음' if probability is None else f'{probability:.4f}'
            margin = '없음' if answer['margin'] is None else f"{answer['margin']:.4f}"
            print(f"  {key}={answer['choice']} 선택 확률={value} 차이={margin}")

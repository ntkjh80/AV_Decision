"""색상 이미지와 가상 모델 답으로 확신도별 복사·보고서까지 실습합니다.

색깔은 실제 날씨와 관련이 없습니다. 실제 모델 정확도 예제가 아닙니다.
PNG 파일은 표준 라이브러리로 만들어 CPU 기본 설치에서도 실행할 수 있습니다.
"""
import argparse
import copy
from pathlib import Path
import struct
import zlib

from av_decision.classification import load_classification, print_scores
from av_decision.semantic import FixtureProvider, demo_data, load_rubric, run_batch


def write_color_png(path, color):
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data) & 0xffffffff)
    width, height = 48, 32
    raw = b''.join(b'\x00' + bytes(color) * width for _ in range(height))
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 2, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def main():
    parser = argparse.ArgumentParser(description='가상 확률로 확신도별 폴더와 보고서를 만드는 예제')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    originals = root / 'synthetic-inputs'; originals.mkdir()
    rubric = load_rubric()
    _, fixture = demo_data(rubric)
    base = next(iter(fixture.values()))
    cases = [('high-rain', 'rain', .95), ('high-clear', 'clear', .94),
             ('medium-rain', 'rain', .85), ('medium-fog', 'fog', .82),
             ('low-rain', 'rain', .65), ('unknown', 'unknown', .94),
             ('unscored', 'rain', None), ('error', 'rain', .94)]
    records, observations = [], {}
    for index, (name, choice, probability) in enumerate(cases):
        path = originals / f'{name}.png'
        write_color_png(path, ((index * 31) % 256, 125, 200))
        row = {'clip_id': name, 'source_group': 'synthetic-colors', 'split': 'validation',
               'media_type': 'image', 'image_path': str(path), 'target': {}, 'attempt': 1}
        raw = copy.deepcopy(base)
        options = rubric['questions']['weather']['criteria']
        raw['answers']['weather'] = {'choice': choice, 'probabilities': None if probability is None else
            {key: probability if key == choice else (1 - probability) / (len(options) - 1) for key in options}}
        records.append(row)
        observations[name] = {} if name == 'error' else raw
    report = run_batch(records, FixtureProvider(observations), rubric, root / 'result',
                       classification=load_classification())
    print_scores(report)
    expected = {'high': 2, 'medium': 2, 'low': 1, 'unknown': 1, 'unscored': 1, 'error': 1}
    assert report['classification_summary']['confidence_band_counts'] == expected
    assert report['classification_summary']['copied_count'] == 8
    assert all(Path(row['image_path']).is_file() for row in records)
    print('실제 모델 추론 여부:', report['provider']['real_inference'])
    print('시각 보고서:', root / 'result/summary.html')


if __name__ == '__main__':
    main()

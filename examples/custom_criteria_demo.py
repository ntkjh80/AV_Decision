"""새 질문을 추가하는 완성 예제입니다. 실제 영상이나 모델을 호출하지 않습니다.

저장소에서 패키지를 설치한 뒤 다음처럼 실행합니다.
python examples/custom_criteria_demo.py --out .study-runs/road-wet-001
"""
import argparse
from pathlib import Path

from av_decision.semantic import (
    FixtureProvider, demo_data, load_rubric, run_batch, write_json,
)


def main():
    parser = argparse.ArgumentParser(description='젖은 노면 질문 추가 실습')
    parser.add_argument('--out', required=True, help='새로 만들 결과 폴더')
    args = parser.parse_args()

    # 기본 예제는 여덟 질문에만 답하므로, 새 질문의 가상 응답을 직접 추가합니다.
    rubric = load_rubric(Path(__file__).with_name('criteria.road-wet.ko.json'))
    base_records, base_observations = demo_data(load_rubric())
    records, observations = [], {}
    for index, choice in enumerate(('yes', 'no', 'unknown'), start=1):
        # 가상 원본 주기와 점수입니다. 실제 영상 메타데이터의 대체값이 아닙니다.
        record = dict(base_records[0], clip_id=f'wet-road-demo-{index}',
                      source_group=f'fictional-drive-{index}', split='validation',
                      target={'road_wet': 'yes'})
        raw = {'answers': dict(base_observations[base_records[0]['clip_id']]['answers'])}
        raw['answers']['road_wet'] = {
            'choice': choice,
            'probabilities': {key: 0.94 if key == choice else 0.03
                              for key in rubric['questions']['road_wet']['criteria']},
        }
        records.append(record)
        observations[record['clip_id']] = raw

    report = run_batch(records, FixtureProvider(observations), rubric, args.out)
    write_json(Path(args.out) / 'criteria.ko.json', rubric)
    expected = {'candidate': 1, 'relabel_candidate': 1, 'review': 1, 'quarantine': 0}
    if report['counts'] != expected:
        raise RuntimeError('예상과 다른 분기입니다. 보고서와 기준 변경을 확인해 주세요.')
    for row in report['results']:
        if 'road_wet' not in row['checked_criteria']:
            raise RuntimeError('새 질문이 목표 비교에 포함되지 않았습니다.')
        print(row['clip_id'], row['observed']['road_wet']['choice'], row['disposition'])
    print('가상 관측에 대한 분기:', report['counts'])
    print('실제 모델 추론 여부:', report['provider']['real_inference'])
    print('결과 폴더:', Path(args.out))


if __name__ == '__main__':
    main()

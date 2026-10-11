"""노트북과 같은 함수를 사용하는 명령행 진입점입니다."""
import argparse
from pathlib import Path

from .semantic import (DEFAULT_RUBRIC, FixtureProvider, demo_data, import_manifest,
                       load_rubric, read_json, run_batch, write_json)


def main():
    parser = argparse.ArgumentParser(description='자율주행 영상의 의미 재검증 스터디')
    parser.add_argument('--rubric', default=str(DEFAULT_RUBRIC))
    sub = parser.add_subparsers(dest='command', required=True)
    demo = sub.add_parser('demo', help='가상 관측값으로 판정 흐름만 실습합니다.')
    demo.add_argument('--out', required=True)
    ingest = sub.add_parser('import-manifest', help='생성 결과를 입력 계약으로 변환합니다.')
    ingest.add_argument('--manifest', required=True)
    ingest.add_argument('--video-root', required=True)
    ingest.add_argument('--source-group', required=True)
    ingest.add_argument('--split', choices=['train','validation','test','unassigned'], default='unassigned')
    ingest.add_argument('--out', required=True)
    run = sub.add_parser('run', help='실제 영상을 지정한 모델로 분석합니다.')
    run.add_argument('--records', required=True)
    run.add_argument('--provider', choices=['valen','cosmos-http'], required=True)
    run.add_argument('--model-dir')
    run.add_argument('--endpoint')
    run.add_argument('--model')
    run.add_argument('--revision', required=True)
    run.add_argument('--device', default='cuda')
    run.add_argument('--sample-fps', type=float, default=4)
    run.add_argument('--max-frames', type=int, default=32)
    run.add_argument('--out', required=True)
    args = parser.parse_args()
    try:
        rubric = load_rubric(args.rubric)
        if args.command == 'demo':
            records, observations = demo_data(rubric)
            report = run_batch(records, FixtureProvider(observations), rubric, args.out)
        elif args.command == 'import-manifest':
            records = import_manifest(args.manifest,args.video_root,rubric,args.source_group,args.split)
            write_json(args.out,records)
            print(f'{len(records)}개 입력을 저장했습니다. target·원본 관측 주기·window를 검토해 주세요: {args.out}')
            return
        else:
            from .semantic_providers import ValenProvider, CosmosHTTPProvider
            sampling = {'sample_fps':args.sample_fps,'max_frames':args.max_frames}
            if args.provider == 'valen':
                if not args.model_dir:
                    parser.error('--model-dir가 필요합니다.')
                provider = ValenProvider(args.model_dir,args.revision,args.device,**sampling)
            else:
                if not args.endpoint or not args.model:
                    parser.error('--endpoint와 --model이 필요합니다.')
                provider = CosmosHTTPProvider(args.endpoint,args.model,args.revision,**sampling)
            report = run_batch(read_json(args.records),provider,rubric,args.out)
        print('처리 결과:', report['counts'])
        print('후보 라벨은 사람 검수 전의 결과입니다. 보고서:', Path(args.out)/'report.json')
        if any('error' in r for r in report['results']):
            parser.exit(2,'일부 영상 처리에 실패했습니다. report.json의 error를 확인해 주세요.\n')
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(2,f'실행 실패: {exc}\n')


if __name__ == '__main__':
    main()

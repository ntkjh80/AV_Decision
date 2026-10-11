"""노트북과 같은 함수를 사용하는 명령행 진입점입니다."""
import argparse
from pathlib import Path

from .semantic import (DEFAULT_RUBRIC, FixtureProvider, demo_data, import_manifest,
                       load_rubric, read_json, run_batch, write_json)
from .provider_factory import PROVIDER_NAMES, create_provider, load_provider_config
from .classification import DEFAULT_CLASSIFICATION, load_classification, print_scores


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
    prepare = sub.add_parser('prepare-inputs', help='영상·이미지 파일 또는 폴더에서 입력 목록을 만듭니다.')
    prepare.add_argument('--input', required=True)
    prepare.add_argument('--source-group', required=True)
    prepare.add_argument('--split', choices=['train', 'validation', 'test', 'unassigned'], default='unassigned')
    prepare.add_argument('--target-file', help='선택 사항: 질문 ID→목표 라벨의 JSON. 생략하면 목표 없이 분류합니다.')
    prepare.add_argument('--source-observation-hz', type=float)
    prepare.add_argument('--start-s', type=float, default=0)
    prepare.add_argument('--end-s', type=float)
    prepare.add_argument('--out', required=True)
    run = sub.add_parser('run', help='실제 영상을 지정한 모델로 분석합니다.')
    run.add_argument('--records', required=True)
    selection = run.add_mutually_exclusive_group(required=True)
    selection.add_argument('--provider', choices=PROVIDER_NAMES)
    selection.add_argument('--provider-config', help='모델 설정 JSON. 판별 기준 --rubric과 별개입니다.')
    run.add_argument('--model-dir')
    run.add_argument('--endpoint')
    run.add_argument('--model')
    run.add_argument('--revision')
    run.add_argument('--device')
    run.add_argument('--sample-fps', type=float)
    run.add_argument('--max-frames', type=int)
    run.add_argument('--max-duration-s', type=float)
    run.add_argument('--timeout-s', type=float)
    run.add_argument('--classification-config', default=str(DEFAULT_CLASSIFICATION), help='확신도 구간과 그 안의 상황 폴더 설정')
    run.add_argument('--no-copy', action='store_true', help='파일 복사 없이 점수 보고서만 저장합니다.')
    run.add_argument('--gpus', help='Valen 작업 자동 분산: auto 또는 현재 보이는 GPU 번호 0,1')
    run.add_argument('--worker-timeout-s', type=float, default=600)
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
        elif args.command == 'prepare-inputs':
            from .inputs import prepare_inputs
            records = prepare_inputs(args.input, rubric, source_group=args.source_group, split=args.split,
                target=read_json(args.target_file) if args.target_file else None,
                source_observation_hz=args.source_observation_hz, start_s=args.start_s, end_s=args.end_s)
            write_json(args.out, records)
            print(f'{len(records)}개 입력을 저장했습니다. 원본 그룹·구간·목표를 검토해 주세요: {args.out}')
            return
        else:
            # 파일과 개별 옵션을 섞으면 실제 적용값이 모호해지므로 한 방식만 허용합니다.
            option_keys = ('model_dir', 'endpoint', 'model', 'revision', 'device', 'timeout_s')
            sampling_keys = ('sample_fps', 'max_frames', 'max_duration_s')
            selected = {key: getattr(args, key) for key in option_keys if getattr(args, key) is not None}
            sampling = {key: getattr(args, key) for key in sampling_keys if getattr(args, key) is not None}
            if args.provider_config:
                if selected or sampling:
                    parser.error('--provider-config와 개별 모델·샘플링 옵션을 함께 사용할 수 없습니다. JSON을 수정해 주세요.')
                config = load_provider_config(args.provider_config)
            else:
                config = {'schema_version': 1, 'provider': args.provider, 'sampling': sampling, **selected}
            classification = load_classification(args.classification_config)
            if args.gpus:
                from .distributed import run_distributed
                report = run_distributed(read_json(args.records), config, rubric, args.out,
                    devices=args.gpus, classification=classification, worker_timeout_s=args.worker_timeout_s,
                    copy_files=not args.no_copy)
            else:
                provider = create_provider(config)
                report = run_batch(read_json(args.records),provider,rubric,args.out,
                                   classification=classification, copy_files=not args.no_copy)
        print_scores(report)
        print('처리 결과:', report['counts'])
        print('확신도별 분포:', report['classification_summary']['confidence_band_counts'])
        print('후보 라벨은 사람 검수 전의 결과입니다. 시각 보고서:', Path(args.out)/'summary.html')
        if (any('error' in r for r in report['results']) or report['classification_summary']['copy_failed_count']
                or report['provider'].get('worker_errors')):
            parser.exit(2,'일부 추론·분산 작업·파일 복사에 실패했습니다. report.json과 classification.json을 확인해 주세요.\n')
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(2,f'실행 실패: {exc}\n')


if __name__ == '__main__':
    main()

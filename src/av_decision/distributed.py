"""GPU마다 모델 한 개를 두고 다음 입력을 자동 배정하는 추론 실행기입니다.

모델 가중치를 여러 장에 쪼개는 모델 병렬화가 아닙니다. CUDA 상태를 fork로
물려받지 않도록 spawn을 사용하고, 프로세스별 모델은 한 번만 생성합니다.
"""
from __future__ import annotations

from collections import deque
import copy
import multiprocessing as mp
from multiprocessing.connection import wait
from pathlib import Path
import time

from .provider_factory import create_provider
from .semantic import failed_result, finite_number, finish_batch, prepare_batch, process_record


def select_devices(selection='auto'):
    """현재 프로세스에 노출된 CUDA 장치만 선택합니다. 메모리 여유를 예약하지 않습니다."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('GPU 자동 분산에는 CUDA용 PyTorch가 필요합니다.') from exc
    if not torch.cuda.is_available():
        raise ValueError('사용 가능한 CUDA GPU가 없습니다.')
    count = torch.cuda.device_count()
    if selection == 'auto':
        return [f'cuda:{i}' for i in range(count)]
    if not isinstance(selection, str):
        raise ValueError('GPU 목록은 auto 또는 0,1처럼 지정합니다.')
    try:
        ids = [int(value.strip()) for value in selection.split(',')]
    except ValueError as exc:
        raise ValueError('GPU 목록은 auto 또는 0,1처럼 지정합니다.') from exc
    if not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= count for i in ids):
        raise ValueError('중복 없이 현재 보이는 GPU 번호를 지정해 주세요.')
    return [f'cuda:{i}' for i in ids]


def _worker(connection, config, rubric, output, additional, factory, configure_cuda):
    """GPU 메모리와 모델 객체는 작업 프로세스 안에만 둡니다."""
    try:
        if configure_cuda:
            import torch
            torch.cuda.set_device(config['device'])
        provider = factory(config)
        connection.send(('ready', dict(provider.metadata)))
        while True:
            task = connection.recv()
            if task is None:
                break
            index, record = task
            result, plan = process_record(index, record, provider, rubric, output, additional)
            connection.send(('result', index, result, plan))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as exc:
        try:
            connection.send(('worker_error', type(exc).__name__ + ': ' + str(exc)))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


def _execute_workers(records, config, rubric, output, devices, additional, timeout_s,
                     factory=create_provider, configure_cuda=True):
    """완료된 GPU에 다음 자료를 보냅니다. 비정상 종료·시간 초과 자료도 누락하지 않습니다.

    factory와 configure_cuda 인자는 실제 GPU 없이 spawn·통신·실패 처리를 검사하는
    테스트용 연결점입니다. 공개 CLI에서는 항상 실제 provider와 CUDA 설정을 사용합니다.
    """
    context = mp.get_context('spawn')
    pending = deque(enumerate(records))
    results, plans = [None] * len(records), [None] * len(records)
    workers, errors = [], []

    def fail(worker, message):
        errors.append({'device': worker['device'], 'error': message})
        task = worker['task']
        if task is not None:
            index, record = task
            result = failed_result(record, RuntimeError(message), worker['metadata'])
            result['elapsed_s'] = round(time.monotonic() - worker['started'], 6)
            results[index] = result
        worker['active'] = False
        worker['connection'].close()
        if worker['process'].is_alive():
            worker['process'].terminate()

    def dispatch(worker):
        if pending:
            task = pending.popleft()
            worker['task'], worker['started'] = task, time.monotonic()
            try:
                worker['connection'].send(task)
            except (BrokenPipeError, EOFError, OSError) as exc:
                fail(worker, '작업 전달 실패: ' + str(exc))
        else:
            worker['task'] = None
            worker['active'] = False
            try:
                worker['connection'].send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass

    try:
        for device in devices[:len(records)]:
            parent, child = context.Pipe()
            selected = {**copy.deepcopy(config), 'device': device}
            process = context.Process(target=_worker,
                args=(child, selected, rubric, str(output), additional, factory, configure_cuda))
            worker = {'device': device, 'process': process, 'connection': parent, 'active': True,
                      'task': None, 'started': time.monotonic(),
                      'metadata': {'name': config['provider'], 'model': Path(config.get('model_dir', '')).name,
                                   'revision': config['revision'], 'device': device,
                                   'real_inference': configure_cuda, 'probability_kind': 'not_available'}}
            try:
                process.start()
            except Exception as exc:
                errors.append({'device': device, 'error': '프로세스 시작 실패: ' + str(exc)})
                parent.close()
                child.close()
                continue
            child.close()
            workers.append(worker)
        while any(w['active'] for w in workers):
            active = [w for w in workers if w['active']]
            remaining = min(max(0, timeout_s - (time.monotonic() - w['started'])) for w in active)
            ready = wait([w['connection'] for w in active], timeout=min(0.5, remaining))
            for worker in active:
                if worker['connection'] in ready:
                    try:
                        message = worker['connection'].recv()
                    except (EOFError, OSError):
                        fail(worker, '작업 프로세스가 결과 없이 종료되었습니다.')
                        continue
                    if message[0] == 'ready':
                        worker['metadata'] = message[1]
                        dispatch(worker)
                    elif message[0] == 'result':
                        _, index, result, plan = message
                        results[index], plans[index] = result, plan
                        dispatch(worker)
                    else:
                        fail(worker, '작업 프로세스 오류: ' + message[1])
                elif time.monotonic() - worker['started'] >= timeout_s:
                    fail(worker, f'GPU 작업 시간이 {timeout_s}초를 초과했습니다.')
                elif not worker['process'].is_alive():
                    fail(worker, 'GPU 작업 프로세스가 비정상 종료되었습니다.')
        for index, record in pending:
            results[index] = failed_result(record, RuntimeError('실행 가능한 GPU 작업 프로세스가 남지 않았습니다.'),
                {'name': config['provider'], 'revision': config['revision'],
                 'real_inference': configure_cuda, 'probability_kind': 'not_available'})
    finally:
        for worker in workers:
            worker['connection'].close()
            worker['process'].join(timeout=2)
            if worker['process'].is_alive():
                worker['process'].terminate()
                worker['process'].join(timeout=2)
            if worker['process'].is_alive():
                worker['process'].kill()
                worker['process'].join(timeout=2)
    if any(result is None for result in results):
        raise RuntimeError('분산 실행 결과 수가 입력 수와 일치하지 않습니다.')
    return results, [plan for plan in plans if plan is not None], errors


def run_distributed(records, provider_config, rubric, output_dir, *, devices='auto',
                    classification=None, worker_timeout_s=600, copy_files=True):
    """단일 호스트의 여러 GPU에 영상·이미지를 자동 배정합니다. Valen 로컬 전용입니다."""
    if provider_config.get('provider') != 'valen':
        raise ValueError('클라이언트 GPU 자동 분산은 Valen 로컬 경로에서만 사용합니다. HTTP 분산은 서버에서 설정하세요.')
    if not finite_number(worker_timeout_s) or worker_timeout_s <= 0:
        raise ValueError('worker_timeout_s는 양수여야 합니다.')
    # 기본 설정 검사는 가중치를 로드하지 않습니다.
    provider = create_provider(provider_config)
    selected = select_devices(devices)
    output = prepare_batch(records, rubric, output_dir, classification)
    additional = classification['scene_criteria'] if classification else ()
    results, plans, errors = _execute_workers(records, provider_config, rubric, output, selected,
                                              additional, worker_timeout_s)
    metadata = {**provider.metadata, 'device': 'multiple',
                'execution': 'dynamic_data_parallel_spawn', 'devices': selected[:len(records)],
                'worker_timeout_s': worker_timeout_s, 'worker_errors': errors}
    return finish_batch(records, results, plans, metadata, rubric, output, classification, copy_files)

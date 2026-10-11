"""실제 spawn 프로세스로 자동 배정과 실패 처리를 확인합니다. GPU 추론은 대체합니다."""
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from av_decision.distributed import _execute_workers, run_distributed, select_devices
from av_decision.semantic import FixtureProvider, demo_data, load_rubric, prepare_batch


class WorkerFixture(FixtureProvider):
    def __init__(self, config):
        _, observations = demo_data(load_rubric())
        super().__init__(observations)
        self.mode = config.get('test_mode')
        self.metadata = {**FixtureProvider.metadata, 'device': config['device']}

    def observe(self, record, rubric, evidence_dir):
        if self.mode == 'crash':
            os._exit(7)
        time.sleep(.03)
        return super().observe(record, rubric, evidence_dir)


def worker_fixture_factory(config):
    if config.get('test_mode') == 'initialization_timeout':
        time.sleep(5)
    return WorkerFixture(config)


class DistributedTests(unittest.TestCase):
    def setUp(self):
        self.rubric = load_rubric()
        self.records, _ = demo_data(self.rubric)
        self.config = {'provider': 'valen', 'revision': 'fixture', 'model_dir': 'not-loaded'}

    def execute(self, mode=None, timeout=15):
        with tempfile.TemporaryDirectory() as tmp:
            output = prepare_batch(self.records, self.rubric, Path(tmp) / 'out')
            return _execute_workers(self.records, {**self.config, 'test_mode': mode}, self.rubric,
                output, ['cuda:0', 'cuda:1'], (), timeout,
                factory=worker_fixture_factory, configure_cuda=False)

    def test_dynamic_spawn_preserves_order_and_serial_decisions(self):
        results, plans, errors = self.execute()
        self.assertEqual(errors, [])
        self.assertEqual([r['clip_id'] for r in results], [r['clip_id'] for r in self.records])
        self.assertEqual([r['disposition'] for r in results],
                         ['candidate', 'relabel_candidate', 'review', 'quarantine', 'review', 'candidate'])
        self.assertEqual(len(plans), 6)
        self.assertTrue(all(r['provider']['real_inference'] is False for r in results))

    def test_crashed_workers_do_not_drop_assigned_or_pending_records(self):
        results, plans, errors = self.execute('crash')
        self.assertEqual(len(results), 6)
        self.assertEqual(len(errors), 2)
        self.assertEqual(plans, [])
        self.assertTrue(all('error' in r and r['disposition'] == 'review' for r in results))

    def test_initialization_timeout_is_bounded_and_preserves_inputs(self):
        results, plans, errors = self.execute('initialization_timeout', timeout=.8)
        self.assertEqual(len(results), 6)
        self.assertTrue(all('error' in r for r in results))
        self.assertTrue(any('초과' in item['error'] for item in errors))

    def test_gpu_selection_uses_visible_indices_and_rejects_duplicates(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 3))
        with patch.dict(sys.modules, {'torch': fake}):
            self.assertEqual(select_devices(), ['cuda:0', 'cuda:1', 'cuda:2'])
            self.assertEqual(select_devices('2,0'), ['cuda:2', 'cuda:0'])
            for value in ('0,0', '3', '-1', 'cuda:0'):
                with self.assertRaises(ValueError):
                    select_devices(value)

    def test_http_provider_cannot_trigger_local_gpu_distribution(self):
        with self.assertRaises(ValueError):
            run_distributed(self.records, {'provider': 'vlm-http'}, self.rubric, 'unused')


if __name__ == '__main__':
    unittest.main()

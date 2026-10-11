"""모델을 교체해도 같은 판정·기록 계약을 지키는지 CPU에서 확인합니다."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from av_decision.provider_factory import create_provider, load_provider
from av_decision.semantic import demo_data, load_rubric
from av_decision.semantic_cli import main
from av_decision.semantic_providers import CosmosHTTPProvider, ValenProvider, VLMHTTPProvider


class ProviderSelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = {'schema_version': 1, 'provider': 'vlm-http',
                       'endpoint': 'http://127.0.0.1:8000/v1',
                       'model': 'study-vision-model', 'revision': 'snapshot-abc'}

    def test_http_model_switch_keeps_revision_and_sampling(self):
        for name, cls in [('vlm-http', VLMHTTPProvider), ('cosmos-http', CosmosHTTPProvider)]:
            with self.subTest(name=name):
                provider = create_provider({**self.config, 'provider': name,
                                            'sampling': {'sample_fps': 5, 'max_frames': 40}})
                self.assertIsInstance(provider, cls)
                self.assertEqual(provider.metadata['name'], name.replace('-', '_'))
                self.assertEqual(provider.metadata['model'], 'study-vision-model')
                self.assertEqual(provider.metadata['revision'], 'snapshot-abc')
                self.assertEqual(provider.metadata['sampling'],
                                 {'sample_fps': 5, 'max_frames': 40, 'max_duration_s': 8.0})

    def test_valen_relative_path_is_relative_to_config_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'checkpoint').mkdir()
            path = root / 'provider.json'
            path.write_text(json.dumps({'schema_version': 1, 'provider': 'valen',
                                       'model_dir': 'checkpoint', 'revision': 'local-abc'}))
            provider = load_provider(path)
            self.assertIsInstance(provider, ValenProvider)
            self.assertEqual(provider.model_dir, (root / 'checkpoint').resolve())
            self.assertEqual(provider.metadata['model'], 'checkpoint')
            self.assertIsNone(provider.model)  # 설정만 읽을 때 GPU나 가중치를 로드하지 않습니다.

    def test_invalid_configuration_fails_before_inference(self):
        cases = [None, {}, {**self.config, 'schema_version': True},
                 {**self.config, 'provider': 'typo'}, {**self.config, 'revision': ' '},
                 {**self.config, 'api_key': 'not-allowed'},
                 {**self.config, 'device': 'cuda'}, {**self.config, 'timeout_s': 0},
                 {**self.config, 'sampling': {'sample_fps': float('nan')}},
                 {**self.config, 'sampling': {'sample_fps': True}},
                 {**self.config, 'sampling': {'max_frames': 1}},
                 {**self.config, 'sampling': {'max_duration_s': 0}},
                 {**self.config, 'sampling': {'frames': 32}}]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                create_provider(value)

    def test_cli_config_and_legacy_flags_use_common_inference_contract(self):
        rubric = load_rubric()
        records, observations = demo_data(rubric)
        row = records[0]
        response = {'choices': [{'message': {'content': json.dumps(observations[row['clip_id']])}}]}
        for use_config in (True, False):
            with self.subTest(config=use_config), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                jpg = root / 'frame.jpg'; jpg.write_bytes(b'fake JPEG')
                frames = [{'timestamp_s': 0., 'path': str(jpg)}, {'timestamp_s': .25, 'path': str(jpg)}]
                records_path = root / 'records.json'; records_path.write_text(json.dumps([row]))
                config_path = root / 'provider.json'; config_path.write_text(json.dumps(self.config))
                out = root / 'report'
                args = ['av-decision', 'run', '--records', str(records_path), '--out', str(out), '--no-copy']
                if use_config:
                    args += ['--provider-config', str(config_path)]
                else:
                    args += ['--provider', 'cosmos-http', '--endpoint', self.config['endpoint'],
                             '--model', self.config['model'], '--revision', self.config['revision']]
                with patch.object(sys, 'argv', args), contextlib.redirect_stdout(io.StringIO()), \
                     patch('av_decision.semantic_providers.sample_video', return_value=(frames, {'max_gap_s': .25})), \
                     patch('av_decision.semantic_providers.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as call:
                    main()
                body = json.loads(call.call_args.args[0].data)
                self.assertEqual(body['model'], self.config['model'])
                report = json.loads((out / 'report.json').read_text())
                self.assertEqual(report['counts']['review'], 1)
                self.assertNotIn('error', report['results'][0])
                self.assertEqual(report['provider']['sampling']['sample_fps'], 4)
                self.assertTrue(all(answer['probabilities'] is None
                                    for answer in report['results'][0]['observed'].values()))

    def test_cli_rejects_config_overrides_before_reading_records(self):
        args = ['av-decision', 'run', '--records', 'unused.json', '--out', 'unused',
                '--provider-config', 'unused.json', '--revision', 'ambiguous-override']
        with patch.object(sys, 'argv', args), contextlib.redirect_stderr(io.StringIO()) as error, \
             self.assertRaises(SystemExit) as raised:
            main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn('함께 사용할 수 없습니다', error.getvalue())


if __name__ == '__main__':
    unittest.main()

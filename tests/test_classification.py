"""점수 구간, 이미지의 시간 조건, 원본 보존, 누락 없는 보고서를 검사합니다."""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from av_decision.classification import confidence_summary, load_classification, safe_component
from av_decision.inputs import prepare_inputs
from av_decision.semantic import FixtureProvider, demo_data, judge, load_rubric, run_batch
from av_decision.semantic_providers import sample_media, frame_content


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        self.rubric = load_rubric()
        self.records, self.answers = demo_data(self.rubric)
        self.config = load_classification()

    def test_confidence_boundaries_unknown_and_unscored(self):
        for value, expected in [(0.9, 'high'), (0.899, 'medium'), (0.8, 'medium'), (0.799, 'low'), (None, 'unscored')]:
            result = {'provider': {}, 'checked_criteria': ['a'],
                      'observed': {'a': {'choice': 'yes', 'confidence': value, 'margin': .4}}}
            self.assertEqual(confidence_summary(result, self.config)['band'], expected)
        result['observed']['a']['choice'] = 'unknown'
        self.assertEqual(confidence_summary(result, self.config)['band'], 'unknown')
        result['error'] = 'failure'
        self.assertEqual(confidence_summary(result, self.config)['band'], 'error')

    def test_images_never_assert_temporal_events_but_can_classify_static_conditions(self):
        row = dict(self.records[0], media_type='image', target={'weather': 'rain'})
        raw = self.answers[row['clip_id']]
        result = judge(row, raw, {'source_kind': 'single_image'}, self.rubric)
        self.assertEqual(result['disposition'], 'candidate')
        self.assertEqual(result['observed']['cut_in']['choice'], 'unknown')
        self.assertIsNone(result['observed']['cut_in']['confidence'])
        self.assertNotIn('continuity', result['checked_criteria'])
        temporal = judge(dict(row, target={'cut_in': 'yes'}), raw, {}, self.rubric)
        self.assertEqual(temporal['disposition'], 'review')
        self.assertTrue(any('정지이미지' in value for value in temporal['review_reasons']))

    def test_copy_bands_scores_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = copy.deepcopy(self.records[:2])
            for i, row in enumerate(rows):
                folder = root / str(i); folder.mkdir()
                source = folder / 'same-name.mp4'; source.write_bytes(f'video-{i}'.encode())
                row['video_path'] = str(source)
                row['window'] = {'start_s': 1, 'end_s': 3}
            report = run_batch(rows, FixtureProvider(self.answers), self.rubric, root / 'out', self.config)
            summary = report['classification_summary']
            self.assertEqual(summary['copied_count'], 2)
            self.assertEqual(sum(summary['confidence_band_counts'].values()), 2)
            copies = [result['classification']['copied_path'] for result in report['results']]
            self.assertEqual(len(set(copies)), 2)
            for row, relative in zip(rows, copies):
                self.assertTrue(relative.startswith('classified/high/'))
                destination = root / 'out' / relative
                self.assertEqual(destination.read_bytes(), Path(row['video_path']).read_bytes())
                sidecar = json.loads(destination.with_suffix('.mp4.json').read_text())
                self.assertEqual(sidecar['assessed_window'], row['window'])
            with (root / 'out/scores.csv').open(encoding='utf-8-sig', newline='') as stream:
                scores = list(csv.DictReader(stream))
            mismatch = next(s for s in scores if s['clip_id'] == rows[1]['clip_id'] and s['criterion'] == 'cut_in')
            self.assertEqual(mismatch['choice'], 'no')
            self.assertAlmostEqual(float(mismatch['selected_probability']), .94)
            self.assertAlmostEqual(float(mismatch['target_probability']), .03)
            self.assertTrue((root / 'out/summary.html').is_file())

    def test_low_scene_score_is_checked_even_without_target(self):
        row = dict(self.records[0], target={'lighting': 'night'})
        raw = copy.deepcopy(self.answers[row['clip_id']])
        raw['answers']['weather']['probabilities'] = {'rain': .6, 'clear': .1, 'fog': .1, 'snow': .1, 'unknown': .1}
        with tempfile.TemporaryDirectory() as tmp:
            report = run_batch([row], FixtureProvider({row['clip_id']: raw}), self.rubric, Path(tmp) / 'out',
                               self.config, copy_files=False)
            result = report['results'][0]
            self.assertIn('weather', result['checked_criteria'])
            self.assertEqual(result['confidence_summary']['band'], 'low')
            self.assertEqual(result['disposition'], 'review')
            self.assertFalse((Path(tmp) / 'out/classified').exists())
            self.assertIn('파일 복사를 껐습니다', (Path(tmp) / 'out/summary.html').read_text())

    def test_failed_inference_and_missing_file_are_counted_separately(self):
        row = dict(self.records[0], video_path='/not-present-for-test.mp4')
        with tempfile.TemporaryDirectory() as tmp:
            report = run_batch([row], FixtureProvider({}), self.rubric, Path(tmp) / 'out', self.config)
            self.assertEqual(report['classification_summary']['confidence_band_counts']['error'], 1)
            self.assertEqual(report['classification_summary']['copy_failed_count'], 1)
            self.assertEqual(report['counts']['review'], 1)

    def test_source_changed_after_observation_is_not_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.mp4'; path.write_bytes(b'before')
            row = dict(self.records[0], video_path=str(path))
            raw = self.answers[row['clip_id']]
            class Changed:
                metadata = FixtureProvider.metadata
                def observe(self, *args):
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    path.write_bytes(b'after')
                    return raw, {'video_sha256': digest, 'max_gap_s': .25}
            report = run_batch([row], Changed(), self.rubric, Path(tmp) / 'out', self.config)
            self.assertEqual(report['classification_summary']['copy_failed_count'], 1)
            self.assertEqual(list((Path(tmp) / 'out/classified').rglob('*.mp4')), [])

    def test_paths_and_html_cannot_execute_labels(self):
        for value in ('../outside', 'x/y', 'CON', '<script>alert(1)</script>'):
            part = safe_component(value)
            self.assertNotIn('/', part)
            self.assertNotIn('..', part)
        row = dict(self.records[0], clip_id='<script>alert(1)</script>')
        with tempfile.TemporaryDirectory() as tmp:
            run_batch([row], FixtureProvider({row['clip_id']: self.answers[self.records[0]['clip_id']]}),
                      self.rubric, Path(tmp) / 'out')
            page = (Path(tmp) / 'out/summary.html').read_text()
            self.assertNotIn('<script>', page)
            self.assertIn('&lt;script&gt;', page)

    def test_input_scanner_supports_mixed_media_and_empty_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'a.JPG').write_bytes(b'image')
            (root / 'b.mp4').write_bytes(b'video')
            (root / 'ignore.txt').write_text('ignore')
            rows = prepare_inputs(root, self.rubric, source_group='drive-001', end_s=3)
            self.assertEqual({r['media_type'] for r in rows}, {'image', 'video'})
            self.assertTrue(all(r['target'] == {} for r in rows))
            self.assertEqual(len({r['clip_id'] for r in rows}), 2)
            self.assertIsNone(next(r for r in rows if r['media_type'] == 'video')['source_observation_hz'])

    @unittest.skipUnless(importlib.util.find_spec('PIL'), 'Pillow를 설치한 환경에서 실제 이미지 디코딩을 검사합니다.')
    def test_real_image_decoder_and_temporal_prompt(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'image.png'
            Image.new('RGB', (24, 16), 'blue').save(path)
            frames, evidence = sample_media({'image_path': str(path)}, Path(tmp) / 'evidence')
            self.assertIsNone(frames[0]['timestamp_s'])
            self.assertIsNone(evidence['max_gap_s'])
            self.assertEqual(evidence['media_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertIn('단일 정지 이미지', frame_content(frames)[0]['text'])


if __name__ == '__main__':
    unittest.main()

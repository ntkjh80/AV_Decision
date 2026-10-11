"""판정의 의미, 실패 보존, 시간축, 실제 모델에 보내는 입력 계약을 확인합니다.

실제 GPU 가중치의 정확도를 측정하는 테스트는 아닙니다. 외부 모델과 디코더는
작은 대역으로 바꿔서, 잘못된 입력이 조용히 통과하지 않는지 확인합니다.
"""
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from av_decision.semantic import (FixtureProvider, check_group_splits, demo_data,
    evaluate_human_labels, import_manifest, judge, load_rubric, normalize_answers,
    regeneration_plan, run_batch, validate_record, write_json)
from av_decision.semantic_providers import CosmosHTTPProvider, ValenProvider, sample_video


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.rubric = load_rubric()
        self.records, self.answers = demo_data(self.rubric)
        self.record = self.records[0]
        self.raw = self.answers[self.record['clip_id']]
        self.evidence = {'max_gap_s': 0.25}

    def judge(self, record=None, raw=None, evidence=None):
        return judge(record or self.record, raw or self.raw, evidence or self.evidence, self.rubric)

    def test_demo_covers_distinct_dispositions(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'run'
            report = run_batch(self.records, FixtureProvider(self.answers), self.rubric, out)
            self.assertEqual(report['counts'], {'candidate':2,'relabel_candidate':1,'review':2,'quarantine':1})
            self.assertTrue(all(r['requires_human_approval'] for r in report['results']))
            self.assertFalse(report['provider']['real_inference'])
            with self.assertRaises(FileExistsError):
                run_batch(self.records, FixtureProvider(self.answers), self.rubric, out)

    def test_relabel_keeps_target_separate_from_observation(self):
        row = self.records[1]
        result = self.judge(row, self.answers[row['clip_id']])
        self.assertEqual(result['target']['cut_in'], 'yes')
        self.assertEqual(result['observed']['cut_in']['choice'], 'no')
        self.assertEqual(result['mismatched_criteria'], ['cut_in'])
        self.assertIn('continuity', result['checked_criteria'])
        self.assertNotIn('pedestrian_crossing', result['checked_criteria'])

    def test_limited_visibility_is_not_a_defect(self):
        row = self.records[5]
        result = self.judge(row, self.answers[row['clip_id']])
        self.assertEqual(result['disposition'], 'candidate')
        self.assertTrue(result['difficult_scene'])

    def test_probability_schema_rejects_nan_sum_missing_and_wrong_choice(self):
        for probs in ({'rain':float('nan')}, {'rain':1},
                      {'clear':0.1,'rain':0.8,'fog':0.1,'snow':0.1,'unknown':0.1},
                      {'clear':1.,'rain':0.,'fog':0.,'snow':0.,'unknown':0.}):
            raw = copy.deepcopy(self.raw)
            raw['answers']['weather']['probabilities'] = probs
            with self.subTest(probs=probs), self.assertRaises(ValueError):
                normalize_answers(raw,self.rubric)

    def test_all_question_ids_required(self):
        raw = copy.deepcopy(self.raw)
        raw['answers'].pop('weather')
        with self.assertRaises(ValueError):
            normalize_answers(raw,self.rubric)

    def test_low_margin_or_missing_probability_requires_review(self):
        for probs in ({'yes':0.5,'no':0.5,'unknown':0}, None):
            raw = copy.deepcopy(self.raw)
            raw['answers']['cut_in']['probabilities'] = probs
            self.assertEqual(self.judge(raw=raw)['disposition'], 'review')

    def test_unknown_is_not_treated_as_negative(self):
        row = self.records[2]
        result = self.judge(row,self.answers[row['clip_id']])
        self.assertEqual(result['disposition'],'review')
        self.assertNotIn('cut_in',result['mismatched_criteria'])

    def test_original_observation_rate_cannot_be_invented_by_encoding(self):
        for hz in (2, None):
            row = dict(self.record, source_observation_hz=hz)
            self.assertEqual(self.judge(row)['disposition'], 'review')
        for gap in (0.5, -1, float('nan')):
            self.assertEqual(self.judge(evidence={'max_gap_s':gap})['disposition'], 'review')

    def test_uncertain_artifact_stays_in_review(self):
        row = self.records[3]
        result = self.judge(dict(row,source_observation_hz=None),self.answers[row['clip_id']])
        self.assertEqual(result['disposition'],'review')
        self.assertEqual(result['observed']['continuity']['choice'],'artifact')

    def test_regeneration_budget_and_test_partition_are_held(self):
        row = self.records[1]
        result = self.judge(row,self.answers[row['clip_id']])
        self.assertTrue(regeneration_plan(row,result,self.rubric)['ready_for_generation'])
        for changes in ({'attempt':2},{'split':'test'},{'split':'unassigned'}):
            plan = regeneration_plan(dict(row,**changes), result, self.rubric)
            self.assertFalse(plan['ready_for_generation'])
            self.assertEqual(plan['parent_source_group'], row['source_group'])

    def test_derived_clips_must_share_partition(self):
        with self.assertRaises(ValueError):
            check_group_splits([self.record,dict(self.record,split='test')])
        check_group_splits([self.record,dict(self.record,clip_id='variant')])

    def test_model_failure_is_preserved_in_report_and_review_csv(self):
        class Broken:
            metadata = {'name':'broken'}
            def observe(self, *args):
                raise IndexError('응답 목록이 비었습니다')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'run'
            report = run_batch([self.record],Broken(),self.rubric,out)
            self.assertEqual(report['counts']['review'],1)
            self.assertEqual(report['results'][0]['error_type'],'IndexError')
            with (out/'human_review.csv').open(encoding='utf-8-sig') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows),len(self.rubric['questions']))
            self.assertTrue(all(r['model_choice']=='' and r['human_choice']=='' for r in rows))

    def test_nan_output_and_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.json'
            with self.assertRaises(ValueError):
                write_json(p,{'probability':float('nan')})
            self.assertFalse(p.exists())
            write_json(p,{'a':1})
            with self.assertRaises(FileExistsError):
                write_json(p,{'a':2})

    def test_manifests_use_fields_not_module_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'video.mp4').write_bytes(b'fixture')
            manifest=root/'manifest.json'
            write_json(manifest,{'results':[{'condition':'rain','generated_video':'video.mp4'},
                                            {'scenario':'lane_cut_in','generated_video':'video.mp4'},
                                            {'scenario':'near_collision','generated_video':'video.mp4'}]})
            rows=import_manifest(manifest,root,self.rubric,'drive-1','train')
            self.assertEqual(rows[0]['target'],{'weather':'rain'})
            self.assertEqual(rows[1]['target'],{'cut_in':'yes'})
            self.assertEqual(rows[2]['target'],{})
            self.assertTrue(rows[2]['target_needs_review'])
            self.assertIsNone(rows[0]['source_observation_hz'])
            with self.assertRaises(ValueError):
                validate_record(rows[2],self.rubric)

    def test_manifest_cannot_escape_video_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'videos').mkdir(); (root/'outside.mp4').write_bytes(b'fixture')
            p=root/'manifest.json'
            write_json(p,{'results':[{'condition':'rain','generated_video':'../outside.mp4'}]})
            with self.assertRaises(ValueError):
                import_manifest(p,root/'videos',self.rubric)

    def test_metrics_keep_unknown_and_failed_positive_in_denominator(self):
        results=[{'clip_id':'a','observed':{'cut_in':{'choice':'no'}}},
                 {'clip_id':'b','observed':{'cut_in':{'choice':'unknown'}}},
                 {'clip_id':'c','error':'failure'}]
        labels=[{'clip_id':r['clip_id'],'criterion':'cut_in','human_choice':'yes'} for r in results]
        m=evaluate_human_labels(results,labels,self.rubric)['cut_in']
        self.assertEqual((m['n'],m['missing'],m['false_negative'],m['unknown_on_positive']), (3,1,1,2))
        self.assertAlmostEqual(m['false_negative_rate'],1/3)
        self.assertEqual(m['accuracy'],0)
        self.assertIsNone(m['false_positive_rate'])
        labels[0]['human_choice']='yes_typo'
        with self.assertRaises(ValueError):
            evaluate_human_labels(results,labels,self.rubric)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.rubric=load_rubric()
        self.records,self.answers=demo_data(self.rubric)

    def test_cosmos_receives_time_order_not_target_and_has_no_fake_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            jpg=Path(tmp)/'frame-0000.jpg'; jpg.write_bytes(b'fake JPEG')
            frames=[{'timestamp_s':0.,'path':str(jpg)}, {'timestamp_s':0.25,'path':str(jpg)}]
            raw=self.answers[self.records[0]['clip_id']]
            response={'choices':[{'message':{'content':json.dumps(raw)}}]}
            row=dict(self.records[0], original_prompt='SECRET_GOAL_123', video_path='rain_target_name.mp4')
            provider=CosmosHTTPProvider('http://127.0.0.1:8000/v1','reason','test-revision')
            with patch('av_decision.semantic_providers.sample_video',return_value=(frames,{'max_gap_s':0.25})), \
                 patch('av_decision.semantic_providers.urlopen',return_value=io.BytesIO(json.dumps(response).encode())) as call:
                answers,evidence=provider.observe(row,self.rubric,Path(tmp)/'evidence')
            body=json.loads(call.call_args.args[0].data)
            serialized=json.dumps(body)
            self.assertNotIn('SECRET_GOAL_123',serialized)
            self.assertNotIn('rain_target_name',serialized)
            texts=[c['text'] for c in body['messages'][0]['content'] if c['type']=='text']
            self.assertIn('0.000',texts[1]); self.assertIn('0.250',texts[2])
            self.assertTrue(all(a['probabilities'] is None for a in answers['answers'].values()))
            self.assertEqual(judge(row,answers,evidence,self.rubric)['disposition'],'review')

    def test_valen_local_model_request_matches_documented_api(self):
        from contextlib import nullcontext
        with tempfile.TemporaryDirectory() as tmp:
            model=SimpleNamespace(predict=lambda request,execution: {'request':request,'execution':execution})
            provider=ValenProvider(tmp,'reviewed-snapshot')
            provider.model=model
            fake_torch=SimpleNamespace(inference_mode=nullcontext)
            frames=[{'timestamp_s':0.,'path':'/tmp/frame-0000.jpg'}]
            with patch.dict('sys.modules',{'torch':fake_torch}), \
                 patch('av_decision.semantic_providers.sample_video',return_value=(frames,{'max_gap_s':0.25})):
                result,_=provider.observe(dict(self.records[0],original_prompt='SECRET_GOAL'),self.rubric,'unused')
            self.assertEqual(result['execution'],'shared_state')
            self.assertEqual(set(result['request']['questions']),set(self.rubric['questions']))
            self.assertNotIn('SECRET_GOAL',json.dumps(result))
            url=result['request']['state']['messages'][0]['content'][2]['image_url']['url']
            self.assertEqual(url,'/tmp/frame-0000.jpg')

    def test_endpoint_does_not_accept_embedded_credentials(self):
        with self.assertRaises(ValueError):
            CosmosHTTPProvider('https://user:password@example.com/v1','model','revision')


class SamplingTests(unittest.TestCase):
    def fake_av(self,times,duration):
        class Image:
            def convert(self,mode): return self
            def thumbnail(self,size): pass
            def save(self,path,*args,**kwargs): Path(path).write_bytes(b'frame')
        class Container:
            streams=SimpleNamespace(video=[SimpleNamespace(duration=duration,time_base=1)])
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def decode(self,stream):
                for t in times:
                    yield SimpleNamespace(pts=t,time_base=1,to_image=lambda:Image())
        return SimpleNamespace(open=lambda path:Container(),time_base=1000000)

    def sample(self,times,duration=1.,**kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'video.mp4'; path.write_bytes(b'fixture')
            with patch.dict('sys.modules',{'av':self.fake_av(times,duration)}):
                return sample_video({'video_path':str(path)},Path(tmp)/'frames',**kwargs)

    def test_pts_sampling_excludes_end_and_preserves_gaps(self):
        frames,evidence=self.sample([i/10 for i in range(11)])
        self.assertEqual([f['timestamp_s'] for f in frames],[0,0.3,0.5,0.8])
        self.assertAlmostEqual(evidence['max_gap_s'],0.3)

    def test_sparse_source_is_not_duplicated(self):
        frames,evidence=self.sample([0,0.5])
        self.assertEqual(len(frames),2)
        self.assertEqual(evidence['max_gap_s'],0.5)

    def test_duplicate_and_missing_pts_fail(self):
        for times in ([0,0.25,0.25],[0,None,0.5],[0,0.5,0.25]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                self.sample(times)

    def test_frame_limit_never_silently_reduces_resolution(self):
        with self.assertRaises(ValueError):
            self.sample([0,0.25,0.5,0.75],max_frames=2)

    def test_long_clips_need_an_explicit_short_window(self):
        with self.assertRaises(ValueError):
            self.sample([0,1],duration=20.)


if __name__=='__main__':
    unittest.main()

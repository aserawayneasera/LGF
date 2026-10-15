import copy
import functools
import hashlib
import http.server
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import test_data as td
import evaluate_test as et
import models

def image(path, value):
    from PIL import Image
    path.parent.mkdir(parents=True,exist_ok=True)
    Image.new('RGB',(24,16),(value,value//2,255-value)).save(path)

def fixture(root):
    cats=[{'id':i+1,'name':n} for i,n in enumerate(td.CLASSES)]
    for j,dataset in enumerate(['coco_nw','coco_weather','acdc']):
        for k,(folder,annotations) in enumerate(td.data_paths(dataset,root/'data')):
            filename=f'{100+j*10+k:012d}.png' if dataset!='acdc' else f'GOPR{j:04d}_frame_{k:06d}_rgb_anon.png'
            image(folder/filename,10+j*20+k)
            td.write_json(annotations,{'images':[{'id':j*10+k,'file_name':filename,'width':24,'height':16}],
                'categories':cats,'annotations':[{'id':1,'image_id':j*10+k,'category_id':1,'bbox':[1,1,6,7],'area':42,'iscrowd':0}]})
    records=[]
    for i in range(3):
        name=f'{900+i:012d}.png';p=root/'test'/name;image(p,150+i)
        records.append(td.image_record(p,name,900+i,'coco_nw','clear'))
    keys,hashes,sources=td.exclusion_audit('coco_nw',root/'data');td.check_overlap(records,keys,hashes)
    path=root/'manifest.json'
    manifest=td.save_manifest(path,'coco_nw',root/'test',records,[{'id':td.COCO_IDS[n],'name':n} for n in td.CLASSES],sources,
                              benchmark='synthetic fixture',server_mapping_verified=True,official_class_count=80)
    return path,manifest

def fake_loader(args, names):
    import torch
    def infer(images):
        return [{'boxes':torch.tensor([[1.,1.,7.,8.]]),'scores':torch.tensor([.9]),'labels':torch.tensor([0])} for _ in images]
    return object(),torch.device('cpu'),{'model':args.method,'label':'Synthetic dummy','seed':args.seed,'training_protocol':None,'original_config':None},1,infer

def export_args(root, manifest):
    source=Path(os.environ.get('P1_TEST_UGS_PACKAGE',Path(__file__).resolve().parents[2]/'paper1_ugs_comparison_v1'))
    checkpoint=root/'best_map.pth';checkpoint.write_bytes(b'synthetic-checkpoint')
    return SimpleNamespace(manifest=manifest,train_dataset='coco_nw',transfer=False,data_root=root/'data',checkpoint=checkpoint,
        method='lgf_original',seed=42,level='C3',selection='best_map',selection_note='synthetic fixture',limit=None,batch_size=2,
        threads=1,device='cpu',output=root/'export',resume=False,lgf_package=source,recent_package=source.parent/'paper1_reviewer34_v1',runtime_root=root/'runtime')

class DatasetTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_prepared_manifest_and_image_changes(self):
        path,manifest=fixture(self.root);td.verify_manifest(manifest)
        image(Path(manifest['image_root'])/manifest['images'][0]['file_name'],20)
        with self.assertRaisesRegex(ValueError,'changed'):td.verify_manifest(manifest)
    def test_manifest_edit_rejected(self):
        _,m=fixture(self.root);m['categories'][0]['id']=99
        with self.assertRaisesRegex(ValueError,'manifest changed'):td.verify_manifest(m)
    def test_source_variants_cannot_cross_partitions(self):
        self.assertEqual(td.source_key('coco_nw','000000000100.jpg'),td.source_key('coco_weather','000000000100_snow.jpg'))
        self.assertEqual(td.source_key('acdc','GOPR0033_frame_000055_rgb_anon.png'),td.source_key('acdc','GOPR0033_frame_000055_rgb_ref_anon.png'))
        with self.assertRaises(ValueError):td.check_overlap([{'file_name':'x','source_key':'coco:100','sha256':'a'}],{'coco:100'},set())
    def test_acdc_extraction_only_adverse_test(self):
        valid='rgb_anon/fog/test/GOPR0022_frame_000111_rgb_anon.png'
        self.assertTrue(td.acdc_test_path(valid))
        self.assertFalse(td.acdc_test_path(valid.replace('/test/','/train/')))
        self.assertFalse(td.acdc_test_path(valid.replace('_rgb_anon','_rgb_ref_anon')))
        archive=self.root/'a.zip'
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr(valid,b'test');z.writestr(valid.replace('/test/','/train/'),b'train')
        result=td.extract_selected(archive,self.root/'unpacked',td.acdc_test_path)
        self.assertEqual(len(result['files']),1)
        self.assertEqual(result,td.extract_selected(archive,self.root/'unpacked',td.acdc_test_path))
    def test_archive_traversal_rejected(self):
        archive=self.root/'bad.zip'
        with zipfile.ZipFile(archive,'w') as z:z.writestr('../escape.png',b'bad')
        with self.assertRaises(ValueError):td.extract_selected(archive,self.root/'unpacked',lambda n:True)
        self.assertFalse((self.root/'escape.png').exists())
    def test_labelled_preparation_rejects_image_info(self):
        info=self.root/'info.json';td.write_json(info,{'images':[],'categories':[]})
        args=SimpleNamespace(images=self.root,annotations=info,output=self.root/'out.json',dataset='coco_nw',data_root=self.root/'data',exclude_json=[])
        with self.assertRaisesRegex(ValueError,'ground-truth'):td.prepare_labelled(args)
    def test_labelled_holdout_and_weather_keep_boxes(self):
        source,manifest=fixture(self.root)
        gt={'images':[{k:im[k] for k in ['id','file_name','width','height']} for im in manifest['images']],
            'categories':manifest['categories'],'annotations':[{'id':i,'image_id':im['id'],'category_id':1,'bbox':[1,1,6,7],'area':42,'iscrowd':0} for i,im in enumerate(manifest['images'])]}
        gt_path=self.root/'test_gt.json';td.write_json(gt_path,gt)
        args=SimpleNamespace(images=Path(manifest['image_root']),annotations=gt_path,output=self.root/'labelled.json',dataset='coco_nw',data_root=self.root/'data',exclude_json=[])
        labelled=td.prepare_labelled(args)
        weather_args=SimpleNamespace(base_manifest=args.output,test_root=self.root/'weather',condition='fog',weather_seed=42,output=self.root/'weather.json')
        weather=td.prepare_weather(weather_args);td.verify_manifest(weather)
        self.assertTrue(weather['ground_truth_available'])
        self.assertEqual(td.read_json(weather['annotations'])['annotations'],gt['annotations'])
        self.assertEqual([im['source_key'] for im in labelled['images']],[im['source_key'] for im in weather['images']])

class DownloadTests(unittest.TestCase):
    def test_resume_checksum_and_private_url_redaction(self):
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as z:z.writestr('test/a.txt','x'*3000)
        payload=stream.getvalue();etag='"unit-etag"';requests=[]
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                start=int(self.headers.get('Range','bytes=0-').split('=')[1].split('-')[0]);requests.append(start)
                self.send_response(206 if start else 200);self.send_header('ETag',etag)
                self.send_header('Content-Length',str(len(payload)-start))
                if start:self.send_header('Content-Range',f'bytes {start}-{len(payload)-1}/{len(payload)}')
                self.end_headers();self.wfile.write(payload[start:])
            def log_message(self,*args):pass
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                destination=Path(tmp)/'download.zip';url=f'http://127.0.0.1:{server.server_port}/file?token=private-test-token'
                destination.with_name('download.zip.part').write_bytes(payload[:100])
                td.write_json(destination.with_name('download.zip.download.json'),{'source_sha256':td.signature(url),'etag':etag})
                td.download(url,destination,hashlib.sha256(payload).hexdigest())
                self.assertEqual(requests,[100]);self.assertEqual(destination.read_bytes(),payload)
                self.assertNotIn('private-test-token',destination.with_name('download.zip.download.json').read_text())
                with self.assertRaisesRegex(ValueError,'SHA-256'):td.download(url,destination,'0'*64)
        finally:server.shutdown();server.server_close();thread.join()

class ExportTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.path,self.manifest=fixture(self.root);self.args=export_args(self.root,self.path)
    def tearDown(self):self.tmp.cleanup()
    def export(self):
        with patch.object(et,'load_model',fake_loader):return et.export(self.args)
    def test_coco_export_preserves_ids_and_remaps_names(self):
        report=self.export();preds=td.read_json(report['official_prediction_file'])
        self.assertEqual({p['image_id'] for p in preds},{900,901,902});self.assertEqual({p['category_id'] for p in preds},{1})
        self.assertIsNone(report['metrics']);self.assertEqual(report['checkpoint_sha256'],td.digest(report['checkpoint']))
        with zipfile.ZipFile(report['submission_zip']) as z:self.assertEqual(len(z.namelist()),1)
    def test_hidden_labels_never_produce_zero_ap(self):
        self.export()
        with self.assertRaisesRegex(ValueError,'hidden'):et.score(SimpleNamespace(export=self.args.output,output=None))
    def test_complete_export_resume_and_changed_selection_rejected(self):
        report=self.export();self.args.resume=True
        with patch.object(et,'load_model',side_effect=AssertionError('Must not rerun finished inference')):
            self.assertEqual(report,et.export(self.args))
        self.args.selection='best_ap_small'
        with self.assertRaises(ValueError):et.export(self.args)
    def test_partial_export_not_submission_or_result(self):
        self.args.limit=1;report=self.export()
        self.assertFalse(report['scientific_result']);self.assertIsNone(report['submission_zip'])
        with self.assertRaisesRegex(ValueError,'Partial'):et.score(SimpleNamespace(export=self.args.output,output=None))
    def test_interrupted_export_reuses_completed_batches(self):
        count=[0]
        def interrupt_loader(args,names):
            model,device,meta,params,infer=fake_loader(args,names)
            def wrapped(images):
                count[0]+=1
                if count[0]==2:raise RuntimeError('injected interruption')
                return infer(images)
            return model,device,meta,params,wrapped
        with patch.object(et,'load_model',interrupt_loader),self.assertRaisesRegex(RuntimeError,'injected'):et.export(self.args)
        completed=self.args.output/'batches/00000000.json';before=td.digest(completed)
        self.args.resume=True;report=self.export()
        self.assertEqual(td.digest(completed),before);self.assertEqual(len(td.read_json(report['prediction_file'])),3)
    def test_modified_predictions_rejected(self):
        report=self.export();Path(report['prediction_file']).write_text('[]')
        with self.assertRaisesRegex(ValueError,'Predictions changed'):et.verify_export(self.args.output)
    def test_acdc_mapping_uses_official_ids_and_condition_files(self):
        # Small synthetic ACDC manifest exercises the packaging route, not a benchmark claim.
        m=copy.deepcopy(self.manifest);m['dataset']='acdc';m['server_mapping_verified']=False
        for i,im in enumerate(m['images']):im['condition']=['fog','night','rain'][i]
        m['content_sha256']=td.signature({k:v for k,v in m.items() if k not in ['image_root','content_sha256']});td.write_json(self.path,m)
        self.args.train_dataset='acdc';report=self.export()
        info=self.root/'official.json';td.write_json(info,{'images':[{'id':70+i,'file_name':im['file_name'],'width':24,'height':16} for i,im in enumerate(m['images'])],
            'categories':[{'id':30+i,'name':n} for i,n in enumerate(td.CLASSES)]})
        packaged=et.package_acdc(SimpleNamespace(export=self.args.output,image_info=info,categories_json=None,metadata_provenance='synthetic test fixture',output=None))
        rows=td.read_json(self.args.output/'acdc_submission/all.json')
        self.assertEqual({r['image_id'] for r in rows},{70,71,72});self.assertEqual({r['category_id'] for r in rows},{30})
        self.assertEqual(len(packaged['submissions']),5)
    def test_ambiguous_official_ids_rejected(self):
        doc={'images':[{'id':1,'file_name':'a.png'},{'id':1,'file_name':'b.png'}],'categories':[]}
        with self.assertRaisesRegex(ValueError,'ambiguous'):et.official_image_map(doc)
    def make_labelled(self):
        gt={'images':[{k:im[k] for k in ['id','file_name','width','height']} for im in self.manifest['images']],
            'categories':self.manifest['categories'],'annotations':[{'id':i,'image_id':im['id'],'category_id':1,'bbox':[1,1,6,7],'area':42,'iscrowd':0} for i,im in enumerate(self.manifest['images'])]}
        path=self.root/'gt.json';td.write_json(path,gt)
        self.manifest.update(ground_truth_available=True,annotations=str(path),annotations_sha256=td.digest(path),server_mapping_verified=False)
        self.manifest['content_sha256']=td.signature({k:v for k,v in self.manifest.items() if k not in ['image_root','content_sha256']})
        td.write_json(self.path,self.manifest)
    def test_export_to_local_six_class_score(self):
        self.make_labelled();self.export()
        result=et.score(SimpleNamespace(export=self.args.output,output=None))
        self.assertAlmostEqual(result['metrics']['mAP'],1.)
        self.assertAlmostEqual(result['metrics']['AP_small'],1.)
        self.assertEqual(result['evaluation_split'],'test')
        self.assertIn('clear',result['condition_metrics'])
    def test_comparison_matches_seeds_and_rejects_export_reports(self):
        self.make_labelled();paths=[]
        for method in ['baseline','lgf_original']:
            for seed in [42,1337]:
                self.args.method=method;self.args.seed=seed;self.args.output=self.root/f'{method}_{seed}'
                self.export();et.score(SimpleNamespace(export=self.args.output,output=None));paths.append(self.args.output/'test_metrics.json')
        args=SimpleNamespace(reports=paths,output=self.root/'comparison');et.compare(args)
        rows=td.read_json(args.output/'comparison.json')['rows']
        self.assertEqual(len(rows),2);self.assertEqual(rows[0]['n'],2)
        self.assertEqual(rows[0]['mAP_std'],0.)
        with self.assertRaisesRegex(ValueError,'matching seed'):et.compare(SimpleNamespace(reports=paths[:-1],output=self.root/'bad_seeds'))
        with self.assertRaisesRegex(ValueError,'Only scored'):et.compare(SimpleNamespace(reports=[paths[0],self.args.output/'export_report.json'],output=self.root/'bad_exports'))

class MetricTests(unittest.TestCase):
    def test_perfect_empty_and_unsupported_classes(self):
        doc={'images':[{'id':1,'width':40,'height':40,'file_name':'a.png'}],
             'categories':[{'id':td.COCO_IDS[n],'name':n} for n in td.CLASSES]+[{'id':17,'name':'cat'}],
             'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[1,1,10,10],'area':100,'iscrowd':0},
                            {'id':2,'image_id':1,'category_id':17,'bbox':[20,20,10,10],'area':100,'iscrowd':0}]}
        predictions=[{'image_id':1,'category_id':1,'bbox':[1,1,10,10],'score':.9}]
        self.assertAlmostEqual(et.coco_score(doc,predictions)['mAP'],1.)
        self.assertEqual(et.coco_score(doc,[])['mAP'],0.)
        self.assertEqual(et.coco_score(doc,predictions)['AP_large'],-1.)

if __name__=='__main__':unittest.main()

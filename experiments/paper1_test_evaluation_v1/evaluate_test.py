#!/usr/bin/env python3
"""Export frozen detector predictions, package official submissions, or score labelled tests."""
from __future__ import annotations
import argparse
import collections
import contextlib
import copy
import csv
import io
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import time
import zipfile

from test_data import (ROOT, CLASSES, DEFAULT_DATA_ROOT, check_overlap, data_paths, digest,
                       exclusion_audit, read_json, safe_relative, signature, validate_labels, verify_manifest, write_json)
from models import METHODS, LEGACY, load_model, rgb_tensors

METRICS=['mAP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large']

def json_array(path, records):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with open(tmp,'w') as f:
        f.write('[');first=True
        for r in records:
            if not first:f.write(',')
            json.dump(r,f,separators=(',',':'),allow_nan=False);first=False
        f.write(']\n')
    os.replace(tmp,path)

def model_classes(dataset, data_root):
    records=[];categories=None
    for images,annotations in data_paths(dataset,data_root):
        doc=read_json(annotations);cats=sorted(doc['categories'],key=lambda c:c['id'])
        if len(cats)!=6 or {c['name'] for c in cats}!=set(CLASSES):raise ValueError('Original training data must define the six classes')
        if categories is not None and cats!=categories:raise ValueError('Train/validation classes differ')
        categories=cats
        image_hashes=[{'image_id':im['id'],'file_name':im['file_name'],'sha256':digest(images/safe_relative(im['file_name']))}
                      for im in sorted(doc['images'],key=lambda x:x['id'])]
        records.append(signature({'annotations':digest(annotations),'images':image_hashes}))
    return [c['name'] for c in categories],records

def prediction_rows(records):
    for image in records:
        for d in image['detections']:
            yield {'image_id':image['image_id'],'file_name':image['file_name'],**d}

def iter_batches(output, batch_records):
    for record in batch_records:
        path=Path(output)/record['file']
        if digest(path)!=record['sha256']:raise ValueError('Saved inference batch changed')
        yield from read_json(path)['images']

def zip_predictions(path, json_path, name='predictions.json'):
    tmp=Path(str(path)+'.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:z.write(json_path,name)
    os.replace(tmp,path)

def export(args):
    doc=read_json(args.manifest);verify_manifest(doc)
    if args.train_dataset is None:args.train_dataset=doc['dataset']
    if args.train_dataset!=doc['dataset'] and not args.transfer:
        raise ValueError('Cross-dataset/condition transfer requires --transfer and is reported separately')
    for source in doc['exclusion_sources']:
        path=source.get('annotations',source.get('extra_history_json'))
        if digest(path)!=source['sha256']:raise ValueError('Original split/history metadata changed since test preparation')
    keys,hashes,_=exclusion_audit(args.train_dataset,args.data_root)
    check_overlap(doc['images'],keys,hashes)
    names,data_hashes=model_classes(args.train_dataset,args.data_root)
    source_checkpoint=Path(args.checkpoint).resolve();checkpoint_sha=digest(source_checkpoint)
    if source_checkpoint.stem in ['best_map','best_ap_small'] and source_checkpoint.stem!=args.selection:
        raise ValueError('Checkpoint filename and selection rule differ')
    selected=doc['images'][:args.limit] if args.limit else doc['images']
    scientific=not bool(args.limit)
    if args.limit is not None and args.limit<1:raise ValueError('--limit must be positive')
    protocol={'schema':'paper1-test-export-1.0','manifest_sha256':digest(args.manifest),'test_content_sha256':doc['content_sha256'],
        'method':args.method,'train_dataset':args.train_dataset,'test_dataset':doc['dataset'],'seed':args.seed,'level':args.level,
        'training_data_fingerprints':data_hashes,'class_order':names,'checkpoint_sha256':checkpoint_sha,
        'selection':args.selection,'selection_note':args.selection_note,'transfer':args.transfer,
        'min_size':800,'max_size':1333,'batch_size':args.batch_size,'score_floor':.05,'max_dets':[1,10,100],
        'precision':'FP32','weights':'EMA','image_count':len(selected),'scientific_result':scientific,
        'tool_sources':{str(p.relative_to(ROOT)):digest(p) for p in sorted(ROOT.rglob('*.py')) if '__pycache__' not in p.parts and 'tests' not in p.parts},
        'adapter_sources':{p.name:digest(p) for p in ([Path(args.recent_package)/'reviewer34_compare.py'] if args.method not in LEGACY else
                          [Path(args.lgf_package)/'core_bridge.py',Path(args.lgf_package)/'ugs.py',Path(args.lgf_package)/'vendor/original_core.py'])}}
    protocol_id=signature(protocol);output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    import fcntl
    with open(output/'run.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_file=output/'run.json'
        if run_file.exists():
            if not args.resume:raise FileExistsError('Export exists. Use --resume with the same arguments.')
            if read_json(run_file)['protocol_sha256']!=protocol_id:raise ValueError('Resume checkpoint, data, selection, or inference protocol changed')
            if (output/'export_report.json').exists():return verify_export(output)
        elif args.resume:raise ValueError('No export to resume')
        else:
            if any(p.name!='run.lock' for p in output.iterdir()):raise ValueError('Export output must be empty')
            write_json(run_file,{'protocol':protocol,'protocol_sha256':protocol_id,'source_checkpoint':str(source_checkpoint)})
        frozen=output/('frozen_checkpoint'+source_checkpoint.suffix)
        if not frozen.exists():
            tmp=Path(str(frozen)+'.tmp');shutil.copyfile(source_checkpoint,tmp)
            if digest(tmp)!=checkpoint_sha:raise ValueError('Checkpoint changed while creating the frozen copy')
            os.replace(tmp,frozen)
        if digest(frozen)!=checkpoint_sha:raise ValueError('Frozen checkpoint changed')
        local=copy.copy(args);local.checkpoint=frozen
        model,device,meta,params,infer=load_model(local,names)
        training=meta.get('training_protocol') or {}
        known_train=training.get('train_content_sha256',training.get('training_content_sha256'))
        known_val=training.get('val_content_sha256',training.get('validation_content_sha256'))
        if known_train and known_train!=data_hashes[0]:raise ValueError('Training data differ from checkpoint provenance')
        if known_val and known_val!=data_hashes[1]:raise ValueError('Selection validation data differ from checkpoint provenance')
        batch_dir=output/'batches';batch_dir.mkdir(exist_ok=True);batches=[];started=time.perf_counter()
        for start in range(0,len(selected),args.batch_size):
            subset=selected[start:start+args.batch_size];path=batch_dir/f'{start:08d}.json'
            if path.exists():
                old=read_json(path)
                if old['protocol_sha256']!=protocol_id or old['start']!=start or [r['file_name'] for r in old['images']]!=[r['file_name'] for r in subset]:
                    raise ValueError('Saved inference batch does not match frozen test order')
            else:
                predictions=infer(rgb_tensors(subset,doc['image_root'],device))
                if len(predictions)!=len(subset):raise ValueError('Model returned a different batch size')
                rows=[]
                for im,pred in zip(subset,predictions):
                    detections=[]
                    boxes=pred['boxes'].detach().cpu().numpy();scores=pred['scores'].detach().cpu().numpy();labels=pred['labels'].detach().cpu().numpy()
                    if not len(boxes)==len(scores)==len(labels):raise ValueError('Prediction fields have different lengths')
                    if len(boxes)>100:raise ValueError('Detector exceeded the original detection cap')
                    for box,score,label in zip(boxes,scores,labels):
                        if not all(math.isfinite(float(v)) for v in [*box,score]):raise ValueError('Nonfinite detector output')
                        if int(label)!=label or int(label) not in range(6):raise ValueError('Unexpected model class label')
                        x1,y1,x2,y2=box
                        if x2<x1 or y2<y1 or not 0<=score<=1:raise ValueError('Invalid box or confidence')
                        detections.append({'category_name':names[int(label)],'bbox':[float(x1),float(y1),float(x2-x1),float(y2-y1)],'score':float(score)})
                    rows.append({'image_id':im['id'],'file_name':im['file_name'],'condition':im.get('condition'),'detections':detections})
                write_json(path,{'protocol_sha256':protocol_id,'start':start,'images':rows})
            batches.append({'file':str(path.relative_to(output)),'sha256':digest(path)})
            write_json(output/'progress.json',{'completed_images':start+len(subset),'total_images':len(selected),'protocol_sha256':protocol_id})
            print(f'TEST EXPORT {args.method} {start+len(subset)}/{len(selected)}',flush=True)
        neutral=output/'predictions_by_filename.json'
        json_array(neutral,iter_batches(output,batches))
        official=None
        if doc['server_mapping_verified'] and scientific:
            cat_ids={c['name']:c['id'] for c in doc['categories']}
            official=output/'predictions_coco.json'
            json_array(official,({'image_id':r['image_id'],'category_id':cat_ids[r['category_name']],'bbox':r['bbox'],'score':r['score']}
                                 for r in prediction_rows(iter_batches(output,batches))))
            zip_predictions(output/'submission.zip',official,'detections_test-dev2017_paper1_results.json')
        result={'schema':'paper1-test-export-report-1.0','protocol':protocol,'protocol_sha256':protocol_id,
            'model_metadata':meta,'parameter_count':params,'checkpoint':str(frozen),'checkpoint_sha256':checkpoint_sha,
            'manifest':str(Path(args.manifest).resolve()),'manifest_sha256':digest(args.manifest),'batches':batches,
            'prediction_file':str(neutral),'prediction_sha256':digest(neutral),
            'official_prediction_file':str(official) if official else None,
            'official_prediction_sha256':digest(official) if official else None,
            'submission_zip':str(output/'submission.zip') if official else None,
            'submission_zip_sha256':digest(output/'submission.zip') if official else None,
            'metrics':None,'metric_status':'Not scored. Hidden labels require the benchmark server.' if not doc['ground_truth_available'] else 'Run score for the supplied labelled test.',
            'server_mapping_verified':doc['server_mapping_verified'],'scientific_result':scientific,
            'submission_scope':doc.get('submission_scope',doc['benchmark']),
            'current_session_inference_seconds_including_io':time.perf_counter()-started,
            'complete_scene_history_verified':False}
        write_json(output/'export_report.json',result)
        print('EXPORT COMPLETE',output,flush=True)
        return result

def verify_export(output):
    report=read_json(Path(output)/'export_report.json')
    if signature(report['protocol'])!=report['protocol_sha256']:raise ValueError('Export protocol changed')
    if digest(report['checkpoint'])!=report['checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
    if digest(report['manifest'])!=report['manifest_sha256']:raise ValueError('Test manifest changed')
    if digest(report['prediction_file'])!=report['prediction_sha256']:raise ValueError('Predictions changed')
    for p,h in [('official_prediction_file','official_prediction_sha256'),('submission_zip','submission_zip_sha256')]:
        if report.get(p) and digest(report[p])!=report[h]:raise ValueError('Submission artifact changed')
    return report

def official_image_map(info, categories_doc=None):
    if not isinstance(info.get('images'),list) or not info['images']:raise ValueError('Official test metadata must contain images')
    mapping={};ids=set()
    for im in info['images']:
        name=Path(im['file_name']).name
        if name in mapping or str(im['id']) in ids:raise ValueError('Official metadata has ambiguous filenames or IDs')
        mapping[name]=im;ids.add(str(im['id']))
    cats=(categories_doc or info).get('categories',[])
    names={c['name']:c['id'] for c in cats}
    if len(names)!=len(cats) or len(set(names.values()))!=len(names):raise ValueError('Ambiguous official categories')
    if not set(CLASSES)<=set(names):raise ValueError('Official category metadata must define the six named classes')
    return mapping,names

def package_acdc(args):
    output=Path(args.export);report=verify_export(output);doc=read_json(report['manifest'])
    if doc['dataset']!='acdc' or not report['scientific_result']:raise ValueError('Requires a complete ACDC test export')
    if not args.image_info:raise ValueError('Supply official ACDC test image metadata. Server IDs are never inferred from file order.')
    info=read_json(args.image_info);category_doc=read_json(args.categories_json) if args.categories_json else None
    mapping,categories=official_image_map(info,category_doc)
    exported_names={Path(im['file_name']).name for im in doc['images']}
    if not exported_names<=set(mapping):raise ValueError('Official metadata is missing exported test images')
    for im in doc['images']:
        official=mapping[Path(im['file_name']).name]
        if (official['width'],official['height'])!=(im['width'],im['height']):raise ValueError('Official test dimensions differ')
    destination=Path(args.output or output/'acdc_submission')
    if destination.exists():raise FileExistsError('Submission output exists; choose a new directory')
    destination.mkdir(parents=True)
    neutral=read_json(report['prediction_file'])
    groups={'all':neutral,**{c:[im for im in neutral if im['condition']==c] for c in ['fog','night','rain','snow']}}
    manifests=[]
    for condition,images in groups.items():
        prediction_file=destination/f'{condition}.json'
        json_array(prediction_file,({'image_id':mapping[Path(r['file_name']).name]['id'],'category_id':categories[r['category_name']],
                                    'bbox':r['bbox'],'score':r['score']} for r in prediction_rows(images)))
        path=destination/f'{condition}.zip';zip_predictions(path,prediction_file)
        if path.stat().st_size>100_000_000:raise ValueError('ACDC detection ZIP exceeds the documented 100 MB limit')
        manifests.append({'condition':condition,'image_count':len(images),'zip':str(path),'sha256':digest(path)})
    result={'source_export':str(output/'export_report.json'),'source_export_sha256':digest(output/'export_report.json'),
            'official_image_info':str(Path(args.image_info).resolve()),'official_image_info_sha256':digest(args.image_info),
            'official_categories_sha256':digest(args.categories_json or args.image_info),'metadata_provenance':args.metadata_provenance,
            'category_mapping':{n:categories[n] for n in CLASSES},'submissions':manifests,'metrics':None,
            'metric_scope_note':'Eight-class server aggregate is not six-class Paper 1 AP. Obtain explicit six-class AP/APsmall from the server/provider.'}
    write_json(destination/'submission_report.json',result);print('ACDC SUBMISSIONS READY',destination,flush=True)
    return result

def coco_score(doc, predictions, selected_ids=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np
    validate_labels(doc)
    # COCOeval reserves annotation ID zero for unmatched detections. Normalize
    # identifiers in an evaluation-only copy, retaining every box/area/category.
    doc=copy.deepcopy(doc)
    for index,annotation in enumerate(doc['annotations'],1):annotation['id']=index
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO();gt.dataset=doc;gt.dataset.setdefault('info',{});gt.createIndex()
        if predictions:dt=gt.loadRes(predictions)
        else:
            dt=COCO();dt.dataset={'images':doc['images'],'categories':doc['categories'],'annotations':[]};dt.createIndex()
        ev=COCOeval(gt,dt,'bbox');ev.params.catIds=[c['id'] for c in doc['categories'] if c['name'] in CLASSES]
        ev.params.imgIds=selected_ids if selected_ids is not None else [im['id'] for im in doc['images']]
        ev.params.maxDets=[1,10,100];ev.params.iouThrs=np.linspace(.5,.95,10);ev.evaluate();ev.accumulate();ev.summarize()
    return dict(zip(METRICS,[float(v) for v in ev.stats]))

def score(args):
    report=verify_export(args.export);manifest=read_json(report['manifest'])
    if not report['scientific_result']:raise ValueError('Partial software exports are not scientific test results')
    if not manifest['ground_truth_available']:
        raise ValueError('Official test ground truth is hidden. Submit exported predictions; local AP cannot be calculated from image-info JSON.')
    verify_manifest(manifest);ground_truth=read_json(manifest['annotations'])
    cats={c['name']:c['id'] for c in ground_truth['categories'] if c['name'] in CLASSES}
    by_filename=read_json(report['prediction_file'])
    predictions=[{'image_id':r['image_id'],'category_id':cats[r['category_name']],'bbox':r['bbox'],'score':r['score']} for r in prediction_rows(by_filename)]
    metrics=coco_score(ground_truth,predictions)
    condition_metrics={}
    for condition in sorted({im['condition'] for im in manifest['images'] if im.get('condition')}):
        ids=[im['id'] for im in manifest['images'] if im.get('condition')==condition]
        condition_metrics[condition]=coco_score(ground_truth,predictions,ids)
    p=report['protocol'];evaluation={k:p[k] for k in ['min_size','max_size','batch_size','score_floor','max_dets','precision','weights']}
    result={'schema':'paper1-labelled-test-metrics-1.0','method':p['method'],'seed':p['seed'],'selection':p['selection'],
        'dataset':p['test_dataset'],'train_dataset':p['train_dataset'],'transfer':p['transfer'],'metrics':metrics,
        'condition_metrics':condition_metrics,'class_names':CLASSES,'AP_scale':'0-1','parameter_count':report['parameter_count'],
        'evaluation_signature':signature({'test':p['test_content_sha256'],'evaluation':evaluation,'classes':CLASSES}),
        'export':str(Path(args.export).resolve()),'export_report_sha256':digest(Path(args.export)/'export_report.json'),
        'scientific_result':True,'evaluation_split':'test','complete_scene_history_verified':False}
    path=Path(args.output or Path(args.export)/'test_metrics.json')
    if path.exists():raise FileExistsError('Test metrics output exists')
    write_json(path,result);print(json.dumps(metrics,indent=2));return result

def compare(args):
    reports=[read_json(p) for p in args.reports]
    if len(reports)<2:raise ValueError('Supply at least two labelled test reports')
    for r in reports:
        if r.get('schema')!='paper1-labelled-test-metrics-1.0' or not r.get('scientific_result'):raise ValueError('Only scored labelled tests are accepted; exports/server aggregates are not six-class metrics')
        verify_export(r['export'])
        if digest(Path(r['export'])/'export_report.json')!=r['export_report_sha256']:raise ValueError('Export provenance changed')
    if len({r['evaluation_signature'] for r in reports})!=1 or len({r['selection'] for r in reports})!=1:raise ValueError('Test/evaluation/selection rules differ')
    groups=collections.defaultdict(list)
    for r in reports:groups[(r['method'],r['train_dataset'],r['transfer'])].append(r)
    if len(groups)<2:raise ValueError('Supply at least two method/training configurations')
    seeds=None;rows=[]
    for (method,training,transfer),items in groups.items():
        group_seeds=[r['seed'] for r in items]
        if len(set(group_seeds))!=len(group_seeds):raise ValueError('Duplicate method/seed report')
        if seeds is None:seeds=set(group_seeds)
        elif seeds!=set(group_seeds):raise ValueError('Methods need matching seed sets')
        configs=[]
        for item in items:
            exported=read_json(Path(item['export'])/'export_report.json')
            config=copy.deepcopy(exported['model_metadata'].get('training_protocol') or exported['model_metadata'].get('original_config'))
            if isinstance(config,dict):
                config.pop('seed',None)
                if isinstance(config.get('recipe'),dict):
                    config['recipe'].pop('seed',None);config['recipe'].pop('project',None)
            configs.append(signature({'config':config,'training_data':exported['protocol']['training_data_fingerprints'],'level':exported['protocol']['level']}))
        if len(set(configs))!=1:raise ValueError('Training configuration/data differ within a method seed group')
        row={'method':method,'train_dataset':training,'transfer':transfer,'seeds':sorted(group_seeds),'n':len(items)}
        for metric in METRICS:
            values=[r['metrics'][metric] for r in items]
            row[metric+'_mean']=statistics.mean(values) if all(v>=0 for v in values) else None
            row[metric+'_std']=statistics.stdev(values) if len(values)>1 and all(v>=0 for v in values) else None
        rows.append(row)
    output=Path(args.output)
    if output.exists():raise FileExistsError('Comparison output exists')
    output.mkdir(parents=True);write_json(output/'comparison.json',{'rows':rows,'AP_scale':'0-1','selection':reports[0]['selection'],'evaluation_split':'test'})
    with open(output/'comparison.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('TEST COMPARISON',output)

def arguments():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    e=sub.add_parser('export');e.add_argument('--manifest',type=Path,required=True);e.add_argument('--checkpoint',type=Path,required=True)
    e.add_argument('--method',choices=METHODS,required=True);e.add_argument('--train-dataset',choices=['coco_nw','coco_weather','acdc'])
    e.add_argument('--seed',type=int,default=42);e.add_argument('--level',choices=['C3','C4','C5'],default='C3')
    e.add_argument('--selection',choices=['best_map','best_ap_small'],required=True);e.add_argument('--selection-note',default='User must verify original validation selection history')
    e.add_argument('--data-root',default=DEFAULT_DATA_ROOT);e.add_argument('--device',default='0');e.add_argument('--batch-size',type=int,default=4)
    e.add_argument('--threads',type=int,default=4);e.add_argument('--resume',action='store_true');e.add_argument('--transfer',action='store_true')
    e.add_argument('--limit',type=int,help='Software check only; no scientific scores or server ZIP')
    e.add_argument('--output',type=Path,required=True);e.add_argument('--runtime-root',default='/tmp/asera/paper1_original/test_runtime')
    e.add_argument('--lgf-package',default=str(ROOT.parent/'paper1_ugs_comparison_v1'));e.add_argument('--recent-package',default=str(ROOT.parent/'paper1_reviewer34_v1'))
    a=sub.add_parser('package-acdc');a.add_argument('--export',type=Path,required=True);a.add_argument('--image-info',type=Path,required=True)
    a.add_argument('--categories-json',type=Path);a.add_argument('--metadata-provenance',required=True);a.add_argument('--output',type=Path)
    s=sub.add_parser('score');s.add_argument('--export',type=Path,required=True);s.add_argument('--output',type=Path)
    c=sub.add_parser('compare');c.add_argument('--reports',type=Path,nargs='+',required=True);c.add_argument('--output',type=Path,required=True)
    return p.parse_args()

def main():
    args=arguments()
    {'export':export,'package-acdc':package_acdc,'score':score,'compare':compare}[args.action](args)

if __name__=='__main__':main()

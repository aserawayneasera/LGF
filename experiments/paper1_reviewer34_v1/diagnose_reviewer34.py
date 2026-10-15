#!/usr/bin/env python3
"""Run runtime probes in separate processes so a native crash leaves a useful log."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def probe(kind, device):
    import faulthandler
    faulthandler.enable(all_threads=True)
    print(f'PROBE START {kind} device={device}', flush=True)
    import torch
    torch.set_num_threads(2)
    dev = torch.device('cpu' if device == 'cpu' else 'cuda:'+device)
    if dev.type == 'cuda':
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        torch.cuda.set_device(dev)
    print(f'TORCH {torch.__version__} FROM {torch.__file__}', flush=True)
    print(f'CUDA BUILD {torch.version.cuda}; CUDNN {torch.backends.cudnn.version()}', flush=True)
    if kind == 'tensor':
        a = torch.arange(64, dtype=torch.float32, device=dev).reshape(8,8)
        torch.testing.assert_close((a @ a.T).cpu(), a.cpu() @ a.cpu().T)
    elif kind == 'convolution':
        from reviewer34_compare import fixed_inference_settings
        layer = torch.nn.Conv2d(16,32,3,padding=1).to(dev)
        x = torch.ones(2,16,64,64,device=dev,requires_grad=True)
        print('CONV forward/backward',flush=True)
        layer(x).square().mean().backward()
        if not torch.isfinite(x.grad).all(): raise AssertionError('Nonfinite gradient')
        with fixed_inference_settings(), torch.no_grad():
            print('CONV evaluation',flush=True)
            y = layer.eval()(x.detach())
            if not torch.isfinite(y).all(): raise AssertionError('Nonfinite convolution')
    elif kind == 'nms':
        import torchvision
        print(f'TORCHVISION {torchvision.__version__} FROM {torchvision.__file__}',flush=True)
        boxes = torch.tensor([[0,0,10,10],[1,1,9,9],[20,20,30,30]],dtype=torch.float32,device=dev)
        scores = torch.tensor([.9,.8,.7],dtype=torch.float32,device=dev)
        print('NMS small nonempty input',flush=True)
        actual = torchvision.ops.nms(boxes,scores,.5).cpu().tolist()
        if actual != [0,2]: raise AssertionError(f'NMS output differs: {actual}')
        # Exercise the compiled operator with more than one CUDA bitmask block.
        x = torch.arange(1024,dtype=torch.float32,device=dev)*12
        many = torch.stack([x,x,x+10,x+10],dim=1)
        rank = torch.arange(1024,0,-1,dtype=torch.float32,device=dev)
        print('NMS 1024 nonoverlapping boxes',flush=True)
        kept = torchvision.ops.nms(many,rank,.5).cpu().tolist()
        if kept != list(range(1024)): raise AssertionError('NMS large-input output differs')
    else: raise ValueError(kind)
    if dev.type == 'cuda': torch.cuda.synchronize(dev)
    print(f'PROBE PASS {kind} device={device}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='0')
    p.add_argument('--output-root',default='/tmp/asera/paper1_original/reviewer34/diagnostics')
    p.add_argument('--probe',choices=['tensor','convolution','nms'])
    p.add_argument('--skip-smoke',action='store_true',help='Only low-level probes; not a training pass')
    a=p.parse_args()
    if a.device!='cpu' and not a.device.isdigit(): p.error('Use one GPU index or cpu')
    if a.probe:
        probe(a.probe,a.device)
        return
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=Path(a.output_root).resolve()/f'run_{stamp}'
    out.mkdir(parents=True)
    script=Path(__file__).resolve()
    env=os.environ.copy()
    env['PYTHONFAULTHANDLER']='1'
    env['CUDA_LAUNCH_BLOCKING']='1'
    env.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    print(f'DIAGNOSTIC OUTPUT {out}',flush=True)
    jobs=[('environment',[str(script.with_name('reviewer34_compare.py')),'check','--device',a.device])]
    for device in (['cpu'] if a.device=='cpu' else ['cpu',a.device]):
        for kind in ['tensor','convolution','nms']:
            jobs.append((f'{kind}_{device}',[str(script),'--probe',kind,'--device',device]))
    if not a.skip_smoke:
        for model in ['yolo11m','rtdetr-l']:
            jobs.append((f'smoke_{model}',[str(script.with_name('reviewer34_compare.py')),'smoke',
                '--model',model,'--device',a.device,'--work-root',str(out/'smoke_work'),
                '--output',str(out/f'{model}_smoke.json')]))
    records=[]
    for name,command in jobs:
        log=out/f'{name}.log'
        print(f'RUN {name}; log={log}',flush=True)
        with log.open('w') as f:
            result=subprocess.run([sys.executable,'-X','faulthandler','-u',*command],
                                  stdout=f,stderr=subprocess.STDOUT,env=env,cwd=script.parent)
        record={'name':name,'returncode':result.returncode,'log':str(log),
                'status':'passed' if result.returncode==0 else 'failed'}
        if result.returncode<0:record['signal']=-result.returncode
        records.append(record)
        summary={'python':sys.executable,'device':a.device,'diagnostic_cuda_launch_blocking':True,
                 'checks':records,'complete':False,'scientific_result':False}
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(f'{record["status"].upper()} {name} returncode={result.returncode}',flush=True)
        if result.returncode:
            print('\n'.join(log.read_text(errors='replace').splitlines()[-35:]),flush=True)
    passed=all(r['returncode']==0 for r in records)
    summary.update(complete=True,status='passed' if passed else 'failed',
                   full_smoke_requested=not a.skip_smoke)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    archive=out/'diagnostic_report.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.suffix in {'.json','.log'}:
                z.write(path,str(path.relative_to(out)))
    print(f'DIAGNOSTICS {summary["status"].upper()} {out}/summary.json',flush=True)
    print(f'REPORT ZIP {archive}',flush=True)
    print('These checks use synthetic data. Full experiments have not been started.',flush=True)
    if not passed:sys.exit(1)


if __name__=='__main__':main()

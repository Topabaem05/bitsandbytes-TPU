"""Private batch adapter. The original scientific modules stay unchanged."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import shutil
import zipfile
import time
from types import SimpleNamespace

DELTA = 'BATCH_LOCAL_CPU_GATE_HOST_REVIEW_AFTER_DEVICE'
LABELS = ['01-python-bootstrap','02-create-venv','03-download-wheels','04-install-wheels',
          '05-verify-runtime-wheels','06-build-upstream','07-build-plugin','08-install-source-wheels',
          '09-installed-metadata','09-transfer-installed-source','10-transfer-source-controls',
          '11-cpu-oracle','11b-local-cpu-gate','10-runtime-probe','12-transfer','13-local-science-verify']
MAX_OUTPUT = 100 * 1024 * 1024

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n')
def need(v,code):
    if not v: raise ValueError(code)
def load(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def science(payload,records,manifest):
    T=load(payload/'probe_transfer.py','m8_original_transfer')
    args=SimpleNamespace(backend_probe=payload/'probe_backend.py',route_probe=payload/'probe_routes.py',
        precision_probe=payload/'probe_precision.py',patch_manifest=payload/'patches/params4bit-xla-v1.json',
        admission_sha256=manifest['source_admission_sha256'],oracle=records/'cpu-oracle',
        oracle_sha256=sha(records/'cpu-oracle/oracle-seal.json'),actual=records/'transfer')
    B,R,P,A=T.helpers(args)
    return T,args,(B,R,P,A)

def cpu_audit(payload,records,manifest):
    T,args,(B,R,P,A)=science(payload,records,manifest)
    seal=T.verify_oracle(B,A,args)
    for case in P.selection(B,R):
        raw=B.read(args.oracle/'raw'/(case['id']+'.json'))
        if case['dtype']=='float32':
            witness=P.scalar_forward(case,raw)
            need(B.numeric(raw['outputs']['y']['values'],witness['values'],R.tolerance(B.load_spec()[0],case,'y'))['status']=='PASS','CPU_SCALAR_FORWARD')
        R.gradient_math(case,raw)
    return {'kind':'M8_BATCH_LOCAL_CPU_GATE','status':'PASS','orchestration':DELTA,
            'host_pre_device_review':False,'oracle_sha256':args.oracle_sha256,
            'source_admission_sha256':args.admission_sha256,'case_count':len(seal['case_ids']),
            'method':seal['oracle_method'],'pid':os.getpid(),'pgid':os.getpgid(0)}

def scientific_audit(payload,records,manifest):
    T,args,helpers=science(payload,records,manifest)
    return T.verify(*helpers,args)

def validate_packet(payload,manifest):
    need(manifest.get('experiment')=='transfer-api42','M2_EXPERIMENT')
    for name,row in manifest['files'].items():
        p=payload/name
        need(not p.is_symlink() and p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'PACKET_SOURCE:'+name)
    remote=load(payload/'cloud/remote.py','m8_fixed_remote')
    remote.verify_experiment(manifest);remote.verify_transfer_payload(payload,manifest)
    T=load(payload/'probe_transfer.py','m8_packet_T');B,R,P,A=T.helpers(SimpleNamespace(
        backend_probe=payload/'probe_backend.py',route_probe=payload/'probe_routes.py',
        precision_probe=payload/'probe_precision.py',patch_manifest=payload/'patches/params4bit-xla-v1.json'))
    admission=read(payload/'source-admission.json');A.validate_admission(B,admission,A.load_manifest(B,payload/'patches/params4bit-xla-v1.json'))
    need(admission['bitsandbytes_tpu']['files']==read(payload/'plugin/source-manifest.json')['installed_python_files'],'PLUGIN_ADMISSION_MAP')
    need(sha(payload/'plugin/source-manifest.json')==manifest['plugin_source_manifest_sha256'],'PLUGIN_MANIFEST')
    for row in read(payload/'plugin/source-manifest.json')['files']:
        need(sha(payload/'plugin'/row['path'])==row['sha256'],'PLUGIN_SOURCE')
    return remote

class FixedBatch:
    synthetic=False
    def __init__(self,base,manifest,packet_sha,deadline):
        self.base=base;self.payload=base/'payload';self.out=base/'records';self.manifest=manifest
        self.remote=validate_packet(self.payload,manifest);self.packet_sha=packet_sha
        self.epoch=deadline-3600;self.deadline=deadline-660
    def phase(self,label):
        R=self.remote;b=self.base;p=self.payload;o=self.out;m=self.manifest;installed=b/'venv/bin/python'
        if label=='install':
            need(R.execute(b,self.packet_sha,self.epoch,'install')==0,'INSTALL_BLOCKED')
            need(read(o/'receipt.json')['installation_status']=='PASS','INSTALL_RECEIPT')
            for name in ['bootstrap-result.json','archive-inventory.json']:
                (o/'bootstrap').mkdir(exist_ok=True);shutil.copyfile(b/'bootstrap'/name,o/'bootstrap'/name)
            (o/'built-wheels').mkdir()
            for directory in ['built-upstream','built-plugin']:
                wheels=list((b/directory).glob('*.whl'));need(len(wheels)==1,'BUILT_WHEEL_COUNT')
                shutil.copyfile(wheels[0],o/'built-wheels'/wheels[0].name)
            wheel=next((b/'built-plugin').glob('*.whl'))
            with zipfile.ZipFile(wheel) as z:
                files={n.removeprefix('bitsandbytes_tpu/'):hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('bitsandbytes_tpu/') and n.endswith('.py')}
            need(files==read(p/'source-admission.json')['bitsandbytes_tpu']['files'],'BUILT_PLUGIN_MAP')
            write(o/'built-plugin-source.json',{'wheel_sha256':sha(wheel),'python_files':files,'status':'EXACT_PLUGIN_WHEEL_PASS'})
            return
        receipt=read(o/'receipt.json')
        if label=='cpu':
            receipt.update(science_start_epoch=time.time(),science_deadline_epoch=min(self.deadline,time.time()+1800))
            R.durable_json(o/'receipt.json',receipt)
            deadline=receipt['science_deadline_epoch']
            steps=[('10-transfer-source-controls',[installed,'-B',p/'tests/test_transfer_source.py','--base-source',b/'upstream-base','--patched-source',b/'upstream-patched-controls','--output',o/'source-controls.json'],300),
                   ('11-cpu-oracle',[installed,'-B',p/'probe_transfer.py','prepare','--admission',p/'source-admission.json','--admission-sha256',m['source_admission_sha256'],'--output',o/'cpu-oracle','--backend-probe',p/'probe_backend.py','--route-probe',p/'probe_routes.py','--precision-probe',p/'probe_precision.py','--patch-manifest',p/'patches/params4bit-xla-v1.json'],300),
                   ('11b-local-cpu-gate',[installed,'-B',p/'batch.py','cpu-audit','--base',b],60)]
            for name,argv,limit in steps:
                rec=R.run_step(o,name,[str(x) for x in argv],deadline,limit,cwd=p)
                receipt['steps'].append({'label':name,**rec});R.durable_json(o/'receipt.json',receipt)
                need(rec['status']=='PASS' and rec['exit_code']==0,'CPU_STEP_BLOCKED:'+name)
                if name=='10-transfer-source-controls':R.validate_source_controls(read(o/'source-controls.json'))
            gate=read(o/'batch-cpu-gate.json');need(gate['status']=='PASS' and gate['orchestration']==DELTA and gate['host_pre_device_review'] is False,'CPU_GATE')
            receipt.update(source_controls_status='PASS_QUALIFIED_LINUX_SOURCE_CONTROLS',cpu_status='PASS',
                status='CPU_ORACLE_READY_TPU_NOT_RUN',oracle_sha256=sha(o/'cpu-oracle/oracle-seal.json'))
            R.durable_json(o/'receipt.json',receipt);return
        if label=='runtime':
            need(receipt.get('cpu_status')=='PASS' and (o/'batch-cpu-gate.json').is_file(),'CPU_BEFORE_DEVICE')
            rec=R.run_step(o,'10-runtime-probe',[str(installed),'-B',str(p/'runtime/probe_runtime.py'),'--out',str(o/'runtime-probe.json')],receipt['science_deadline_epoch'],120,cwd=p,tpu=True)
            receipt['steps'].append({'label':'10-runtime-probe',**rec})
            R.durable_json(o/'receipt.json',receipt)
            need(rec['status']=='PASS' and read(o/'runtime-probe.json')['status']=='PASS_TPU_RUNTIME_PROBE_ONLY','RUNTIME_BLOCKED')
            receipt.update(runtime_status='PASS_TPU_RUNTIME_PROBE_ONLY');R.durable_json(o/'receipt.json',receipt);return
        if label=='transfer':
            write(b/'launch.json',{'kind':'M8_BATCH_LOCAL_ORACLE_LAUNCH','orchestration':DELTA,
                'oracle_sha256':receipt['oracle_sha256'],'gate_sha256':sha(o/'batch-cpu-gate.json')})
            need(R.execute(b,self.packet_sha,self.epoch,'transfer')==0,'TRANSFER_INVALID');return
        if label=='verify':
            rec=R.run_step(o,'13-local-science-verify',[str(installed),'-B',str(p/'batch.py'),'science-audit','--base',str(b)],receipt['science_deadline_epoch'],60,cwd=p)
            receipt['steps'].append({'label':'13-local-science-verify',**rec});R.durable_json(o/'receipt.json',receipt)
            need(rec['status']=='PASS','LOCAL_SCIENCE_INVALID');return
        raise ValueError('UNKNOWN_BATCH_PHASE')

def export(records,output,binding):
    members={}
    for p in sorted(records.rglob('*')):
        need(not p.is_symlink(),'OUTPUT_SYMLINK')
        if p.is_file():
            need(p.stat().st_nlink==1,'OUTPUT_HARDLINK')
            members['records/'+p.relative_to(records).as_posix()]={'bytes':p.stat().st_size,'sha256':sha(p)}
    need(len(members)<=2000 and sum(r['bytes'] for r in members.values())<=MAX_OUTPUT,'OUTPUT_BOUND')
    output.mkdir(parents=True,exist_ok=True)
    archive=output/'m8-evidence.tar';need(not archive.exists(),'FRESH_EXPORT')
    with tarfile.open(archive,'w:') as tar:
        for name in members:
            data=(records/name.removeprefix('records/')).read_bytes();item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o600;tar.addfile(item,io.BytesIO(data))
    need(archive.stat().st_size<=MAX_OUTPUT,'FINAL_TAR_BYTE_BOUND')
    write(output/'m8-evidence-manifest.json',{'format':'m8-evidence-v1','binding':binding,
        'archive':{'file':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive)},'members':members})

def execute(base,output,binding,deadline,*,provider=None):
    need(type(deadline) in (int,float) and math.isfinite(deadline),'DEADLINE')
    base=Path(base);output=Path(output);payload=base/'payload';records=base/'records';records.mkdir(exist_ok=True)
    manifest=read(payload/'manifest.json');validate_packet(payload,manifest)
    need(binding['packet_sha256']==sha(base/'payload.zip'),'BINDING_PACKET')
    for key,path in [('runtime_lock_sha256','runtime/requirements.lock.json'),('profile_sha256','probe-profile.json'),('input_sha256','probe-inputs.json'),('source_manifest_sha256','plugin/source-manifest.json')]:
        need(binding[key]==sha(payload/path),'BINDING_'+key)
    need(time.time()<deadline-660,'INSUFFICIENT_WORK_BUDGET')
    provider=provider or FixedBatch(base,manifest,binding['packet_sha256'],deadline)
    record={'kind':'M8_M2_BATCH','status':'RUNNING','binding':binding,'deadline_epoch':deadline,
        'work_deadline_epoch':deadline-660,'orchestration':DELTA,'host_pre_device_review':False,
        'execution_mode':'SYNTHETIC_FIXTURE' if provider.synthetic else 'ACTUAL_REQUIRED',
        'm8_status':'NOT_QUALIFIED','partial_m8_scope':'M2_API42_TRANSFER4_ONLY','phases':[],
        'pid':os.getpid(),'pgid':os.getpgid(0),'parent_pid':os.getppid(),'work_root':str(base.resolve()),'error':None}
    try:
        for label in ['install','cpu','runtime','transfer','verify']:
            need(time.time()<deadline-660,'WORK_DEADLINE')
            start=time.time()
            with (records/('phase-'+label+'.stdout.raw')).open('w') as log,contextlib.redirect_stdout(log):provider.phase(label)
            record['phases'].append({'phase':label,'status':'COMPLETE','start_epoch':start,'end_epoch':time.time()})
            write(records/'batch.json',record)
        record['status']='COMPLETE'
    except BaseException as error:
        record.update(status='INVALID_OR_BLOCKED',error={'type':type(error).__name__,'message':str(error)})
    finally:
        write(records/'batch.json',record)
        export(records,output,binding)
    return record

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['cpu-audit','science-audit']);parser.add_argument('--base',type=Path,required=True);a=parser.parse_args()
    m=read(a.base/'payload/manifest.json');validate_packet(a.base/'payload',m)
    value=cpu_audit(a.base/'payload',a.base/'records',m) if a.command=='cpu-audit' else scientific_audit(a.base/'payload',a.base/'records',m)
    write(a.base/'records'/('batch-cpu-gate.json' if a.command=='cpu-audit' else 'batch-science-verify.json'),value)

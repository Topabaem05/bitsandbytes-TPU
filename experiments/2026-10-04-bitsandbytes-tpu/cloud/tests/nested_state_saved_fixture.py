"""Portable original-source CPU controls and isolated saved-record negatives."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import socket
import struct
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];SCIENCE=HERE.parents[1]
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
S=load(SCIENCE/'probe_nested_state.py','tested_nested_state')
ARGS=SimpleNamespace(backend_probe=SCIENCE/'probe_backend.py',route_probe=SCIENCE/'probe_routes.py',precision_probe=SCIENCE/'probe_precision.py',transfer_admission=SCIENCE/'transfer_admission.py',nested_probe=SCIENCE/'probe_nested.py',patch_manifest=ROOT/'patches/params4bit-xla-v1.json',admission_sha256='a'*64)
B,R,P,A,N=S.helpers(ARGS);DATA,SOURCE,SCHEMAS=N.spec(B);PROFILE,_=B.load_spec()
NT=load(HERE/'nested_saved_fixture.py','nested_fixture_tools')

def state_from_reference(case,reference):
 o=reference['outputs'];state={key:copy.deepcopy(o[name]) for name,key in [('packed','weight'),('scale_codes','weight.absmax'),('second_scales','weight.nested_absmax'),('dynamic_map','weight.nested_quant_map'),('nf4_map','weight.quant_map'),('bias','bias')] if name in o}
 meta={'quant_type':'nf4','blocksize':64,'dtype':case['dtype'],'shape':case['weight_shape'],'nested_blocksize':256,'nested_dtype':'float32','nested_offset':o['offset']['values'][0]}
 values=list(json.dumps(meta).encode());state['weight.quant_state.bitsandbytes__nf4']={'shape':[len(values)],'dtype':'uint8','values':values,'bytes_sha256':__import__('hashlib').sha256(bytes(values)).hexdigest()}
 return state

def tensors(state):
 import torch
 return {k:torch.tensor(v['values'],dtype=getattr(torch,v['dtype'])).reshape(v['shape']) for k,v in state.items()}

def reseal(fixture,oracle=False):
 for phase in ('save','restore'):
  root=fixture.actual/phase;rec=B.read(root/'receipt.json');rec['artifacts']=B.inventory(root,'receipt.json');B.write(root/'receipt.json',rec)
 if oracle:
  rec=B.read(fixture.oracle/'oracle-seal.json');rec['artifacts']=B.inventory(fixture.oracle,'oracle-seal.json');B.write(fixture.oracle/'oracle-seal.json',rec);fixture.oracle_sha256=B.sha(fixture.oracle/'oracle-seal.json')
 parent=B.read(fixture.actual/'receipt.json');parent['oracle_sha256']=fixture.oracle_sha256
 save=B.read(fixture.actual/'save/receipt.json');restore=B.read(fixture.actual/'restore/receipt.json');restore.update(saved_sha256=B.sha(fixture.actual/'save/receipt.json'),saved_pid=save['pid'],saved_process_token=save['process_token'],oracle_sha256=fixture.oracle_sha256);B.write(fixture.actual/'restore/receipt.json',restore)
 save['oracle_sha256']=fixture.oracle_sha256;B.write(fixture.actual/'save/receipt.json',save)
 # Restore link follows final save receipt bytes, then all per-row links and seals.
 restore.update(saved_sha256=B.sha(fixture.actual/'save/receipt.json'));B.write(fixture.actual/'restore/receipt.json',restore)
 for case in S.cases(B,N):
  path=fixture.actual/'restore/raw'/(case['id']+'.json');raw=B.read(path);raw['saved_receipt_sha256']=B.sha(fixture.actual/'save/receipt.json');B.write(path,raw)
 restore['artifacts']=B.inventory(fixture.actual/'restore','receipt.json');B.write(fixture.actual/'restore/receipt.json',restore)
 for launch in parent['children']:launch['receipt_sha256']=B.sha(fixture.actual/launch['phase']/'receipt.json')
 parent['artifacts']=B.inventory(fixture.actual,'receipt.json');B.write(fixture.actual/'receipt.json',parent)

def make_fixture(base,reference,admission='a'*64):
 import torch
 base=Path(base);base.mkdir(parents=True,exist_ok=True);nested=NT.make_fixture(base/'nested',reference,admission=admission)
 fixture=SimpleNamespace(**vars(ARGS));fixture.admission_sha256=admission;fixture.nested_oracle=nested.oracle;fixture.nested_oracle_sha256=nested.oracle_sha256;fixture.oracle=base/'oracle';fixture.actual=base/'actual';fixture.oracle.mkdir();fixture.actual.mkdir()
 gold=dict(S.binding(B,N,fixture),status='COMPLETE',pid=2001,pgid=2001,process_token='a'*32,source_pre=admission,source_post=admission,public_api=dict(N.PUBLIC,original_state_method_identity=True),schemas=SCHEMAS,runtime={'python':'3.12','torch':'2.9.0+cpu','platform':'linux','machine':'x86_64'},oracle_method='NESTED79_DEFAULT_BODY_SEAL_PLUS_ORIGINAL_PUBLIC_PACKED_STATE')
 now=time.time();parent=dict(S.binding(B,N,fixture),status='COMPLETE',pid=2002,pgid=2002,process_token='b'*32,source_pre=admission,source_post=admission,deadline_epoch=now+100,children=[])
 for phase,i in [('save',0),('restore',1)]:
  child=fixture.actual/phase;child.mkdir();logs=fixture.actual/(phase+'-logs');logs.mkdir();(logs/'stdout.raw').write_text('SYNTHETIC_ONLY\n');(logs/'stderr.raw').write_text('SYNTHETIC_ONLY\n')
  rec=dict(gold,pid=2003+i,pgid=2002,process_token=('c' if i==0 else 'd')*32,phase=phase,parent_pid=2002,parent_process_token='b'*32,deadline_epoch=now+50+i*40,runtime=PROFILE['runtime'],device={'type':'xla','hardware':'TPU','pjrt':'TPU'},dispatch={k:True for k in SCHEMAS},precision_environment={k:None for k in P.ENV_KEYS},precision={'requested':'highest','readback':'highest','set_calls':1,'before_graph':True})
  for case in S.cases(B,N):
   name=case['id'];state=state_from_reference(case,reference[name]);raw=copy.deepcopy(reference[name]);raw.update(case_id=name,input_sha256=B.case_sha(case),pid=gold['pid'],process_token=gold['process_token'],state_dict=state,quant_state_public_roundtrip=True,packed_serialization_original=True,module_training=True,status='PASS')
   if phase=='save':
    B.write(fixture.oracle/'raw'/(name+'.json'),raw);torch.save(tensors(state),fixture.oracle/'raw'/(name+'.state.pt'))
   raw.update(pid=rec['pid'],process_token=rec['process_token'],placements={k:'xla:0' for k in raw['outputs']},counters={},execution_metrics={'ExecuteTime':[1,1.,[[1.,1.]]]})
   stem=child/'raw'/name;stem.parent.mkdir(exist_ok=True)
   if phase=='save':torch.save(tensors(state),stem.with_suffix('.state.pt'))
   else:
    source=fixture.actual/'save/raw'/(name+'.state.pt');stem.with_suffix('.state.pt').write_bytes(source.read_bytes());raw.update(loaded_state_before=state,loaded_checkpoint_sha256=B.sha(source),fresh_reconstruction=True)
   text=NT.hlo(case);raw['hlo_precision']=P.hlo_precision(text) if case['dtype']=='float32' and len(case['x_shape'])==2 else None;B.write(stem.with_suffix('.json'),raw);stem.with_suffix('.hlo.txt').write_text(text);stem.with_suffix('.metrics.txt').write_text('SYNTHETIC_ONLY\n')
  rec['artifacts']=B.inventory(child,'receipt.json');B.write(child/'receipt.json',rec)
  parent['children'].append({'phase':phase,'pid':rec['pid'],'launched_pid':rec['pid'],'pgid':2002,'process_token':rec['process_token'],'parent_pid':2002,'parent_process_token':'b'*32,'reaped':True,'child_absent':True,'cleanup_errors':[],'timeout':False,'exit_code':0,'receipt':phase+'/receipt.json','started_epoch':now+i*2,'finished_epoch':now+i*2+1,'started_monotonic':10+i*2,'finished_monotonic':11+i*2,'deadline_epoch':rec['deadline_epoch']})
 gold['artifacts']=B.inventory(fixture.oracle,'oracle-seal.json');B.write(fixture.oracle/'oracle-seal.json',gold);fixture.oracle_sha256=B.sha(fixture.oracle/'oracle-seal.json');B.write(fixture.actual/'receipt.json',parent);reseal(fixture);return fixture


def make_saved_fixture(base,admission='a'*64):
 import gzip,hashlib
 seed=HERE/'fixtures/nested_cpu_seed.json.gz'
 if hashlib.sha256(seed.read_bytes()).hexdigest()!='9b7096cbab45da88e046fddd43bb070bf0e42bbbb35120c0c84fd6aa82bcce6b':raise ValueError('SYNTHETIC_SEED_COMPRESSED_HASH')
 body=gzip.decompress(seed.read_bytes())
 if hashlib.sha256(body).hexdigest()!='7ae0b334645b4d8595c5445dd6c413a1dde89ccb3b113b3cd6861276a570ff86':raise ValueError('SYNTHETIC_SEED_RAW_HASH')
 return make_fixture(base,json.loads(body),admission=admission)

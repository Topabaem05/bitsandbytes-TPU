"""Unqualified CPU public-state fixtures; never writes a scientific oracle or acceptance."""
import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
p=argparse.ArgumentParser()
for key in ('source','repo','generation','oracle','phase','output','saved','token'):p.add_argument('--'+key,required=True)
a=p.parse_args();sys.path[:0]=[a.source,str(Path(a.generation)/'build-v3/source')]
import torch
import bitsandbytes as bnb
import state8_public as S
import exact_admission as A
import state8_contract as K

def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=Path(a.repo)/'experiments/2026-10-04-bitsandbytes-tpu';B=load(base/'probe_backend.py','cpu_fixture_backend');N=load(base/'probe_nested.py','cpu_fixture_nested');Old=load(base/'transfer_admission.py','cpu_fixture_old')
roots={'bitsandbytes':Path(a.generation)/'build-v3/source/bitsandbytes'};torch.set_num_threads(1)
S.state_methods(B,Old,N,bnb,roots)
out=Path(a.output);out.mkdir();rows=[]
for case in K.fixed_cases(B.read(Path(a.source)/'nested-inputs.json')):
 name=case['id'];path=out/(name+'.pt')
 if a.phase=='save':
  historical=B.read(Path(a.oracle)/'raw'/(name+'.json'));state=S.oracle_state(B,N,case,historical,bnb,torch)
  module=S.restore(B,N,case,state,bnb,torch,'cpu');saved={k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
  A.require(S.records(B,N,state)==S.records(B,N,saved),'CPU_SAVE_COMPLETE_STATE');torch.save(saved,path)
 else:
  source=Path(a.saved)/(name+'.pt');state,raw=S.checkpoint(B,N,torch,source,case)
  module=S.restore(B,N,case,state,bnb,torch,'cpu');saved={k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
  A.require(S.records(B,N,saved)==raw,'CPU_RESTORE_COMPLETE_STATE');path.write_bytes(source.read_bytes())
  A.require(B.sha(path)==B.sha(source),'CPU_CHECKPOINT_PAIR_BYTES')
  # Relevant incorrect checkpoint remains rejected by shape, packed metadata and tensor bytes.
  wrong={k:v.clone() for k,v in state.items()};wrong['weight.absmax']=wrong['weight.absmax'].to(torch.float32)
  wrongpath=out/(name+'.wrong.pt');torch.save(wrong,wrongpath)
  try:S.checkpoint(B,N,torch,wrongpath,case)
  except ValueError:pass
  else:raise ValueError('WRONG_STATE_DTYPE_ACCEPTED')
  wrongpath.unlink()
 S.validate_state(B,N,case,S.records(B,N,saved))
 A.require(type(module) is bnb.nn.Linear4bit and type(module.weight) is bnb.nn.Params4bit and module.weight.module is module and module.weight.compress_statistics,'CPU_ORIGINAL_DEFAULT_NESTED_CLASSES')
 rows.append({'case_id':name,'checkpoint_sha256':B.sha(path),'state':S.records(B,N,saved)})
B.write(out/'fixture.json',{'scope':'UNQUALIFIED_MAC_CPU_HISTORICAL_INPUT_FIXTURE_NOT_ORACLE','status':'PASS','phase':a.phase,
       'pid':os.getpid(),'pgid':os.getpgid(0),'parent_pid':os.getppid(),'process_token':a.token,'rows':rows})

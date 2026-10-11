"""Root checks for one sealed diagnostic. No provider calls or milestone claim."""
import argparse, copy, hashlib, json, shutil, sys, tempfile, time
from pathlib import Path
sys.dont_write_bytecode=True
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--packet-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
packet=a.candidate/'packet';a.output.mkdir(exist_ok=False)
sys.path.insert(0,str(packet/'control'))
import contract as C
tests=[]
def reject(label,fn):
    try:fn()
    except Exception as error:tests.append({'label':label,'status':'REJECTED','error_type':type(error).__name__,'message':str(error)});return
    raise AssertionError('INCORRECT_CASE_ACCEPTED:'+label)
m=C.packet_preflight(packet,a.candidate/'packet.zip',a.packet_sha256)
tests.append({'label':'complete_sealed_packet','status':'ACCEPTED','files':len(m['files'])+1})
reject('wrong_packet_sha',lambda:C.packet_preflight(packet,a.candidate/'packet.zip','0'*64))
with tempfile.TemporaryDirectory(prefix='bnb-tpu-root-word-packet-control-') as tmp:
    wrong=Path(tmp)/'packet';shutil.copytree(packet,wrong)
    source=wrong/'original/exact.py';source.write_bytes(source.read_bytes()+b'\n')
    reject('loose_source_differs_from_packet',lambda:C.packet_preflight(wrong,a.candidate/'packet.zip',a.packet_sha256))
assert C.budget(90,30,reserve=60,now=100)==30
tests.append({'label':'remaining_original_budget','status':'ACCEPTED'})
reject('future_birth_renews_budget',lambda:C.budget(101,30,reserve=60,now=100))
reject('closure_reserve_exhausted',lambda:C.budget(90,30,reserve=60,now=631))
sys.path.insert(0,str(packet/'source'))
import run_diagnostic as R
import torch
args=argparse.Namespace(mode='cpu',output=a.output/'cpu',deadline_epoch=time.time()+30,oracle=None,oracle_sha256=None,unqualified_cpu=True)
assert R.run(args)==0
report=json.loads((a.output/'cpu/report.json').read_text())
assert len(report['rows'])==24 and report['runtime']['status']=='UNQUALIFIED_LOCAL_CPU_CONTROL'
assert R.check_cpu(report['rows'])['status']=='PASS'
tests.append({'label':'local_cpu24','status':'ACCEPTED','qualification':'UNQUALIFIED_LOCAL_CPU_CONTROL'})
def changed_rows(label,outputs):
    rows=copy.deepcopy(report['rows']);row=next(r for r in rows if r['label']==label)
    for key,value in outputs.items():row['outputs'][key]=R.array(value)
    return rows
rows=changed_rows('bitcast-raw-vector',{'words':torch.zeros(12,dtype=torch.int32),'roundtrip':torch.zeros(12,dtype=torch.float32)})
reject('mutually_agreeing_wrong_bitcast',lambda:R.check_cpu(rows))
row=next(r for r in report['rows'] if r['label']=='scalar-compare-select')
outputs={key:torch.zeros(v['shape'],dtype=torch.bool if v['dtype']=='bool' else torch.int32) for key,v in row['outputs'].items()}
rows=changed_rows('scalar-compare-select',outputs)
reject('mutually_agreeing_wrong_scalar_select',lambda:R.check_cpu(rows))
for label in ['shift-left','shift-right']:
    rows=changed_rows(label,{'legacy':torch.zeros((8,4),dtype=torch.int32),'tensor':torch.zeros((8,4),dtype=torch.int32)})
    reject('mutually_agreeing_wrong_'+label,lambda rows=rows:R.check_cpu(rows))
rows=changed_rows('amax-255',{key:torch.tensor(0,dtype=torch.int32) for key in ['omitted','dim0','dim_negative1']})
reject('mutually_agreeing_wrong_amax',lambda:R.check_cpu(rows))
C.admit(packet)
result={'kind':'ROOT_M4_WORD_DIAGNOSTIC_PREPARATION_REVIEW','status':'PASS','packet_sha256':a.packet_sha256,'manifest_sha256':C.sha(packet/'manifest.json'),'driver_sha256':C.sha(packet/'control/driver.py'),'tests':tests,'provider_calls':0,'m4_status':'NOT_QUALIFIED','finished_epoch':time.time()}
(a.output/'review.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps({'status':result['status'],'tests':len(tests),'cpu_rows':24,'m4_status':'NOT_QUALIFIED'}))

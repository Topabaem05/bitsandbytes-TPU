"""Small isolated protocol fixtures. These do not represent remote science."""
import copy,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
CLOUD=Path(os.environ.get('PUBLIC15_TEST_CLOUD',str(Path(__file__).resolve().parents[1]/'cloud'))).resolve();HERE=CLOUD.parent;sys.path.insert(0,str(CLOUD));import public_contract as U;import remote as R

def receipt(m):
 return {'kind':'SYNTHETIC_TRANSPORT_FIXTURE','status':'CPU_ORACLE_READY_TPU_NOT_RUN','cpu_status':'PASS','tpu_status':'NOT_RUN','source_controls_status':'PASS_QUALIFIED_LINUX_SOURCE_CONTROLS','oracle_sha256':'a'*64,'science_deadline_epoch':time.time()+90,'allocation_epoch':time.time()-10,'packet_sha256':'b'*64,'public_cpu_inventory_sha256':'c'*64,'public_cpu_evidence_sha256':'d'*64,'steps':[],**{k:m[k]for k in U.FIELDS}}

WRITER='''import json,os,sys,time
from pathlib import Path
c=json.loads(sys.argv[1]);p=Path(c['base'])/'records/public/device/receipt.json';p.parent.mkdir(parents=True,exist_ok=True)
r={'scope':'SYNTHETIC_TRANSPORT_FIXTURE_NOT_DEVICE','kind':'PUBLIC_PALLAS_DEVICE_PROBE','status':'COMPLETE','scientific_variant':c['generation'],'admission_sha256':c['admission'],'source_pre':c['admission'],'source_post':c['admission'],'protocol_sha256':c['protocol'],'oracle_sha256':c['oracle'],'native_acceptance_sha256':c['native'],'pid':os.getpid(),'pgid':os.getpgid(0),'parent_pid':os.getppid(),'process_token':c['token'],'deadline_epoch':c['deadline']}
if c.get('fault')=='pid':r['pid']+=1
if c.get('fault')=='parent':r['parent_pid']+=1
if c.get('fault')=='token':r['process_token']='0'*32
if c.get('fault')=='source':r['source_post']='0'*64
if c.get('fault')=='deadline':r['deadline_epoch']+=1
p.write_text(json.dumps(r))
'''

def synthetic_direct_run(base,fault=None):
 def run(out,label,argv,deadline,limit,**kwargs):
  assert argv[1:3]==['-B',str(base/'payload/public/probe_public.py')]
  assert '--unqualified-local'not in argv and kwargs['tpu']is True
  values={argv[i][2:]:argv[i+1]for i in range(3,len(argv),2)}
  cfg={'base':str(base),'generation':U.GENERATION,'admission':U.ADMISSION_SHA,'protocol':U.PROTOCOL_SHA,'oracle':values['oracle-sha256'],'native':U.NATIVE_ACCEPTANCE_SHA,'token':values['process-token'],'deadline':float(values['deadline-epoch']),'fault':fault}
  return R.run_step(out,label,[sys.executable,'-B','-c',WRITER,json.dumps(cfg)],deadline,limit,cwd=base,tpu=False)
 return run

"""Explicit public15 source, CPU gate and recovery contract. No provider calls."""
import hashlib,importlib.util,json,math,os,re,stat,sys,zipfile
from pathlib import Path
MODE='m6-public-pallas-15'
GENERATION='public-pallas-device-mosaic7-gather-bf16-fp32-v1'
VARIANT='pallas-forward-mosaic7-gather-bf16-fp32-v1'
SCOPE='PUBLIC15_NON_NESTED_FP32_GRAD_BF16_FORWARD'
MANIFEST_SHA='131cdd25935b3216858b823c59732a30f3e5e43c25bbd8cc2eb9f284108e12bf'
ADMISSION_SHA='e467e9d711b61140bf5be91f7ecf1da518986683a2bb530b3341194dd28fa518'
PROTOCOL_SHA='dc22c7a3891786947d1127dc45e6d82fcd993f6b6a8d64ba804ee5d835a915c6'
NATIVE_ACCEPTANCE_SHA='c64fca47f57374fab691d9870005cf614fc6eebbc8a02706ed70781f9dd1042a'
BINDINGS={'public_manifest_sha256':'public/public-device-manifest.json','public_admission_sha256':'public/package-admission.json','public_protocol_sha256':'public/public-protocol.json','public_native_acceptance_sha256':'public-baseline/experiments/2026-10-04-bitsandbytes-tpu/results/native-bf16-acceptance.json'}
RESULT_MAX_BYTES=256*1024*1024
RESULT_MAX_MEMBERS=2000
FIELDS=(*BINDINGS,'public_generation','public_source_variant','public_scope','public_result_max_bytes','public_result_max_members')
PREFIXES=('public/','public-plugin/','public-baseline/')

def require(v,m):
 if not v:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pairs(rows):
 out={}
 for k,v in rows:require(k not in out,'PUBLIC_DUPLICATE_JSON');out[k]=v
 return out
def read(p):return json.loads(Path(p).read_text(),object_pairs_hook=pairs)
def write(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n')
def row(p):p=Path(p);return {'sha256':sha(p),'bytes':p.stat().st_size}
def verify_manifest(m):
 if m.get('experiment')!=MODE:
  require(not any(k in m for k in FIELDS) and not any(n.startswith(PREFIXES)for n in m['files']),'PUBLIC_NOT_REQUESTED');return
 require(m.get('public_generation')==GENERATION and m.get('public_source_variant')==VARIANT and m.get('public_scope')==SCOPE,'PUBLIC_EXACT_MODE_SCOPE')
 expected={'public_manifest_sha256':MANIFEST_SHA,'public_admission_sha256':ADMISSION_SHA,'public_protocol_sha256':PROTOCOL_SHA,'public_native_acceptance_sha256':NATIVE_ACCEPTANCE_SHA}
 require(all(m.get(k)==v and m['files'].get(BINDINGS[k],{}).get('sha256')==v for k,v in expected.items()),'PUBLIC_LITERAL_BINDINGS')
 require(m.get('public_result_max_bytes')==RESULT_MAX_BYTES and m.get('public_result_max_members')==RESULT_MAX_MEMBERS,'PUBLIC_EXACT_RESULT_LIMITS')
 require(m.get('precision')=='highest' and m.get('m6_status')=='NOT_QUALIFIED','PUBLIC_CLAIM_SCOPE')

def module(packet):
 root=Path(packet)/'public';sys.path.insert(0,str(root));import contract as C
 require(C.HERE==root.resolve() and C.ADMISSION_SHA==ADMISSION_SHA and C.PROTOCOL_SHA==PROTOCOL_SHA and C.VARIANT==GENERATION,'PUBLIC_MODULE_IDENTITY')
 return C

def payload(packet,m):
 packet=Path(packet);verify_manifest(m);require(m.get('experiment')==MODE,'PUBLIC_MODE_REQUIRED');C=module(packet);s=read(packet/'public/public-device-manifest.json');a=read(packet/'public/package-admission.json')
 require(sha(packet/'public/public-device-manifest.json')==MANIFEST_SHA and s['source_files']==C.scientific_sources(),'PUBLIC_SCIENTIFIC_SOURCE_MAP')
 expected={**s['source_files'],'public-device-manifest.json':row(packet/'public/public-device-manifest.json')}
 require({n.removeprefix('public/'):r for n,r in m['files'].items()if n.startswith('public/')}==expected,'PUBLIC_COMPLETE_SCIENTIFIC_MAP')
 require({n.removeprefix('public-plugin/'):r for n,r in m['files'].items()if n.startswith('public-plugin/')}==a['plugin_files'],'PUBLIC_COMPLETE_PLUGIN_MAP')
 for n,r in a['plugin_files'].items():require(row(packet/'public-plugin'/n)==r,'PUBLIC_PLUGIN_BYTES:'+n)
 baseline=packet/'public-baseline';C.native_sources(a,baseline)
 expected_baseline=dict(a['fixed_files']);prefix='experiments/2026-10-04-bitsandbytes-tpu/native/';nm=read(baseline/(prefix+'manifest.json'))
 for n,r in nm['sources'].items():expected_baseline[prefix+n]=r
 expected_baseline['patches/params4bit-xla-v1.json']=row(packet/'patches/params4bit-xla-v1.json')
 require(expected_baseline['patches/params4bit-xla-v1.json']['sha256']==a['patch_manifest_sha256'],'PUBLIC_PATCH_BINDING')
 require({n.removeprefix('public-baseline/'):r for n,r in m['files'].items()if n.startswith('public-baseline/')}==expected_baseline,'PUBLIC_FULL_BASELINE_MAP')
 for n,r in expected_baseline.items():require(row(baseline/n)==r,'PUBLIC_BASELINE_BYTES:'+n)
 C.accepted_native(baseline/BINDINGS['public_native_acceptance_sha256'].removeprefix('public-baseline/'),NATIVE_ACCEPTANCE_SHA,a)
 require(read(packet/'runtime/requirements.lock.json')==read(baseline/'experiments/2026-10-04-bitsandbytes-tpu/runtime/requirements.lock.json'),'PUBLIC_RUNTIME_LOCK_EQUALITY')
 return C,a

def source_args(base):
 base=Path(base);packet=base/'payload';C,a=payload(packet,read(packet/'manifest.json'))
 wheels={n:sorted((base/('built-'+n)).glob('*.whl'))for n in ('plugin','upstream')};require(all(len(v)==1 for v in wheels.values()),'PUBLIC_BUILT_WHEEL_COUNT')
 return ['--admission',str(packet/'public/package-admission.json'),'--protocol',str(packet/'public/public-protocol.json'),'--package-root',str(base/'venv/lib/python3.12/site-packages/bitsandbytes_tpu_pallas'),'--upstream-source',str(base/'venv/lib/python3.12/site-packages/bitsandbytes'),'--repo-root',str(packet/'public-baseline'),'--plugin-wheel',str(wheels['plugin'][0]),'--upstream-wheel',str(wheels['upstream'][0]),'--runtime-wheels',str(base/'wheels')]

def cpu_archive(out,base):
 out=Path(out);base=Path(base);paths={p.relative_to(out).as_posix():p for p in sorted((out/'cpu-oracle').rglob('*'))if p.is_file()}
 for n in ['source-controls.json','installed-source.json','built-source.json','built-plugin-source.json','steps/11-cpu-oracle/ownership.json','steps/11-cpu-oracle/result.json']:paths[n]=out/n
 for kind in ('plugin','upstream'):
  wheels=sorted((base/('built-'+kind)).glob('*.whl'));require(len(wheels)==1,'PUBLIC_CPU_WHEEL');paths['wheels/'+kind+'.whl']=wheels[0]
 inventory={n:row(p)for n,p in paths.items()};require(len(inventory)<=100 and sum(r['bytes']for r in inventory.values())<=64*1024*1024,'PUBLIC_CPU_ARCHIVE_BOUND')
 write(out/'public-cpu-inventory.json',inventory)
 with zipfile.ZipFile(out/'public-cpu-evidence.zip','w',zipfile.ZIP_DEFLATED)as z:
  for n,p in sorted(paths.items()):
   require(p.is_file()and not p.is_symlink(),'PUBLIC_CPU_REGULAR_FILE');i=zipfile.ZipInfo(n,date_time=(2026,10,9,0,0,0));i.create_system=3;i.external_attr=0o100644<<16;i.compress_type=zipfile.ZIP_DEFLATED;z.writestr(i,p.read_bytes())
 return {'public_cpu_inventory_sha256':sha(out/'public-cpu-inventory.json'),'public_cpu_evidence_sha256':sha(out/'public-cpu-evidence.zip'),'public_cpu_evidence_bytes':(out/'public-cpu-evidence.zip').stat().st_size}

def recover_cpu(directory,receipt):
 root=Path(directory);require(sha(root/'inventory.json')==receipt['public_cpu_inventory_sha256'] and sha(root/'evidence.zip')==receipt['public_cpu_evidence_sha256'] and (root/'evidence.zip').stat().st_size==receipt['public_cpu_evidence_bytes'],'PUBLIC_CPU_ARCHIVE_SEAL');inventory=read(root/'inventory.json')
 require(len(inventory)<=100 and sum(r['bytes']for r in inventory.values())<=64*1024*1024,'PUBLIC_CPU_RECOVERY_BOUND');target=root/'recovered';target.mkdir(exist_ok=False)
 with zipfile.ZipFile(root/'evidence.zip')as z:
  require(len(z.namelist())==len(inventory)and set(z.namelist())==set(inventory),'PUBLIC_CPU_EXACT_MEMBERS')
  for i in z.infolist():
   n=Path(i.filename);require(not n.is_absolute()and'..'not in n.parts and'\\'not in i.filename and i.create_system==3 and i.external_attr==0o100644<<16,'PUBLIC_CPU_MEMBER_TYPE');b=z.read(i);require({'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}==inventory[i.filename],'PUBLIC_CPU_MEMBER_BYTES');q=target/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b)
 return target

def gate_bindings(m,r):
 return {'kind':'ROOT_PUBLIC15_CPU_ORACLE_GATE','status':'QUALIFIED_LINUX_CPU15_ORACLE_VERIFIED','oracle_sha256':r['oracle_sha256'],'allocation_epoch':r['allocation_epoch'],'science_deadline_epoch':r['science_deadline_epoch'],**{k:m[k]for k in FIELDS},'source_admission_sha256':m['source_admission_sha256'],'packet_sha256':r['packet_sha256'],'public_cpu_inventory_sha256':r['public_cpu_inventory_sha256'],'public_cpu_evidence_sha256':r['public_cpu_evidence_sha256'],'m6':'NOT_QUALIFIED'}
def verify_root_gate(gate,receipt,m):require(gate==gate_bindings(m,receipt),'PUBLIC_ROOT_ORACLE_GATE')

def readback_missing(owner,receipt,out,gate):
 missing=[];out=Path(out)
 if not isinstance(gate,dict)or gate.get('status')!='QUALIFIED_LINUX_CPU15_ORACLE_VERIFIED':missing.append('root_cpu15_gate')
 if owner.get('recovered_oracle_sha256')is None:missing.append('cpu15_seal')
 if receipt.get('status')!='PUBLIC_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED' or receipt.get('tpu_status')!='PUBLIC_RECORDS_TERMINAL':missing.append('public_terminal')
 for n in ('public/device/receipt.json','steps/12-public/ownership.json'):
  if not(out/'recovered'/n).is_file():missing.append(n)
 return missing

class ReadbackUnavailable(Exception):pass

def host_args(packet,recovered,runtime_wheels):
 from types import SimpleNamespace
 from remote import unpack_source
 packet=Path(packet);recovered=Path(recovered);source=recovered.parent/'admitted-source'
 if not source.exists():unpack_source(packet/'patched-upstream.tar',source)
 return SimpleNamespace(admission=str(packet/'public/package-admission.json'),protocol=str(packet/'public/public-protocol.json'),package_root=str(packet/'public-plugin/src/bitsandbytes_tpu_pallas'),upstream_source=str(source/'bitsandbytes'),repo_root=str(packet/'public-baseline'),plugin_wheel=str(recovered/'wheels/plugin.whl'),upstream_wheel=str(recovered/'wheels/upstream.whl'),runtime_wheels=str(runtime_wheels),oracle=str(recovered/'cpu-oracle'),oracle_sha256=sha(recovered/'cpu-oracle/oracle-seal.json'),cpu_ownership=str(recovered/'steps/11-cpu-oracle/ownership.json'),cpu_ownership_sha256=sha(recovered/'steps/11-cpu-oracle/ownership.json'))

def cpu_gate(packet,recovered,receipt,runtime_wheels):
 packet=Path(packet);root=Path(recovered);m=read(packet/'manifest.json');C,a=payload(packet,m)
 require(receipt.get('runtime_status')=='PASS_TPU_RUNTIME_PROBE_ONLY' and receipt.get('cpu_status')=='PASS' and receipt.get('source_controls_status')=='PASS_QUALIFIED_LINUX_SOURCE_CONTROLS' and receipt.get('built_source_status')=='POST_PATCH_WHEEL_PYTHON_SOURCE_PASS' and receipt.get('installed_source_status')=='PUBLIC_POST_PATCH_INSTALLED_SOURCE_PASS' and receipt.get('built_plugin_source_status')=='PUBLIC_PLUGIN_WHEEL_SOURCE_PASS','PUBLIC_CPU_PHASE_PROOFS')
 require(runtime_wheels is not None and Path(runtime_wheels).is_absolute(),'PUBLIC_EXPLICIT_HOST_RUNTIME_WHEELS')
 args=host_args(packet,root,runtime_wheels);O=C.load(packet/'public/verify_public_oracle.py','public_cpu_replay');result,rows=O.verify(args,qualified=True);seal=C.read(root/'cpu-oracle/oracle-seal.json');ownership=C.read(args.cpu_ownership);step=C.read(root/'steps/11-cpu-oracle/result.json')
 require(ownership['pid']==ownership['pgid']==seal['pid']==seal['pgid'] and ownership['owner_pid']==seal['parent_pid'],'PUBLIC_CPU_ACTUAL_PARENT')
 require(step.get('status')=='PASS'and step.get('exit_code')==0 and step.get('cleanup',{}).get('errors')==[],'PUBLIC_CPU_STEP_PASS')
 argv=ownership['argv'];require('--unqualified-local'not in argv and argv[1:3]==['-B','/content/bnb-tpu-first/payload/public/prepare_public_oracle.py'],'PUBLIC_CPU_EXACT_ENTRY')
 from remote import validate_source_controls
 validate_source_controls(read(root/'source-controls.json'))
 proof=read(root/'installed-source.json');built=read(root/'built-source.json');plugin=read(root/'built-plugin-source.json')
 require(proof.get('status')=='PUBLIC_POST_PATCH_INSTALLED_SOURCE_PASS' and proof.get('public_admission_sha256')==ADMISSION_SHA and proof.get('installed_python_files')=={'bitsandbytes':a['upstream_python_files'],'bitsandbytes_tpu_pallas':a['plugin_python_files']},'PUBLIC_INSTALLED_SOURCE_PROOF')
 require(built.get('status')=='POST_PATCH_WHEEL_PYTHON_SOURCE_PASS' and built.get('python_files')==a['upstream_python_files'] and built.get('patch_manifest_sha256')==m['patch_manifest_sha256'],'PUBLIC_BUILT_UPSTREAM_SOURCE_PROOF')
 require(plugin.get('status')=='PUBLIC_PLUGIN_WHEEL_SOURCE_PASS' and plugin.get('plugin_python_files')==a['plugin_python_files'] and plugin.get('public_admission_sha256')==ADMISSION_SHA and plugin.get('wheel_sha256')==sha(root/'wheels/plugin.whl'),'PUBLIC_BUILT_PLUGIN_SOURCE_PROOF')
 require(seal['wheel_records']['plugin']['sha256']==sha(root/'wheels/plugin.whl')and seal['wheel_records']['upstream']['sha256']==sha(root/'wheels/upstream.whl'),'PUBLIC_REMOTE_WHEEL_BODY_BINDING')
 require(receipt['oracle_sha256']==args.oracle_sha256,'PUBLIC_CPU_RECEIPT_SEAL')
 require(all(type(receipt.get(k))in(int,float)and math.isfinite(receipt[k])for k in ('allocation_epoch','science_deadline_epoch'))and seal['deadline_epoch']<=receipt['science_deadline_epoch']<=receipt['allocation_epoch']+3600-660,'PUBLIC_ORIGINAL_CPU_DEADLINE')
 require(proof.get('no_package_client_import')is True,'PUBLIC_INSTALLED_SOURCE_NO_IMPORT')
 require(receipt['allocation_epoch']<=seal['started_epoch']<=seal['finished_epoch']<=receipt['science_deadline_epoch'],'PUBLIC_FRESH_CPU_ALLOCATION_TIME')
 remote='/content/bnb-tpu-first';expected_args={'admission':remote+'/payload/public/package-admission.json','protocol':remote+'/payload/public/public-protocol.json','package-root':remote+'/venv/lib/python3.12/site-packages/bitsandbytes_tpu_pallas','upstream-source':remote+'/venv/lib/python3.12/site-packages/bitsandbytes','repo-root':remote+'/payload/public-baseline','plugin-wheel':remote+'/built-plugin/'+plugin['wheel_name'],'upstream-wheel':remote+'/built-upstream/'+built['wheel_name'],'runtime-wheels':remote+'/wheels','output':remote+'/records/cpu-oracle','process-token':seal['process_token'],'deadline-epoch':str(seal['deadline_epoch'])}
 require(argv[0]==remote+'/venv/bin/python'and len(argv)==3+2*len(expected_args)and all(argv.count('--'+k)==1 and argv[argv.index('--'+k)+1]==v for k,v in expected_args.items()),'PUBLIC_CPU_EXACT_SOURCE_ARGV')
 require(any(r.get('sha256')==sha(root/'wheels/plugin.whl')and r.get('name')==plugin.get('wheel_name')for r in receipt.get('built_wheels',[]))and any(r.get('sha256')==sha(root/'wheels/upstream.whl')and r.get('name')==built.get('wheel_name')for r in receipt.get('built_wheels',[])),'PUBLIC_RECEIPT_BUILT_WHEEL_BINDING')
 return gate_bindings(m,receipt),{'scope':'QUALIFIED_RECORD_REPLAY_ON_HOST_CPU','case_count':len(rows),'record_validation':result['record_validation'],'qualified_record_runtime':seal['runtime'],'source_generation':GENERATION}

def verifier_cli(packet,recovered_cpu,out,receipt,runtime_wheels,audit_python,deadline):
 packet=Path(packet);out=Path(out);args=host_args(packet,recovered_cpu,runtime_wheels);C,a=payload(packet,read(packet/'manifest.json'))
 argv=[sys.executable,'-B',str(packet/'public/verify_public.py')]
 for k,v in vars(args).items():argv+=['--'+k.replace('_','-'),str(v)]
 values={'actual':out/'recovered/public/device','receipt-sha256':receipt['public_receipt_sha256'],'ownership':out/'recovered/steps/12-public/ownership.json','ownership-sha256':receipt['public_outer_owner_sha256'],'native-acceptance':packet/'public-baseline/experiments/2026-10-04-bitsandbytes-tpu/results/native-bf16-acceptance.json','native-acceptance-sha256':NATIVE_ACCEPTANCE_SHA,'mosaic-audit-python':audit_python,'audit-deadline-epoch':deadline}
 for k,v in values.items():argv+=['--'+k,str(v)]
 return argv

def validate_report(r,packet):
 ids=[v['id']for v in read(Path(packet)/'public/public-protocol.json')['cases']];require(isinstance(r.get('rows'),list)and [v.get('case')for v in r['rows']]==ids,'PUBLIC_REPORT_CASE_MATRIX')
 require(r.get('record_validation')=='PASS'and r.get('scope')=='ACTUAL_PUBLIC_API'and r.get('case_count')==15 and r.get('native_case_count')==11 and r.get('reference_case_count')==4,'PUBLIC_SCIENTIFIC_REPORT')
 require(r.get('numerical_status')in('PASS','FAIL','ERROR') and r.get('m4_status')==r.get('m6_memory_status')=='NOT_QUALIFIED' and r.get('performance')=='NOT_MEASURED','PUBLIC_OUTCOME_CLAIM_SCOPE')
 statuses=[v.get('status')for v in r['rows']];require(all(v in('PASS','FAIL','ERROR','NOT_RUN')for v in statuses),'PUBLIC_REPORT_CASE_STATUS')
 first_error=None
 for index,status in enumerate(statuses):
  if first_error is not None:require(status=='NOT_RUN','PUBLIC_REPORT_AFTER_FIRST_ERROR')
  elif status=='ERROR':first_error=index
  else:require(status!='NOT_RUN','PUBLIC_REPORT_NOT_RUN_WITHOUT_ERROR')
 derived='ERROR'if first_error is not None else'FAIL'if any(v=='FAIL'for v in statuses)else'PASS';require(r['numerical_status']==derived,'PUBLIC_REPORT_NUMERIC_DERIVATION')
 return r['numerical_status']

def result_limits(inventory,infos=None):
 require(isinstance(inventory,dict)and len(inventory)<=RESULT_MAX_MEMBERS,'PUBLIC_RESULT_MEMBER_BOUND')
 require(all(isinstance(r,dict)and set(r)=={'sha256','bytes'}and type(r['bytes'])is int and r['bytes']>=0 and re.fullmatch('[0-9a-f]{64}',r['sha256'])is not None for r in inventory.values()),'PUBLIC_RESULT_INVENTORY_SCHEMA')
 require(sum(r['bytes']for r in inventory.values())<=RESULT_MAX_BYTES,'PUBLIC_RESULT_TOTAL_UNCOMPRESSED_BOUND')
 if infos is not None:require(len(infos)<=RESULT_MAX_MEMBERS and sum(i.file_size for i in infos)<=RESULT_MAX_BYTES,'PUBLIC_RESULT_ZIP_UNCOMPRESSED_BOUND')

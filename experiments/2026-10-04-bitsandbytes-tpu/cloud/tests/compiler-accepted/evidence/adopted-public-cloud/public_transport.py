"""Public transport coordinator functions; run_step owns the probe directly."""
import argparse,importlib.util,json,secrets,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('admitted_public_contract',HERE/'public_contract.py');U=importlib.util.module_from_spec(s);s.loader.exec_module(U)

def installed_source(base):
 base=Path(base);packet=base/'payload';m=U.read(packet/'manifest.json');C,a=U.payload(packet,m);values=U.source_args(base);v={values[i][2:].replace('-','_'):values[i+1]for i in range(0,len(values),2)}
 C.admission(v['admission'],v['protocol'],v['package_root'],v['upstream_source'],v['repo_root']);C.wheels(a,v['plugin_wheel'],v['upstream_wheel'])
 proof={'status':'PUBLIC_POST_PATCH_INSTALLED_SOURCE_PASS','public_admission_sha256':U.ADMISSION_SHA,'patch_manifest_sha256':m['patch_manifest_sha256'],'installed_python_files':{'bitsandbytes':a['upstream_python_files'],'bitsandbytes_tpu_pallas':a['plugin_python_files']},'source_roots':{k:v[k]for k in ('package_root','upstream_source')},'native_acceptance':'REQUIRED_BOUNDED_NATIVE_RECORDS_DOCUMENT_CHECKED_PUBLIC_NOT_RUN','public_device_outcome':'NOT_RUN','no_package_client_import':not any(n in sys.modules for n in ('bitsandbytes','bitsandbytes_tpu_pallas','jax','torch_xla'))}
 U.require(proof['no_package_client_import'],'PUBLIC_INSTALLED_NO_IMPORT');U.write(base/'records/installed-source.json',proof)
 return proof

def built_plugin(base):
 base=Path(base);packet=base/'payload';C,a=U.payload(packet,U.read(packet/'manifest.json'));w=sorted((base/'built-plugin').glob('*.whl'));U.require(len(w)==1,'PUBLIC_PLUGIN_WHEEL_COUNT');r=C.wheel_source(w[0],a['import_name'],{**a['plugin_python_files'],'source_manifest.json':a['plugin_manifest_sha256']},'[bitsandbytes.backends]\ntpu_pallas_forward = bitsandbytes_tpu_pallas:register')
 U.write(base/'records/built-plugin-source.json',{'status':'PUBLIC_PLUGIN_WHEEL_SOURCE_PASS','public_admission_sha256':U.ADMISSION_SHA,'plugin_python_files':a['plugin_python_files'],'wheel_sha256':r['sha256'],'wheel_name':w[0].name})

def cpu_argv(base,deadline):
 base=Path(base);return [str(base/'venv/bin/python'),'-B',str(base/'payload/public/prepare_public_oracle.py'),*U.source_args(base),'--output',str(base/'records/cpu-oracle'),'--process-token',secrets.token_hex(16),'--deadline-epoch',str(deadline)]

def execute_public(base,receipt,m,run_step,deadline):
 base=Path(base);U.payload(base/'payload',m);U.require(receipt.get('cpu_status')=='PASS'and receipt.get('tpu_status')=='NOT_RUN'and receipt.get('source_controls_status')=='PASS_QUALIFIED_LINUX_SOURCE_CONTROLS','PUBLIC_DEVICE_PHASE_ORDER');gate=U.read(base/'launch.json');U.verify_root_gate(gate,receipt,m)
 U.require(time.time()+30<deadline<=receipt['science_deadline_epoch'],'PUBLIC_DEVICE_ORIGINAL_DEADLINE');token=secrets.token_hex(16)
 cpu_owner=base/'records/steps/11-cpu-oracle/ownership.json';acceptance=base/'payload/public-baseline/experiments/2026-10-04-bitsandbytes-tpu/results/native-bf16-acceptance.json'
 argv=[str(base/'venv/bin/python'),'-B',str(base/'payload/public/probe_public.py'),*U.source_args(base),'--oracle',str(base/'records/cpu-oracle'),'--oracle-sha256',gate['oracle_sha256'],'--cpu-ownership',str(cpu_owner),'--cpu-ownership-sha256',U.sha(cpu_owner),'--native-acceptance',str(acceptance),'--native-acceptance-sha256',U.NATIVE_ACCEPTANCE_SHA,'--output',str(base/'records/public/device'),'--process-token',token,'--deadline-epoch',str(deadline)]
 receipt.update(status='PUBLIC_CHILD_RUNNING',tpu_status='RUNNING',tpu_attempted=True,public_root_gate_sha256=U.sha(base/'launch.json'));U.write(base/'records/receipt.json',receipt)
 rec=run_step(base/'records','12-public',argv,deadline,1500,cwd=base/'payload',tpu=True);receipt['steps'].append({'label':'12-public',**rec})
 U.require(not rec['cleanup']['errors']and not rec.get('error')and(rec['status'],rec['exit_code'])in{('PASS',0),('CHILD_FAILED',2)},'PUBLIC_CHILD_TERMINAL_OR_CLEANUP')
 path=base/'records/public/device/receipt.json';r=U.read(path);own=U.read(base/'records/steps/12-public/ownership.json')
 U.require(r.get('kind')=='PUBLIC_PALLAS_DEVICE_PROBE'and r.get('status')=='COMPLETE'and r.get('scientific_variant')==U.GENERATION and r.get('admission_sha256')==r.get('source_pre')==r.get('source_post')==U.ADMISSION_SHA and r.get('protocol_sha256')==U.PROTOCOL_SHA and r.get('oracle_sha256')==gate['oracle_sha256']and r.get('native_acceptance_sha256')==U.NATIVE_ACCEPTANCE_SHA,'PUBLIC_RECEIPT_SOURCE_ORACLE_BINDING')
 U.require(r.get('pid')==r.get('pgid')==own['pid']==own['pgid']and r.get('parent_pid')==own['owner_pid']and r.get('process_token')==token and r.get('deadline_epoch')==deadline,'PUBLIC_DIRECT_OWNER_IDENTITY')
 receipt.update(status='PUBLIC_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED',tpu_status='PUBLIC_RECORDS_TERMINAL',public_receipt_sha256=U.sha(path),public_outer_owner_sha256=U.sha(base/'records/steps/12-public/ownership.json'))
 return rec

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['installed-source']);p.add_argument('--base',type=Path,required=True);a=p.parse_args();print(json.dumps(installed_source(a.base),sort_keys=True))

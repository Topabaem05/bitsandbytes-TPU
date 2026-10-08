"""Generated CPU arithmetic and synthetic qualified-shaped records. Never device evidence.

The original local CPU seal is retained. Its qualification fields are changed only
in a separate fixture record to exercise the strict recovered-record consumer.
No generated fixture is a root acceptance or a dispatch dependency.
"""
import copy,json,os,secrets,shutil,sys,time,zipfile
from pathlib import Path
from types import SimpleNamespace
from control_helpers import U,R
U.module(Path(os.environ['PUBLIC15_TEST_PACKET']).resolve())
import record_fixture as F
import transport


def source_controls():
 rows=[]
 for method in ('direct_to_fixture','module_to_fixture','direct_quantize_meta','module_apply_meta'):
  rows.append(dict(method=method,scope='CPU_META_TYPE_MECHANICS_ONLY',bias_device='meta'if method=='module_apply_meta'else'cpu',**dict.fromkeys(('identity','class','attrs','module_alias'),True)))
 for negative in ('python_weakref','cpp_weakref','held_impl','requires_grad','subclass'):
  rows.append(dict(negative=negative,error='synthetic rejection shape',**dict.fromkeys(('rejected','tensorimpl_unchanged','dict_identity_unchanged','data_unchanged','module_unchanged'),True)))
 rows += [dict(failure='ORIGINAL_INCOMPATIBLE_TYPE',original_unchanged=True)for _ in range(3)]
 rows.append(dict(keys=['weight','bias'],**dict.fromkeys(('compatible_cpu','weight_identity','custom_attrs','state_format','weights_only_load'),True)))
 for case in ('existing_gradient','overwrite_flag','swap_flag'):
  rows.append(dict(case=case,scope='CPU_META_TYPE_MECHANICS_ONLY',**dict.fromkeys(('rejected_before_swap','parameter_identity','tensorimpl_identity','dict_identity','class_identity','values_unchanged','quant_state_identity','module_alias_unchanged','gradient_identity','bias_identity'),True)))
 rows.append(dict(case='genuine_cpu_forward_backward',forward_values=[[0.,0.]]*3,**dict.fromkeys(('forward_exact','dx_exact','db_exact','frozen_base','state_plain_tensors'),True)))
 rows.append(dict(case='fresh_process_cpu_public_restore',**dict.fromkeys(('forward_exact','weights_only','original_classes','module_state_alias','frozen_base'),True)))
 return dict(record_validation='PASS',qualified_source_controls=True,runtime={'torch':'2.9.0+cpu','python':'3.12.14','platform':'Linux'},patch_manifest_sha256=R.TRANSFER_PATCH_MANIFEST_SHA,tpu='NOT_RUN',xla='NOT_RUN',failures=[],whole_module_transactionality='NOT_PROMISED',controls=rows)


def source_wheel(path,root,namespace,mapping,entry=None):
 with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED)as z:
  for n in sorted(mapping):
   i=zipfile.ZipInfo(namespace+'/'+n,date_time=(2026,10,9,0,0,0));i.create_system=3;i.external_attr=0o100644<<16;i.compress_type=zipfile.ZIP_DEFLATED;z.writestr(i,(root/n).read_bytes())
  if entry:z.writestr(namespace+'-0.0.0.dist-info/entry_points.txt',entry+'\n')


class Fixture:
 def __init__(self,packet,base,runtime_wheels,audit_python):
  self.packet=Path(packet);self.base=Path(base);self.out=self.base/'records';self.out.mkdir(parents=True);self.m=U.read(self.packet/'manifest.json');self.C,self.a=U.payload(self.packet,self.m);C=self.C
  R.unpack_source(self.packet/'patched-upstream.tar',self.base/'upstream');up=self.base/'upstream/bitsandbytes';plugin=self.packet/'public-plugin/src/bitsandbytes_tpu_pallas'
  for kind,ns,root,mapping,entry in [('plugin',self.a['import_name'],plugin,{**self.a['plugin_python_files'],'source_manifest.json':self.a['plugin_manifest_sha256']},'[bitsandbytes.backends]\ntpu_pallas_forward = bitsandbytes_tpu_pallas:register'),('upstream','bitsandbytes',up,self.a['upstream_python_files'],None)]:
   d=self.base/('built-'+kind);d.mkdir();source_wheel(d/(kind+'-synthetic-source-only.whl'),root,ns,mapping,entry)
  self.args=SimpleNamespace(admission=str(self.packet/'public/package-admission.json'),protocol=str(self.packet/'public/public-protocol.json'),package_root=str(plugin),upstream_source=str(up),repo_root=str(self.packet/'public-baseline'),plugin_wheel=str(next((self.base/'built-plugin').glob('*.whl'))),upstream_wheel=str(next((self.base/'built-upstream').glob('*.whl'))),runtime_wheels=str(runtime_wheels))
  deadline=time.time()+180;token=secrets.token_hex(16);argv=[sys.executable,'-B',str(self.packet/'public/prepare_public_oracle.py')]
  for k,v in vars(self.args).items():argv+=['--'+k.replace('_','-'),v]
  argv+=['--output',str(self.out/'cpu-oracle'),'--process-token',token,'--deadline-epoch',str(deadline),'--unqualified-local']
  result=R.run_step(self.out,'11-cpu-oracle',argv,deadline,170,cwd=self.packet,tpu=False)
  assert result['status']=='PASS'and not result['cleanup']['errors'],result
  self.args.oracle=str(self.out/'cpu-oracle');self.args.oracle_sha256=C.sha(self.out/'cpu-oracle/oracle-seal.json');self.args.cpu_ownership=str(self.out/'steps/11-cpu-oracle/ownership.json');self.args.cpu_ownership_sha256=C.sha(self.args.cpu_ownership)
  self.original_seal=C.read(self.out/'cpu-oracle/oracle-seal.json');C.write(self.base/'genuine-unqualified-seal.json',self.original_seal)
  C.write(self.base/'fixture-provenance.json',dict(scope='SYNTHETIC_QUALIFIED_RECORD_SHAPE_ONLY',actual_linux_oracle='NOT_RUN',actual_device='NOT_RUN',provider_calls=0,source_only_wheels='NOT_INSTALLED',genuine_cpu_seal_sha256=C.sha(self.base/'genuine-unqualified-seal.json')))
  lock=C.read(Path(self.args.repo_root)/'experiments/2026-10-04-bitsandbytes-tpu/runtime/requirements.lock.json');self.metadata={w['name']:{'version':w['version'],'metadata_sha256':w['metadata_sha256']}for w in lock['wheels']}
  U.write(self.out/'source-controls.json',source_controls());R.validate_source_controls(source_controls())
  U.write(self.out/'installed-source.json',dict(status='PUBLIC_POST_PATCH_INSTALLED_SOURCE_PASS',public_admission_sha256=U.ADMISSION_SHA,installed_python_files={'bitsandbytes':self.a['upstream_python_files'],'bitsandbytes_tpu_pallas':self.a['plugin_python_files']},no_package_client_import=True))
  U.write(self.out/'built-source.json',dict(status='POST_PATCH_WHEEL_PYTHON_SOURCE_PASS',python_files=self.a['upstream_python_files'],patch_manifest_sha256=self.m['patch_manifest_sha256'],wheel_name=Path(self.args.upstream_wheel).name,wheel_sha256=C.sha(self.args.upstream_wheel)))
  U.write(self.out/'built-plugin-source.json',dict(status='PUBLIC_PLUGIN_WHEEL_SOURCE_PASS',public_admission_sha256=U.ADMISSION_SHA,plugin_python_files=self.a['plugin_python_files'],wheel_name=Path(self.args.plugin_wheel).name,wheel_sha256=C.sha(self.args.plugin_wheel)))
  for k in ('plugin','upstream'):
   d=self.out/'wheels';d.mkdir(exist_ok=True);shutil.copyfile(getattr(self.args,k+'_wheel'),d/(k+'.whl'))
  self.args.native_acceptance=None;self.args.native_acceptance_sha256=None;self.args.mosaic_audit_python=str(audit_python);self.args.audit_deadline_epoch=time.time()+120
  self.actual=F.make_fixture(self.out/'public/device',self.args);(self.out/'steps/12-public').mkdir();shutil.move(self.out/'public/outer.json',self.out/'steps/12-public/ownership.json')
  self.bind_epoch(time.time()-2)
 def bind_epoch(self,epoch):
  C=self.C;deadline=epoch+1800;self.epoch=epoch;self.deadline=deadline
  seal=copy.deepcopy(self.original_seal);seal.update(qualified_runtime=True,scope='QUALIFIED_LINUX_CPU',runtime={'platform':'Linux','machine':'x86_64','python':'3.12.14','torch':'2.9.0+cpu'},installed_runtime_metadata=self.metadata,started_epoch=epoch,finished_epoch=time.time(),deadline_epoch=deadline)
  C.write(self.out/'cpu-oracle/oracle-seal.json',seal);self.args.oracle_sha256=C.sha(self.out/'cpu-oracle/oracle-seal.json')
  owner=C.read(self.args.cpu_ownership);argv=owner['argv'];argv=argv[:];
  if '--unqualified-local'in argv:argv.remove('--unqualified-local')
  remote='/content/bnb-tpu-first';argv[0]=remote+'/venv/bin/python';argv[2]=remote+'/payload/public/prepare_public_oracle.py'
  fields={'admission':remote+'/payload/public/package-admission.json','protocol':remote+'/payload/public/public-protocol.json','package-root':remote+'/venv/lib/python3.12/site-packages/bitsandbytes_tpu_pallas','upstream-source':remote+'/venv/lib/python3.12/site-packages/bitsandbytes','repo-root':remote+'/payload/public-baseline','plugin-wheel':remote+'/built-plugin/'+Path(self.args.plugin_wheel).name,'upstream-wheel':remote+'/built-upstream/'+Path(self.args.upstream_wheel).name,'runtime-wheels':remote+'/wheels','output':remote+'/records/cpu-oracle','process-token':seal['process_token'],'deadline-epoch':str(deadline)}
  for k,v in fields.items():argv[argv.index('--'+k)+1]=v
  owner['argv']=argv;C.write(self.args.cpu_ownership,owner);self.args.cpu_ownership_sha256=C.sha(self.args.cpu_ownership)
  device=C.read(self.actual/'receipt.json');device.update(scope='ACTUAL_PUBLIC_API',native_acceptance_sha256=U.NATIVE_ACCEPTANCE_SHA,runtime_wheel_records=C.runtime_wheels(self.a,self.args.repo_root,self.args.runtime_wheels,qualified=True),installed_runtime_metadata=self.metadata,oracle_sha256=self.args.oracle_sha256,started_epoch=epoch,finished_epoch=time.time(),deadline_epoch=deadline);C.write(self.actual/'receipt.json',device)
  outer=C.read(self.out/'steps/12-public/ownership.json');outer['argv'][-1]=str(deadline);C.write(self.out/'steps/12-public/ownership.json',outer);F.reseal(self.actual)
 def receipt(self):
  return dict(status='CPU_ORACLE_READY_TPU_NOT_RUN',installation_status='PASS',runtime_status='PASS_TPU_RUNTIME_PROBE_ONLY',cpu_status='PASS',tpu_status='NOT_RUN',source_controls_status='PASS_QUALIFIED_LINUX_SOURCE_CONTROLS',built_source_status='POST_PATCH_WHEEL_PYTHON_SOURCE_PASS',installed_source_status='PUBLIC_POST_PATCH_INSTALLED_SOURCE_PASS',built_plugin_source_status='PUBLIC_PLUGIN_WHEEL_SOURCE_PASS',oracle_sha256=self.args.oracle_sha256,allocation_epoch=self.epoch,science_deadline_epoch=self.deadline,packet_sha256=U.sha(self.packet/'payload.zip'),manifest_sha256=U.sha(self.packet/'manifest.json'),runtime_lock_sha256=self.m['runtime_lock_sha256'],source_admission_sha256=self.m['source_admission_sha256'],experiment=U.MODE,precision='highest',steps=[],error=None,built_wheels=[dict(name=Path(getattr(self.args,k+'_wheel')).name,**U.row(getattr(self.args,k+'_wheel')))for k in ('plugin','upstream')],**{k:self.m[k]for k in U.FIELDS},**{k:self.m[k]for k in R.TRANSFER_BINDINGS})

"""Explicit no-provider API double for early CPU failure/recovery/closure only."""
import json,zipfile
from pathlib import Path
from control_helpers import U
import transport
class BlockedCPU:
 def __init__(self,packet,root):
  self.packet=Path(packet);self.root=Path(root);self.calls=[];self.m=U.read(self.packet/'manifest.json');self.session=None;self.remote={};self.receipt={'status':'INSTALLED_NOT_QUALIFIED','installation_status':'PASS','runtime_status':'NOT_RUN','cpu_status':'NOT_RUN','tpu_status':'NOT_RUN','packet_sha256':U.sha(self.packet/'payload.zip'),'manifest_sha256':U.sha(self.packet/'manifest.json'),'runtime_lock_sha256':self.m['runtime_lock_sha256'],'source_admission_sha256':self.m['source_admission_sha256'],'allocation_epoch':None,'steps':[],'error':None,'experiment':U.MODE,'precision':'highest',**{k:self.m[k]for k in U.FIELDS}}
  from remote import TRANSFER_BINDINGS
  self.receipt.update({k:self.m[k]for k in TRANSFER_BINDINGS})
 def export(self):
  root=self.root/'fake-export';root.mkdir();U.write(root/'receipt.json',self.receipt)
  inventory={'receipt.json':U.row(root/'receipt.json')};U.write(root/'archive-members.json',inventory)
  with zipfile.ZipFile(root/'evidence.zip','w',zipfile.ZIP_DEFLATED)as z:
   i=zipfile.ZipInfo('receipt.json');i.create_system=3;i.external_attr=0o100644<<16;z.writestr(i,(root/'receipt.json').read_bytes())
  self.receipt.update(evidence_sha256=U.sha(root/'evidence.zip'),evidence_bytes=(root/'evidence.zip').stat().st_size,archive_members_sha256=U.sha(root/'archive-members.json'))
  self.remote['records/archive-members.json']=root/'archive-members.json';self.remote['records/evidence.zip']=root/'evidence.zip'
  transport.split(root/'evidence.zip',root/'parts',expected_sha256=self.receipt['evidence_sha256'],expected_bytes=self.receipt['evidence_bytes'])
  for p in (root/'parts').iterdir():self.remote['result-parts/'+p.name]=p
 def __call__(self,label,argv,timeout):
  self.calls.append(label);text='';record={'status':'PASS','scope':'SIMULATED_NO_CLOUD','argv':list(argv),'timeout_seconds':timeout}
  if label=='03-one-allocation':self.session=argv[argv.index('-s')+1]
  if label=='04-sessions-after':text='['+self.session+'] fixture | Hardware: V6E1'
  if label in('01-sessions-before','92-after-stop'):text='No active sessions found on server.'
  if label.endswith('usage-before')or label=='93-usage-after':text='Active assignments: 0\nUsage rate: 0.00/hr'
  if label=='09-install-receipt':
   # The fake owner already has the original epoch; do not invent a replacement.
   self.receipt['allocation_epoch']=U.read(self.root/'host/owner.json')['allocation_epoch']
  if label=='10-cpu':self.receipt.update(status='BLOCKED',error={'type':'ValueError','message':'SYNTHETIC_CPU_SOURCE_FAILURE'})
  if label=='21-export':self.export()
  if argv[0]=='download':
   remote=argv[-2].removeprefix('content/bnb-tpu-first/');target=Path(argv[-1]);target.parent.mkdir(parents=True,exist_ok=True)
   if remote=='records/receipt.json':U.write(target,self.receipt)
   else:target.write_bytes(self.remote[remote].read_bytes())
  return record,text

class CompletePublic(BlockedCPU):
 """Full owner routing against synthetic records; never actual CPU/TPU qualification."""
 def __init__(self,packet,root,fixture,*,fault=None):
  super().__init__(packet,root);self.fixture=fixture;self.fault=fault
 def export(self):
  # Use the production uncompressed-size admission and regular-file archive.
  from control_helpers import R
  source=self.fixture.out;U.write(source/'receipt.json',self.receipt)
  self.receipt.update(R.package(source))
  self.remote['records/archive-members.json']=source/'archive-members.json';self.remote['records/evidence.zip']=source/'evidence.zip'
  parts=self.root/'result-parts';transport.split(source/'evidence.zip',parts,expected_sha256=self.receipt['evidence_sha256'],expected_bytes=self.receipt['evidence_bytes'])
  for p in parts.iterdir():self.remote['result-parts/'+p.name]=p
 def __call__(self,label,argv,timeout):
  if label=='09-install-receipt':
   self.fixture.bind_epoch(U.read(self.root/'host/owner.json')['allocation_epoch']);self.receipt=self.fixture.receipt();self.receipt.update(status='INSTALLED_NOT_QUALIFIED',cpu_status='NOT_RUN')
  if label=='10-cpu':
   self.receipt=self.fixture.receipt();self.receipt.update(U.cpu_archive(self.fixture.out,self.fixture.base));parts=self.root/'cpu-parts';transport.split(self.fixture.out/'public-cpu-evidence.zip',parts,expected_sha256=self.receipt['public_cpu_evidence_sha256'],expected_bytes=self.receipt['public_cpu_evidence_bytes']);self.remote.update({'records/cpu-oracle/oracle-seal.json':self.fixture.out/'cpu-oracle/oracle-seal.json','records/public-cpu-inventory.json':self.fixture.out/'public-cpu-inventory.json'})
   for p in parts.iterdir():self.remote['public-cpu-parts/'+p.name]=p
   if self.fault=='oracle':
    # Agreeing transport hashes cannot authorize an independently wrong CPU output.
    self.receipt['oracle_sha256']='0'*64
   self.calls.append(label);return {'status':'PASS','scope':'SIMULATED_NO_CLOUD','argv':list(argv)},''
  if label=='13-tpu':
   self.receipt.update(status='PUBLIC_CHILD_TERMINAL_LOCAL_REVIEW_REQUIRED',tpu_status='PUBLIC_RECORDS_TERMINAL',public_root_gate_sha256=U.sha(self.root/'host/launch.json'),public_receipt_sha256=U.sha(self.fixture.actual/'receipt.json'),public_outer_owner_sha256=U.sha(self.fixture.out/'steps/12-public/ownership.json'))
   if self.fault=='boundary':self.receipt['public_outer_owner_sha256']='0'*64
   self.calls.append(label);return {'status':'PASS','scope':'SIMULATED_NO_CLOUD','argv':list(argv)},''
  return super().__call__(label,argv,timeout)

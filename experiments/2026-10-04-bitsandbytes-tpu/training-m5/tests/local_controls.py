"""Authorized local unqualified20/save10/freshresume smoke and wrong-record controls."""
import copy,json,os,secrets,shutil,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source'));sys.path.insert(0,str(ROOT/'control/ownership'))
import admission as A,math_model as M,training as T,reader as V
from cleanup_lifecycle import Ownership

def main(output):
 with tempfile.TemporaryDirectory(prefix='bnb-m5-local-controls-')as td:
  out=Path(td);owner=Ownership(out);birth=time.time();argv=[sys.executable,'-B',str(ROOT/'source/training.py'),'prepare','--local-reference','--output',str(out/'actual'),'--process-token',secrets.token_hex(16),'--allocation-epoch',str(birth),'--deadline-epoch',str(birth+300),'--phase-seconds','90']
  with(out/'stdout').open('wb')as stdout,(out/'stderr').open('wb')as stderr:
   with owner.guard(320):
    p=owner.launch(argv,record=out/'ownership.json',stdout=stdout,stderr=stderr,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'));p.wait(timeout=310)
  A.write(out/'closure.json',owner.summary());A.require(p.returncode==0,'LOCAL_COORDINATOR_FAILURE:'+str(p.returncode));C=T.setup(local=True);C.refs=T.references(C);root=out/'actual';rows=[]
  def check(name,fn,want='REJECT'):
   try:r=fn();got='PASS'if r is None or r.get('numerical_status')=='PASS'else'NUMERICAL_FAIL';reason=None
   except Exception as e:got='REJECT';reason=str(e)
   A.require(got==want,name+':'+got);rows.append({'name':name,'result':got,'reason':reason})
  report=V.verify(root,out/'closure.json',_context=C);check('complete20_save10_freshresume10',lambda:report,'PASS')
  checkpoint=C.torch.load(root/'save/checkpoint.pt',weights_only=True,map_location='cpu');inputs=M.digest(M.input_descriptor(C))
  for name,mutate in [
   ('wrong_optimizer_lr',lambda c:c['optimizer']['state']['param_groups'][0].update(lr=.02)),
   ('wrong_optimizer_mapping',lambda c:c['optimizer']['state']['param_groups'][0].update(params=[1,0,2,3])),
   ('wrong_adapter_names',lambda c:c['adapter_names'].reverse()),
   ('wrong_checkpoint_step',lambda c:c.update(completed_step=9)),
   ('wrong_generation_binding',lambda c:c.update(source_bundle_sha256='a'*64)),
   ('missing_model_tensor',lambda c:c['model'].pop('layers.0.A')),
   ('wrong_adapter_shape',lambda c:c['model'].update({'layers.0.A':c['model']['layers.0.A'].reshape(-1)})),
   ('nonfinite_adapter',lambda c:c['model']['layers.0.A'].reshape(-1).__setitem__(0,float('nan'))),
   ('changed_frozen_weight_rehashed',lambda c:c['model']['layers.0.base.weight'].reshape(-1).__setitem__(0,int(c['model']['layers.0.base.weight'].reshape(-1)[0])^1)),
  ]:
   def run(mutate=mutate):
    c=copy.deepcopy(checkpoint);mutate(c);c['model_hashes']=M.state_hashes(c['model']);T.checkpoint_check(C,c,inputs)
   check(name,run)
  def wrong_checkpoint_lr(p):
   c=copy.deepcopy(checkpoint);c['optimizer']['state']['param_groups'][0]['lr']=.02;C.torch.save(c,p/'save/checkpoint.pt');r=A.read(p/'resume/receipt.json');r['checkpoint_sha256']=A.sha(p/'save/checkpoint.pt');A.write(p/'resume/receipt.json',r)
  for name,mutate,want in [
   ('changed_gradient_with_updated_envelope',lambda p:A.write(p/'baseline/steps/01.json',{**A.read(p/'baseline/steps/01.json'),'gradients':{**A.read(p/'baseline/steps/01.json')['gradients'],'layers.0.A':{**A.read(p/'baseline/steps/01.json')['gradients']['layers.0.A'],'values':[1.0]*1024}}}),'NUMERICAL_FAIL'),
   ('wrong_checkpoint_lr_full_reader_resealed',wrong_checkpoint_lr,'REJECT'),
   ('false_qualified_claim',lambda p:A.write(p/'oracle-seal.json',{**A.read(p/'oracle-seal.json'),'qualified_runtime':True}),'REJECT'),
   ('wrong_resume_pid',lambda p:A.write(p/'resume/receipt.json',{**A.read(p/'resume/receipt.json'),'pid':987654321}),'REJECT'),
   ('removed_final_update',lambda p:(p/'baseline/steps/20.json').unlink(),'REJECT'),
   ('error_bearing_complete_parent',lambda p:A.write(p/'oracle-seal.json',{**A.read(p/'oracle-seal.json'),'error':{'message':'synthetic retained failure'}}),'REJECT'),
   ('error_bearing_complete_phase',lambda p:A.write(p/'save/receipt.json',{**A.read(p/'save/receipt.json'),'error':{'message':'synthetic retained failure'}}),'REJECT'),
  ]:
   with tempfile.TemporaryDirectory(prefix='bnb-m5-wrong-')as wd:
    p=Path(wd)/'actual';shutil.copytree(root,p);mutate(p)
    for phase in ['baseline','save','resume']:
     q=p/phase/'receipt.json';r=A.read(q);r['artifacts']=C.B.inventory(p/phase,'receipt.json');A.write(q,r)
    q=p/'oracle-seal.json';r=A.read(q)
    for child in r['children']:child['receipt_sha256']=A.sha(p/child['phase']/'receipt.json')
    r['artifacts']=C.B.inventory(p,'oracle-seal.json');A.write(q,r);check(name,lambda:V.verify(p,_context=C),want)
  broken=copy.deepcopy(owner.summary());broken['groups'][0]['errors']=[{'synthetic':'wrong closure'}];A.write(out/'wrong-closure.json',broken);check('wrong_owned_group_closure',lambda:V.verify(root,out/'wrong-closure.json',_context=C))
  result={'status':'PASS','scope':'LOCAL_UNQUALIFIED_ONLY','controls':rows,'control_count':len(rows),'source_bundle_sha256':T.bundle(),'science_manifest_sha256':A.sha(ROOT/'source/science-manifest.json'),'protocol_sha256':M.PROTOCOL_SHA,'inputs_sha256':inputs,'smoke_receipt_sha256':A.sha(root/'oracle-seal.json'),'checkpoint_sha256':A.sha(root/'save/checkpoint.pt'),'checkpoint_bytes':(root/'save/checkpoint.pt').stat().st_size,'closure':owner.summary(),'reader':{k:v for k,v in report.items()if k not in ['gates','independent_dense_gates']},'physical_bytes':sum(p.stat().st_size for p in root.rglob('*')if p.is_file()),'parent_pid':A.read(root/'oracle-seal.json')['pid'],'child_pids':[x['launched_pid']for x in A.read(root/'oracle-seal.json')['children']],'runtime':A.read(root/'baseline/receipt.json')['runtime'],'provider_calls':0,'bulk_deleted':True,'actual_m4_acceptance':'NOT_SUPPLIED','m5_status':'NOT_QUALIFIED'}
 A.write(output,result);print(json.dumps({'controls':len(rows),'status':'PASS','bulk_deleted':True,'qualified':False}))
if __name__=='__main__':main(Path(sys.argv[1]))

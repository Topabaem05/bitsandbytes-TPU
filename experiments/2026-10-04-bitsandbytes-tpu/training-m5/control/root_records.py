"""Explicit root source/adoption/dispatch decisions. Never write an M4 or M5 acceptance."""
import argparse,hashlib,json,math,re,secrets,sys,time
from pathlib import Path
import contract as C
import cli_transport as T
import browser_adoption as B
HERE=Path(__file__).resolve().parent

def fresh(path,value):
 path=Path(path);C.require(not path.exists()and not path.is_symlink(),'FRESH_ROOT_RECORD_NO_CLOCK_RESET');path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x')as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')

def empty(kind,text):
 if kind=='sessions':C.require('No active sessions found on server.'in text and'Hardware:'not in text,'EMPTY_SERVER')
 else:
  import driver;C.require(driver.usage_zero(text)['zero_observed'],'ZERO_USAGE')

def request(review,packet_sha,now=None):
 now=time.time()if now is None else now;prior=C.read(review/'preconnect.json');C.require(prior['status']=='EMPTY_SERVER_ZERO_USAGE_CLOSED'and prior['packet_sha256']==packet_sha and 0<=now-prior['epoch']<=180,'FRESH_PRECONNECT_BINDING')
 fresh(review/'allocation-request.json',{'allocation_epoch':now,'deadline_epoch':now+3600,'hardware':'V5E1','variant':'TPU','packet_sha256':packet_sha,'one_allocation_only':True,'preconnect_sha256':C.sha(review/'preconnect.json')})
 return C.read(review/'allocation-request.json')

def observation(review,packet_sha,stdout,now=None):
 now=time.time()if now is None else now;marker=C.read(review/'browser-marker.json');expected='M5_BROWSER_MARKER '+marker['sha256']+' '+str(marker['bytes']);r=C.read(review/'allocation-request.json')
 C.require(stdout.strip()==expected and r['packet_sha256']==packet_sha and r['allocation_epoch']<=now<r['deadline_epoch']-660,'OBSERVED_EXACT_ORIGINAL_MARKER')
 fresh(review/'browser-observation.json',{'status':'ROOT_OBSERVED_EMPTY_NOTEBOOK_V5E1_MARKER','hardware':'V5E1','variant':'TPU','epoch':now,'marker_stdout':expected,'marker_sha256':marker['sha256'],'packet_sha256':packet_sha,'allocation_request_sha256':C.sha(review/'allocation-request.json')})

def observed(review,packet_sha,now=None):
 now=time.time()if now is None else now;r=C.read(review/'allocation-request.json');o=C.read(review/'browser-observation.json');marker=C.read(review/'browser-marker.json')
 C.require(r['packet_sha256']==o['packet_sha256']==packet_sha and r['one_allocation_only']is True and r['deadline_epoch']==r['allocation_epoch']+3600 and now<r['deadline_epoch']-660,'ORIGINAL_REQUEST_BINDING_DEADLINE')
 C.require(o['status']=='ROOT_OBSERVED_EMPTY_NOTEBOOK_V5E1_MARKER'and o['hardware']=='V5E1'and o['variant']=='TPU'and o['allocation_request_sha256']==C.sha(review/'allocation-request.json')and o['marker_sha256']==marker['sha256']and o['marker_stdout']=='M5_BROWSER_MARKER '+marker['sha256']+' '+str(marker['bytes'])and type(o['epoch'])in(int,float)and math.isfinite(o['epoch'])and r['allocation_epoch']<=o['epoch']<=now and now-o['epoch']<=180,'FRESH_ROOT_UI_MARKER')
 C.require(marker['sha256']==hashlib.sha256(marker['body'].encode()).hexdigest()and marker['bytes']==len(marker['body'].encode()),'MARKER_BODY_BINDING')
 return r,o,marker

def source_review(source,output,decision,revision):
 C.require(decision=='ACCEPT_SOURCE_AFTER_ROOT_REVIEW'and re.fullmatch('[0-9a-f]{40}',revision)is not None,'EXPLICIT_ROOT_SOURCE_REVIEW');C.source(source)
 fresh(output,{'kind':'ROOT_M5_SOURCE_REVIEW','status':'SOURCE_REVIEW_ACCEPTED_FOR_EXPERIMENT','science_manifest_sha256':C.science(source),'source_admission_sha256':C.sha(source/'integration/source-admission.json'),'reviewed_base_commit':revision,'m4_candidate_source_tar_sha256':'dae4003539f09e3c784dc77234ecc817503059655d2b282f2210df8585297020','device_acceptance':None})

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['source-review','check','preconnect','request','observe','dispatch','fresh-closure','tab-closure']);p.add_argument('--source',type=Path,default=HERE.parent);p.add_argument('--packet',type=Path);p.add_argument('--packet-zip',type=Path);p.add_argument('--packet-sha256');p.add_argument('--review',type=Path);p.add_argument('--output',type=Path);p.add_argument('--reviewer-decision');p.add_argument('--reviewed-base-commit');p.add_argument('--cli-python',type=Path);p.add_argument('--cli-identity',type=Path);p.add_argument('--runtime-wheels',type=Path);p.add_argument('--marker-output');p.add_argument('--tab-id');p.add_argument('--tab-absent',action='store_true');a=p.parse_args()
 if a.command=='source-review':source_review(a.source,a.output,a.reviewer_decision,a.reviewed_base_commit);return
 m=C.packet_preflight(a.packet,a.packet_zip,a.packet_sha256)
 if a.command=='check':
  if a.cli_python:T.verify_cli_identity(a.cli_python,C.read(a.cli_identity))
  if a.runtime_wheels:C.runtime_cache(a.packet,a.runtime_wheels)
  print(json.dumps({'status':'EXACT_M5_PACKET_DEPENDENCY_AND_SOURCE_PASS','provider_calls':0,'m5_status':'NOT_QUALIFIED'}));return
 C.require(a.review is not None,'EXPLICIT_ROOT_REVIEW_DIRECTORY');review=a.review.resolve()
 if a.command in('preconnect','dispatch','fresh-closure'):T.verify_cli_identity(a.cli_python,C.read(a.cli_identity))
 if a.command=='preconnect':
  C.require(a.output is not None and not a.output.exists(),'FRESH_ACTUAL_OUTPUT');C.require(a.runtime_wheels is not None,'EXISTING_LOCKED_LINUX_CACHE');C.runtime_cache(a.packet,a.runtime_wheels);review.mkdir(exist_ok=False)
  cli=T.OfficialCLI(a.cli_python,review/'preconnect')
  for index,kind in enumerate(['sessions','usage']):
   r,text=cli(str(index)+'-'+kind,[kind],60);C.require(r['status']=='PASS'and r['exit_code']==0,'PRECONNECT_READ');C.closed(r['cleanup']);empty(kind,text)
  nonce=secrets.token_hex(16);raw=(json.dumps({'format':'bnb-tpu.browser-marker.v1','scope':'M5_EXACT1024_TRAINING','nonce':nonce,'packet_sha256':a.packet_sha256},sort_keys=True,separators=(',',':'))+'\n').encode();marker={'path':'/tmp/bnb-tpu-root-browser-'+nonce+'.json','body':raw.decode(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()};fresh(review/'browser-marker.json',marker)
  (review/'browser-marker-cell.py').write_text('from pathlib import Path\nimport hashlib\nb='+repr(raw)+'\np=Path('+repr(marker['path'])+')\nwith p.open("xb")as f:f.write(b)\nprint("M5_BROWSER_MARKER",hashlib.sha256(p.read_bytes()).hexdigest(),len(p.read_bytes()))\n')
  fresh(review/'preconnect.json',{'status':'EMPTY_SERVER_ZERO_USAGE_CLOSED','epoch':time.time(),'packet_sha256':a.packet_sha256,'cli_identity_sha256':C.sha(a.cli_identity),'runtime_lock_sha256':C.sha(a.packet/'runtime/requirements.lock.json'),'allocation':'NOT_RUN'});return
 if a.command=='request':request(review,a.packet_sha256);return
 if a.command=='observe':observation(review,a.packet_sha256,a.marker_output);return
 if a.command=='tab-closure':
  C.require(a.tab_id and a.tab_absent and a.output is not None and(a.output/'driver.json').exists(),'ROOT_OBSERVED_OWNED_TAB_CLOSURE');fresh(review/'browser-tab-closure.json',{'status':'OWNED_BROWSER_TAB_CLOSED','tab_id':a.tab_id,'tab_absent':True,'epoch':time.time(),'provenance':'ROOT_OBSERVED_UI_CLOSURE'});return
 if a.command=='fresh-closure':
  cli=T.OfficialCLI(a.cli_python,review/'fresh-resource-closure');r=[]
  for index,kind in enumerate(['sessions','usage']):
   rec,text=cli(str(index)+'-'+kind,[kind],45);C.require(rec['status']=='PASS'and rec['exit_code']==0,'FRESH_CLOSURE_READ');C.closed(rec['cleanup']);empty(kind,text);r.append(rec)
  fresh(review/'fresh-resource-closure.json',{'status':'NO_ACTIVE_SESSION_OR_USAGE','epoch':time.time(),'commands':r,'cli_identity_sha256':C.sha(a.cli_identity),'allocation':'NOT_RUN'});return
 C.require(a.reviewer_decision=='AUTHORIZE_ONE_ALLOCATION_AFTER_ROOT_REVIEW'and a.output is not None and not a.output.exists(),'EXPLICIT_FRESH_ROOT_DISPATCH');req,obs,marker=observed(review,a.packet_sha256)
 r,text=T.OfficialCLI(a.cli_python,review/'dispatch-read')('sessions',['sessions'],60);C.require(r['status']=='PASS'and r['exit_code']==0,'FRESH_ASSIGNMENT_READ');C.closed(r['cleanup']);lines=[l for l in text.splitlines()if'Hardware:'in l];C.require(len(lines)==1,'EXACT_ONE_ASSIGNMENT');match=re.fullmatch(r'\[\?\] ([A-Za-z0-9_-]{1,256}) \| Hardware: V5E1 \| Shape: Standard \| Variant: TPU',lines[0]);C.require(match is not None,'ONE_UNREGISTERED_V5E1_ENDPOINT')
 directory=review/'dispatch';directory.mkdir();nonce=json.loads(marker['body'])['nonce'];adoption={'format':'bnb-tpu.browser-adoption.v1','status':'ROOT_CREATED_BROWSER_RUNTIME_READY','endpoint':match[1],'session':'bnb-tpu-m5-'+nonce,'hardware':'V5E1','variant':'TPU','authuser':'0','allocation_epoch':req['allocation_epoch'],'observed_epoch':time.time(),'packet_sha256':a.packet_sha256,'driver_sha256':C.sha(HERE/'driver.py'),'cli_identity_sha256':C.sha(a.cli_identity),'marker_path':marker['path'],'marker_sha256':marker['sha256']};fresh(directory/'browser-adoption.json',adoption);B.record(directory/'browser-adoption.json',C.sha(directory/'browser-adoption.json'))
 dispatch={'kind':'ROOT_M5_EXACT1024_DISPATCH','status':'ACTUAL_DISPATCH_AUTHORIZED','packet_sha256':a.packet_sha256,'driver_sha256':C.sha(HERE/'driver.py'),'cli_identity_sha256':C.sha(a.cli_identity),'science_manifest_sha256':C.science(a.packet),'m4_acceptance_sha256':m['m4_acceptance_sha256'],'allocation_seconds':3600,'allocation_attempts':1,'output':str(a.output.resolve()),'browser_observation_sha256':C.sha(review/'browser-observation.json'),'allocation_request_sha256':C.sha(review/'allocation-request.json'),'browser_adoption_sha256':C.sha(directory/'browser-adoption.json'),'original_allocation_epoch':req['allocation_epoch'],'m5_status':'NOT_QUALIFIED'};fresh(directory/'root-dispatch.json',dispatch)
 argv=[sys.executable,'-B',str(HERE/'driver.py'),'--packet',str(a.packet.resolve()),'--packet-zip',str(a.packet_zip.resolve()),'--packet-sha256',a.packet_sha256,'--cli-python',str(a.cli_python),'--cli-identity',str(a.cli_identity),'--dispatch',str(directory/'root-dispatch.json'),'--browser-adoption',str(directory/'browser-adoption.json'),'--browser-adoption-sha256',C.sha(directory/'browser-adoption.json'),'--output',str(a.output.resolve())]
 if a.runtime_wheels:argv+=['--runtime-wheels',str(a.runtime_wheels)]
 fresh(directory/'host-config.json',{'argv':argv,'absolute_deadline_epoch':req['deadline_epoch'],'host_output':str(review/'host-owner'),'packet_sha256':a.packet_sha256});print(json.dumps({'status':'ROOT_DISPATCH_READY','host_config':str(directory/'host-config.json')}))
if __name__=='__main__':main()

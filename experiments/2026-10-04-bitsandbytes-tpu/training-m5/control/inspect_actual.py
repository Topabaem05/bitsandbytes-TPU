"""Read back the whole recovered archive and fresh resource closure. No acceptance writer."""
import argparse,json,shutil,tempfile
from pathlib import Path
import contract as C
import host_audit as H
import kernel_protocol as K

def kernel_readbacks(packet,actual,driver,review):
 import kernel_reader
 return kernel_reader.verify(packet,actual,driver,review)

def inspect(packet,archive,packet_sha,actual,review):
 packet=packet.resolve();actual=actual.resolve();review=review.resolve();archive=archive.resolve()
 m=C.packet_preflight(packet,archive,packet_sha);d=C.read(actual/'driver.json');kernel_evidence=kernel_readbacks(packet,actual,d,review);dispatch=C.read(review/'dispatch/root-dispatch.json');host=C.read(review/'host-owner/host-finalization.json');config=C.read(review/'dispatch/host-config.json')
 C.require(d['packet_sha256']==dispatch['packet_sha256']==host['packet_sha256']==packet_sha and d['allocation_attempts']==0,'ONE_REVIEWED_BROWSER_PACKET')
 birth=dispatch['original_allocation_epoch'];C.require(d['allocation_epoch']==birth and d['original_deadline_epoch']==config['absolute_deadline_epoch']==host['absolute_deadline_epoch']==birth+3600 and d['finished_epoch']<=host['finished_epoch']<=birth+3600,'ORIGINAL_LIFECYCLE_DEADLINE')
 C.require(host['status']=='DRIVER_EXITED'and host['exit_code']in(0,2)and host['config_sha256']==C.sha(review/'dispatch/host-config.json'),'HOST_DRIVER_OUTCOME');C.closed(host['cleanup'])
 C.require(d['server_empty_observed']is True and d['usage_zero_observed']is True and not d.get('closure_errors')and d['status']!='BLOCKED_CLEANUP','DRIVER_SERVER_USAGE_CLOSURE')
 for row in d['commands']:C.closed(row['cleanup'])
 import kernel_reader
 fresh=kernel_reader.fresh_resource_closure(packet,actual,d,review)
 tab=C.read(review/'browser-tab-closure.json');C.require(tab['status']=='OWNED_BROWSER_TAB_CLOSED'and tab['tab_absent']is True and tab['epoch']>=d['finished_epoch'],'ROOT_OBSERVED_BROWSER_TAB_CLOSURE')
 path=Path(d['last_recovered_records']);C.require(path.is_relative_to(actual),'EXACT_OWNED_RECOVERY_PATH');envelope=d['last_recovered_envelope'];z=path.parent/'evidence.zip'
 with tempfile.TemporaryDirectory(prefix='bnb-m5-independent-envelope-')as td:
  restored=Path(td)/'records';C.recover(z,envelope['sha256'],restored);C.require(C.inventory(restored)==C.inventory(path),'RECOVERED_WHOLE_ARCHIVE_BYTES')
 phase='device'if(path/'device').exists()else'cpu';result=H.audit(packet,path,d['remote_base'],phase,packet_sha)
 C.require(phase=='device'and d['status']=='TRAINING20_RESUME10_RECOVERED_ROOT_REVIEW_REQUIRED'and host['exit_code']==0,'COMPLETE_TRAINING_REQUIRED')
 return {**result,'kernel_evidence':kernel_evidence,'independent_whole_archive_sha256':envelope['sha256'],'resource_closure':'FRESH_READBACK_PASS','physical_fresh_closure':fresh,'m5_status':'NOT_QUALIFIED','root_milestone_review':'REQUIRED'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--packet-zip',type=Path,required=True);p.add_argument('--packet-sha256',required=True);p.add_argument('--actual',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 try:r=inspect(a.packet,a.packet_zip,a.packet_sha256,a.actual,a.review)
 except Exception as e:r={'record_validation':'FAIL','numerical_status':'NOT_EVALUATED','m5_status':'NOT_QUALIFIED','execution_status':'PARTIAL_OR_INVALID','error':{'type':type(e).__name__,'message':str(e)}}
 C.write(a.output,r);print(json.dumps({k:v for k,v in r.items()if k not in ['gates','independent_dense_gates']}));raise SystemExit(0 if r.get('record_validation')==r.get('numerical_status')=='PASS'else 2)

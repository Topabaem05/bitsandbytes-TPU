"""Synthetic SDK-shaped worker. No network, credentials, or provider import."""
import argparse,hashlib,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();q=json.loads(a.request.read_text());cfg=json.loads(Path(os.environ['M8_CONTINUATION_FIXTURE']).read_text());op=q['operation'];r=q['request'];identity=cfg['admitted']['identity'];plan=cfg['plan']
if op=='submit':raise AssertionError('SUBMIT_REACHED_FAKE_SDK')
if op=='source':
 result={'ref':identity['owner']+'/'+identity['slug'],'kernel_id':identity['kernel_id'],'current_version_number':1,'is_private':True,'language':'python','kernel_type':'script','source_sha256':plan['wrapper_sha256'],'machine_shape':plan['machine_shape']}
 if cfg['fault']=='source':result['current_version_number']=2
elif op=='status':result={'status':'COMPLETE','failure_message':''}
elif op=='outputs':result={'files':['m8-evidence-manifest.json','m8-evidence.tar'],'next_page_token':''}
elif op=='download':
 raw=Path(cfg['artifacts'][r['filePath']]).read_bytes();(a.output/'download.bin').write_bytes(raw);result={'artifact':'download.bin','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
else:raise AssertionError(op)
response={'operation':op,'request':r,'result':result,'source_pins_match':True,'pid':os.getpid(),'pgid':os.getpgid(0),'parent_pid':os.getppid(),'process_token':q['process_token'],'scope':'SYNTHETIC_PROVIDER_CONTROL'};(a.output/'response.json').write_text(json.dumps(response))

"""Prepare root browser evidence locally. This script does not create a runtime."""
import argparse,json,secrets,time
from pathlib import Path
import contract as C
import browser_adoption as B

def prepare(packet_sha,driver,identity,session,output):
 output=Path(output);C.require(not output.exists(),'FRESH_BROWSER_INTENT');output.mkdir();birth=time.time();marker='/tmp/bnb-tpu-root-browser-'+secrets.token_hex(16)+'.json';body=json.dumps({'kind':'ROOT_M4_BROWSER_MARKER','allocation_budget_start_epoch':birth,'packet_sha256':packet_sha,'driver_sha256':C.sha(driver),'cli_identity_sha256':C.sha(identity),'session':session},indent=2,sort_keys=True)+'\n';marker_sha=__import__('hashlib').sha256(body.encode()).hexdigest()
 intent={'format':'bnb-tpu.browser-intent.v1','allocation_epoch':birth,'packet_sha256':packet_sha,'driver_sha256':C.sha(driver),'cli_identity_sha256':C.sha(identity),'session':session,'marker_path':marker,'marker_sha256':marker_sha};C.write(output/'intent.json',intent)
 (output/'first-cell.py').write_text('from pathlib import Path\nimport hashlib\n'+f'p=Path({marker!r});assert not p.exists()and not p.is_symlink();p.write_text({body!r});assert hashlib.sha256(p.read_bytes()).hexdigest()=={marker_sha!r}\nprint("ROOT_BROWSER_MARKER_READY",'+repr(marker_sha)+')\n');return intent

def finalize(intent,endpoint,hardware,output):
 i=C.read(intent);r={'format':'bnb-tpu.browser-adoption.v1','status':'ROOT_CREATED_BROWSER_RUNTIME_READY','endpoint':endpoint,'session':i['session'],'hardware':hardware,'variant':'TPU','authuser':'0','allocation_epoch':i['allocation_epoch'],'observed_epoch':time.time(),'packet_sha256':i['packet_sha256'],'driver_sha256':i['driver_sha256'],'cli_identity_sha256':i['cli_identity_sha256'],'marker_path':i['marker_path'],'marker_sha256':i['marker_sha256']};C.require(not Path(output).exists(),'FRESH_BROWSER_ADOPTION');C.write(output,r);B.record(output,C.sha(output));return {'adoption':str(output),'sha256':C.sha(output),'provider_calls':0}
if __name__=='__main__':
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True);a=s.add_parser('prepare');a.add_argument('--packet-sha256',required=True);a.add_argument('--driver',type=Path,required=True);a.add_argument('--cli-identity',type=Path,required=True);a.add_argument('--session',required=True);a.add_argument('--output',type=Path,required=True);a=s.add_parser('finalize');a.add_argument('--intent',type=Path,required=True);a.add_argument('--endpoint',required=True);a.add_argument('--hardware',choices=('V5E1','V6E1'),required=True);a.add_argument('--output',type=Path,required=True);a=p.parse_args();r=prepare(a.packet_sha256,a.driver,a.cli_identity,a.session,a.output)if a.command=='prepare'else finalize(a.intent,a.endpoint,a.hardware,a.output);print(json.dumps(r))

"""Local preflight and explicit root dispatch. No provider calls."""
import argparse,json,sys
from pathlib import Path
import contract as C,cli_transport as T,browser_adoption as B
HERE=Path(__file__).resolve().parent

def preflight(a):
 m=C.packet_preflight(a.packet,a.packet_zip,a.packet_sha256);return {'status':'DIAGNOSTIC_PACKET_PREFLIGHT_PASS','generation':C.GENERATION,'packet_sha256':a.packet_sha256,'driver_sha256':C.sha(HERE/'driver.py'),'manifest_sha256':C.sha(a.packet/'manifest.json'),'files':len(m['files']),'provider_calls':0,'m4_status':'NOT_QUALIFIED'}
def authorize(a):
 C.require(a.reviewer_decision=='AUTHORIZE_ONE_FOCUSED_DIAGNOSTIC_AFTER_ROOT_REVIEW','EXPLICIT_ROOT_REVIEW_DECISION');preflight(a);T.verify_cli_identity(a.cli_python,C.read(a.cli_identity));adoption=B.record(a.browser_adoption,a.browser_adoption_sha256);C.require(adoption['packet_sha256']==a.packet_sha256 and adoption['driver_sha256']==C.sha(HERE/'driver.py')and adoption['cli_identity_sha256']==C.sha(a.cli_identity),'ROOT_DIAGNOSTIC_BROWSER_BINDING');C.budget(adoption['allocation_epoch'],1,reserve=360);out=a.output.resolve();C.require(not out.exists(),'FRESH_ROOT_DIAGNOSTIC_DISPATCH');out.mkdir()
 dispatch=out/'dispatch.json';C.write(dispatch,{'kind':'ROOT_M4_WORD_DIAGNOSTIC_DISPATCH_V1','status':'ACTUAL_DISPATCH_AUTHORIZED','generation':C.GENERATION,'packet_sha256':a.packet_sha256,'driver_sha256':C.sha(HERE/'driver.py'),'cli_identity_sha256':C.sha(a.cli_identity),'diagnostic_seconds':600,'allocation_attempts':0,'allocation_seconds':3600,'m4_status':'NOT_QUALIFIED'})
 argv=[sys.executable,'-B',str(HERE/'driver.py')]
 for flag,value in[('--packet',a.packet.resolve()),('--packet-zip',a.packet_zip.resolve()),('--packet-sha256',a.packet_sha256),('--cli-python',a.cli_python),('--cli-identity',a.cli_identity.resolve()),('--dispatch',dispatch),('--browser-adoption',a.browser_adoption.resolve()),('--browser-adoption-sha256',a.browser_adoption_sha256),('--output',out/'actual')]:argv.extend([flag,str(value)])
 config=out/'host-config.json';C.write(config,{'argv':argv,'absolute_deadline_epoch':adoption['allocation_epoch']+600,'packet_sha256':a.packet_sha256,'host_output':str(out/'host-owner')});return {'status':'ROOT_DIAGNOSTIC_DISPATCH_PREPARED','host_run_argv':[sys.executable,'-B',str(HERE/'host_run.py'),'--config',str(config)],'provider_calls':0,'m4_status':'NOT_QUALIFIED'}
if __name__=='__main__':
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
 for command in('preflight','authorize'):
  q=sub.add_parser(command);q.add_argument('--packet',type=Path,required=True);q.add_argument('--packet-zip',type=Path,required=True);q.add_argument('--packet-sha256',required=True)
  if command=='authorize':q.add_argument('--cli-python',type=Path,required=True);q.add_argument('--cli-identity',type=Path,required=True);q.add_argument('--browser-adoption',type=Path,required=True);q.add_argument('--browser-adoption-sha256',required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--reviewer-decision',required=True)
 a=p.parse_args();print(json.dumps(preflight(a)if a.command=='preflight'else authorize(a)))

"""Write source/dispatch records only on explicit root decisions. Never device acceptance."""
import argparse,json,sys
from pathlib import Path
import contract as C

def source_review(source,output,decision):
 C.require(decision=='ACCEPT_SOURCE_AFTER_ROOT_REVIEW','ROOT_SOURCE_DECISION_REQUIRED');n=Path(source)/'nested-exact';C.require(C.sha(n/'source/science-manifest.json')==C.SCIENCE_SHA and C.sha(n/'integration/source-admission.json')==C.ADMISSION_SHA and C.sha(n/'integration/adoption-map.json')==C.MAP_SHA and C.sha(n/'integration/patch-manifest.json')==C.PATCH_SHA,'ROOT_SOURCE_PINNED_DOCUMENTS')
 sys.path.insert(0,str(n/'source'));import generation_gate as G
 G.verify(n/'integration/source-admission.json',C.ADMISSION_SHA,upstream=n/'integration/upstream/bitsandbytes',plugin=n/'integration/plugin')
 for name,r in C.read(n/'source/science-manifest.json')['files'].items():C.require(C.sha(n/'source'/C.name(name))==r['sha256']and(n/'source'/name).stat().st_size==r['bytes'],'ROOT_SCIENCE_FILE_BYTES')
 C.require(not Path(output).exists(),'FRESH_ROOT_SOURCE_REVIEW');C.write(output,{'kind':'ROOT_NESTED_EXACT_SOURCE_REVIEW','status':'SOURCE_REVIEW_ACCEPTED_FOR_EXPERIMENT','integration_map_sha256':C.MAP_SHA,'source_admission_sha256':C.ADMISSION_SHA,'science_map_sha256':C.SCIENCE_SHA,'device_acceptance':None});return {'output':str(output),'sha256':C.sha(output),'provider_calls':0}

def dispatch(packet,archive,expected,cli_python,identity,output,decision):
 C.require(decision=='AUTHORIZE_ONE_ALLOCATION_AFTER_ROOT_REVIEW','ROOT_DISPATCH_DECISION_REQUIRED');C.packet_preflight(packet,archive,expected)
 import cli_transport as T;T.verify_cli_identity(cli_python,C.read(identity));C.require(not Path(output).exists(),'FRESH_ROOT_DISPATCH');C.write(output,{'kind':'ROOT_M4_EXACT_NESTED_DISPATCH','status':'ACTUAL_DISPATCH_AUTHORIZED','packet_sha256':expected,'driver_sha256':C.sha(Path(__file__).with_name('driver.py')),'cli_identity_sha256':C.sha(identity),'science_manifest_sha256':C.SCIENCE_SHA,'allocation_seconds':3600,'allocation_attempts':1});return {'output':str(output),'sha256':C.sha(output),'provider_calls':0}
if __name__=='__main__':
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True);s=sub.add_parser('source-review');s.add_argument('--source',type=Path,required=True);s.add_argument('--output',type=Path,required=True);s.add_argument('--reviewer-decision',required=True);s=sub.add_parser('dispatch');s.add_argument('--packet',type=Path,required=True);s.add_argument('--packet-zip',type=Path,required=True);s.add_argument('--packet-sha256',required=True);s.add_argument('--cli-python',type=Path,required=True);s.add_argument('--cli-identity',type=Path,required=True);s.add_argument('--output',type=Path,required=True);s.add_argument('--reviewer-decision',required=True);a=p.parse_args();r=source_review(a.source,a.output,a.reviewer_decision)if a.command=='source-review'else dispatch(a.packet,a.packet_zip,a.packet_sha256,a.cli_python,a.cli_identity,a.output,a.reviewer_decision);print(json.dumps(r))

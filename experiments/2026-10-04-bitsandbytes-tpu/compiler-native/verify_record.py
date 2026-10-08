"""Verify a sealed private boundary record against an independently sealed expected contract."""
import argparse
import importlib.util
import json
from pathlib import Path
import traceback
import verifier as V

PINS={'backend_probe':'e8743e33e6a0454d5d05b77075d47a5c9f983cf32f3956c38f2b544f86f5ef6d','precision_probe':'e0534955507e5f91e67ecddfffc738a367ad752e69a4eea7ae8052c6d76a8852'}
def checked(path,expected):
    body=Path(path).read_bytes();V.require(V.digest(body)==expected,'ARTIFACT_HASH_'+Path(path).name);return body

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('record','record-sha256','hlo','hlo-sha256','expected','expected-sha256','backend-probe','precision-probe','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    try:
        for key,pin in PINS.items():checked(getattr(args,key),pin)
        B=load(args.backend_probe,'r6verifiedB');P=load(args.precision_probe,'r6verifiedP')
        record=json.loads(checked(args.record,args.record_sha256),object_pairs_hook=V.unique_pairs);expected=json.loads(checked(args.expected,args.expected_sha256),object_pairs_hook=V.unique_pairs);hlo=checked(args.hlo,args.hlo_sha256).decode()
        report=V.verify(record,hlo,expected,B,P);report.update(record_sha256=args.record_sha256,hlo_sha256=args.hlo_sha256,expected_sha256=args.expected_sha256);code=0
    except Exception as error:report={'status':'REJECTED','error_type':type(error).__name__,'error':str(error),'m6':'NOT_QUALIFIED','traceback':traceback.format_exc()};code=2
    Path(args.output).write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n');return code
if __name__=='__main__':raise SystemExit(main())

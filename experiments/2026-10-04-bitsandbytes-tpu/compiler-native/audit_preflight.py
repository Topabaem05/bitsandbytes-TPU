"""External-owned host CPU frontend preflight. No provider or device initialization."""
import argparse
import json
import os
from pathlib import Path
import conversion_recovery as C
import verifier as V
def run(python,output,deadline):
    V.require(os.getpid()==os.getpgid(0),'AUDIT_PREFLIGHT_EXTERNAL_OWNER_LEADER')
    result=C.replay({},python,deadline)
    Path(output).write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--python',required=True);p.add_argument('--output',required=True);p.add_argument('--deadline-epoch',type=float,required=True)
    a=p.parse_args();run(a.python,a.output,a.deadline_epoch)

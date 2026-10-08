"""Explicit, externally supervised provider entry. No default provider call."""
import argparse,json,sys
from pathlib import Path
import batch
from recovery import PartialM2Runner,recover
from supervisor import BrokerProvider,supervise
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    run=sub.add_parser('execute');run.add_argument('--live-provider',action='store_true',required=True)
    run.add_argument('--candidate',type=Path,required=True);run.add_argument('--output',type=Path,required=True);run.add_argument('--kaggle-python',type=Path,required=True)
    child=sub.add_parser('_worker')
    for k in ['candidate','output']:child.add_argument('--'+k,type=Path,required=True)
    child.add_argument('--broker-fd',type=int,required=True);child.add_argument('--broker-token',required=True);child.add_argument('--deadline-epoch',type=float,required=True)
    close=sub.add_parser('close');close.add_argument('--candidate',type=Path,required=True);close.add_argument('--output',type=Path,required=True);close.add_argument('--root-observation',type=Path,required=True)
    a=p.parse_args();plan=batch.read(a.candidate/'plan.json')
    if a.command=='close':
        owner=PartialM2Runner.__new__(PartialM2Runner);owner.plan=plan;owner.root=a.output/'owner';owner.report=batch.read(a.output/'owner/owner.json')
        batch.need(owner.report['binding']==plan['binding'],'CLOSURE_PLAN_BINDING')
        result=owner.complete_closure(batch.read(a.root_observation))
    elif a.command=='_worker':
        batch.need(a.deadline_epoch==plan['deadline_epoch'],'BROKER_ORIGINAL_DEADLINE')
        provider=BrokerProvider(a.broker_fd,a.broker_token,plan['deadline_epoch'])
        callback=lambda members,binding:recover(members,binding,a.candidate,a.output/'recovered')
        owner=PartialM2Runner(plan,provider,a.output/'owner',callback);result=owner.run()
    else:
        report=supervise([sys.executable,'-B',str(HERE/'owner_cli.py'),'_worker','--candidate',str(a.candidate.resolve()),'--output',str(a.output.resolve())],
            [str(a.kaggle_python),'-B',str(HERE/'sdk_worker.py'),'--live-provider'],a.output/'supervision',plan['deadline_epoch'])
        path=a.output/'owner/owner.json'
        if not path.exists():
            print(json.dumps({'state':report['status'],'record_validation':'NOT_RUN','closure':'UNCONFIRMED','m8_status':'NOT_QUALIFIED'}));return 2
        result=batch.read(path)
        if report['status']!='OWNER_FINISHED':
            print(json.dumps({'state':report['status'],'original_scientific_state':result['state'],'closure':result['closure'],'m8_status':'NOT_QUALIFIED'}));return 2
    print(json.dumps({'state':result['state'],'record_validation':result['record_validation'],'numerical_status':result['numerical_status'],'closure':result['closure'],'m8_status':result['m8_status']},sort_keys=True))
    return 0 if result['record_validation']=='PASS' else 2

if __name__=='__main__':raise SystemExit(main())

"""Portable host controls. All device, ownership and numerical records are declared fixtures."""
import argparse,json
from pathlib import Path
import mosaic_paths as M
def run(args):
    native=M.admit(args.native);args.output.mkdir(exist_ok=False)
    import mosaic_audit_controls as A,mosaic_recovery_controls as R,mosaic_failure_controls as F,mosaic_harness_controls as H
    A.run(args.output/'audit',str(args.jax_python),str(args.ambient_python),args.fixtures)
    R.run(args.output/'recovery',args.fixtures/'cpu-oracle',str(args.jax_python),str(args.ambient_python),args.fixtures)
    F.run(args.output/'failure',native)
    H.run(args.output/'harness')
    reports={name:json.loads((args.output/name/'results.json').read_text())for name in('audit','recovery','failure','harness')}
    assert all(row['status']=='PASS'for row in reports.values())
    report={'status':'PASS','count':sum(row['count']for row in reports.values()),'phases':reports,'scope':'CPU_FRONTEND_REPLAY_AND_SYNTHETIC_NATIVE_RECORDS','actual_TPU':'NOT_RUN','qualified_linux_cpu':'NOT_RUN','M6':'NOT_QUALIFIED'}
    (args.output/'results.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');print(json.dumps({'status':'PASS','count':report['count']}))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in('native','fixtures','jax-python','ambient-python','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())

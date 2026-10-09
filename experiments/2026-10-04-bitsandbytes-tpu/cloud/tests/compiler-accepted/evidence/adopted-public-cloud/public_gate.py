"""Owned host CPU15 gate. No provider or device client."""
import argparse,json
from pathlib import Path
import public_contract as U
p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--recovered',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--runtime-wheels',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
gate,observations=U.cpu_gate(a.packet,a.recovered,U.read(a.receipt),a.runtime_wheels);U.write(a.output,{'gate':gate,'observations':observations});print(json.dumps({'status':gate['status'],'case_count':observations['case_count']}))

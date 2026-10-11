"""Portable new7 replay. See run_controls.py for original16 and full replay."""
import sys
from run_controls import run
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--historical-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.historical_root,a.output,'new')

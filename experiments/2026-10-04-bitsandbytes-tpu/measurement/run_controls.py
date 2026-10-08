"""Run measurement controls in a fresh isolated directory. No device or provider call."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
FILES = ('collector.py', 'protocol.json', 'xla_runner.py', 'controls.py', 'repair_controls.py')


def run(output):
    output = Path(output).resolve()
    if output == HERE or output in HERE.parents or HERE in output.parents:
        raise ValueError('CONTROL_OUTPUT_MUST_BE_OUTSIDE_SOURCE')
    output.mkdir(parents=True, exist_ok=False)
    for name in FILES:
        shutil.copyfile(HERE / name, output / name)
    subprocess.run([sys.executable, '-B', str(output / 'controls.py'),
                    '--output', str(output / 'baseline-controls')], check=True)
    subprocess.run([sys.executable, '-B', str(output / 'repair_controls.py')], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)

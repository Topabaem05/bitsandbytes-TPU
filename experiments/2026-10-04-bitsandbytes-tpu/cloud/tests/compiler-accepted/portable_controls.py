"""Repeat reviewed analyzer controls from small byte-sealed local fixtures."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent/'compiler-native'))
import argparse,hashlib,json,subprocess,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent

def run(output):
    output=Path(output).resolve();output.mkdir(exist_ok=False)
    inventory=json.loads((HERE/'fixtures/compiler-inventory.json').read_text())
    for name,sealed in inventory.items():
        raw=(HERE/'fixtures/compiler'/name).read_bytes()
        assert len(raw)==sealed['bytes'] and hashlib.sha256(raw).hexdigest()==sealed['sha256']
    rows=[]
    for name in ('controls','independent_controls','dump_inventory_controls'):
        result=subprocess.run([sys.executable,'-B',str(HERE/(name+'.py')),'--output',str(output/name)],cwd=output,timeout=30,capture_output=True)
        (output/(name+'.stdout')).write_bytes(result.stdout);(output/(name+'.stderr')).write_bytes(result.stderr)
        assert result.returncode==0,(name,result.returncode,result.stderr.decode())
        report=json.loads((output/name/'results.json').read_text());assert report['status']=='PASS';rows.append({'suite':name,'count':report['count'],'status':'PASS'})
    report={'status':'PASS','count':sum(r['count'] for r in rows),'suites':rows,'scope':'SYNTHETIC_NO_COMPILER_TPU','selected_executable_link':'UNKNOWN','allocator_peak':'UNKNOWN'}
    (output/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)

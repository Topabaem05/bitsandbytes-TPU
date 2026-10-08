"""Read static admitted hash literals without importing device modules."""
import ast,hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def literal(path,name):
 nodes=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id==name for t in n.targets)]
 if len(nodes)!=1:raise ValueError('STATIC_PIN_ASSIGNMENT:'+name)
 return ast.literal_eval(nodes[0].value)
def validate(native):
 native=Path(native);kernel=sha(native/'kernel.py');pins=sha(native/'source-pins.json')
 if literal(native/'adapter.py','KERNEL_SHA')!=kernel:raise ValueError('ADAPTER_KERNEL_PIN')
 if literal(native/'device_diagnostic.py','PINS_SHA')!=pins:raise ValueError('DIAGNOSTIC_PINS_PIN')
 if json.loads((native/'source-pins.json').read_text())['files']['.work/r6-pallas-preparation/kernel.py']!=kernel:raise ValueError('SOURCE_KERNEL_PIN')
 m=json.loads((native/'manifest.json').read_text())
 for n in ('kernel.py','source-pins.json','adapter.py','device_diagnostic.py'):
  if m['sources'][n]!={'sha256':sha(native/n),'bytes':(native/n).stat().st_size}:raise ValueError('MANIFEST_STATIC_PIN:'+n)
 return {'status':'PASS','kernel_sha256':kernel,'source_pins_sha256':pins,'adapter_kernel_sha256':literal(native/'adapter.py','KERNEL_SHA'),'diagnostic_pins_sha256':literal(native/'device_diagnostic.py','PINS_SHA')}

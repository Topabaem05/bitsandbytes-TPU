"""Write stdlib records for the ordinary-native phase admission control."""
import argparse,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
for name,value in [('parent.json',{'status':'COMPLETE','pid':os.getpid(),'pgid':os.getpgrp(),'scope':'OWNED_STDLIB_ADMISSION_FIXTURE_ONLY'}),('expected.json',{'scope':'NO_NATIVE_OR_COMPILER_SCIENCE'})]:(a.output/name).write_text(json.dumps(value,sort_keys=True)+'\n')

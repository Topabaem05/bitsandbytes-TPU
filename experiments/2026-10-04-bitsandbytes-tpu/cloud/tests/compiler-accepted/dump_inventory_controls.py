"""Isolated raw-dump boundary controls; no compiler/device invocation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent/'compiler-native'))
import argparse, hashlib, json, os
from pathlib import Path
import dump_inventory as D

def run(output):
    output.mkdir(exist_ok=False);rows=[]
    good=output/'correct';good.mkdir();raw=b'fixed synthetic bytes';(good/'module_0901.before_optimizations.hlo.pb').write_bytes(raw);(good/'module_0901.metadata.textproto').write_text('canonical_module_id: 901\n');(good/'unexpected-lib-version-format.unknown').write_bytes(b'retained')
    report=D.capture(good);assert report['files']['module_0901.before_optimizations.hlo.pb']['sha256']==hashlib.sha256(raw).hexdigest();assert report['files']['unexpected-lib-version-format.unknown']['format_hint']=='UNKNOWN_FORMAT_RETAINED';assert report['selected_executable_link']=='UNKNOWN';rows.append({'case':'correct-bytes-and-unrecognized-schema-retained','status':'PASS'})
    def reject(name,prepare,reason,**limits):
        root=output/name;root.mkdir();prepare(root)
        try:D.capture(root,**limits)
        except ValueError as error:assert str(error)==reason,(name,str(error));rows.append({'case':name,'status':'PASS','rejected':reason})
        else:raise AssertionError('FALSE_ACCEPT:'+name)
    reject('file-symlink-outside',lambda r:os.symlink(good/'module_0901.before_optimizations.hlo.pb',r/'escaped'),'DUMP_SYMLINK')
    reject('directory-symlink-outside',lambda r:os.symlink(good,r/'escaped'),'DUMP_SYMLINK')
    reject('fifo',lambda r:os.mkfifo(r/'pipe'),'DUMP_FILE_TYPE_OR_HARDLINK')
    reject('hardlink',lambda r:os.link(good/'module_0901.before_optimizations.hlo.pb',r/'linked'),'DUMP_FILE_TYPE_OR_HARDLINK')
    reject('file-too-large',lambda r:(r/'large').write_bytes(b'12345'),'DUMP_FILE_SIZE_LIMIT',max_file_bytes=4)
    reject('total-too-large',lambda r:[(r/str(i)).write_bytes(b'123')for i in range(2)],'DUMP_TOTAL_SIZE_LIMIT',max_total_bytes=5)
    reject('entry-limit',lambda r:[(r/str(i)).mkdir()for i in range(3)],'DUMP_ENTRY_LIMIT',max_entries=2)
    result={'status':'PASS','count':len(rows),'controls':rows,'scope':'LOCAL_FILES_ONLY_NO_COMPILER_OR_DEVICE'};(output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'PASS','count':len(rows)}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();run(a.output.resolve())

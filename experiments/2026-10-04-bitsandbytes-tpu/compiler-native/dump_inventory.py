"""Bounded raw dump inventory. It proves no compiler stage, ownership, or execution link."""
import argparse, hashlib, json, os, stat
from pathlib import Path

MAX_ENTRIES=1024
MAX_FILE_BYTES=16*1024*1024
MAX_TOTAL_BYTES=128*1024*1024

def capture(directory,*,max_entries=MAX_ENTRIES,max_file_bytes=MAX_FILE_BYTES,max_total_bytes=MAX_TOTAL_BYTES):
    directory=Path(directory).absolute()
    if directory.is_symlink() or not directory.is_dir():raise ValueError('DUMP_DIRECTORY_TYPE')
    if directory.resolve()!=directory:raise ValueError('DUMP_DIRECTORY_ANCESTOR_SYMLINK')
    files={};seen=0;total=0
    pending=[(os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW),Path('.'))]
    try:
        while pending:
            parent_fd,prefix=pending.pop()
            try:
                with os.scandir(parent_fd) as entries:
                    for entry in entries:
                        seen+=1
                        if seen>max_entries:raise ValueError('DUMP_ENTRY_LIMIT')
                        before=entry.stat(follow_symlinks=False)
                        if stat.S_ISLNK(before.st_mode):raise ValueError('DUMP_SYMLINK')
                        if stat.S_ISDIR(before.st_mode):
                            child_fd=os.open(entry.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent_fd)
                            opened=os.fstat(child_fd)
                            if (opened.st_dev,opened.st_ino)!=(before.st_dev,before.st_ino):
                                os.close(child_fd);raise ValueError('DUMP_CHANGED_DURING_CAPTURE')
                            pending.append((child_fd,prefix/entry.name));continue
                        if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1:raise ValueError('DUMP_FILE_TYPE_OR_HARDLINK')
                        if before.st_size>max_file_bytes:raise ValueError('DUMP_FILE_SIZE_LIMIT')
                        total+=before.st_size
                        if total>max_total_bytes:raise ValueError('DUMP_TOTAL_SIZE_LIMIT')
                        digest=hashlib.sha256();count=0
                        with os.fdopen(os.open(entry.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=parent_fd),'rb') as source:
                            opened=os.fstat(source.fileno())
                            if (opened.st_dev,opened.st_ino)!=(before.st_dev,before.st_ino):raise ValueError('DUMP_CHANGED_DURING_CAPTURE')
                            while raw:=source.read(1024*1024):
                                count+=len(raw)
                                if count>max_file_bytes:raise ValueError('DUMP_FILE_SIZE_LIMIT')
                                digest.update(raw)
                            after=os.fstat(source.fileno())
                        if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns) or count!=before.st_size or after.st_nlink!=1:raise ValueError('DUMP_CHANGED_DURING_CAPTURE')
                        name=(prefix/entry.name).as_posix()
                        if name.endswith('.metadata.textproto'):hint='MODULE_METADATA_TEXT'
                        elif name.endswith('.hlo.pb'):hint='HLO_PROTO_ENVELOPE'
                        elif name.endswith('-buffer-assignment.txt'):hint='BUFFER_ASSIGNMENT_TEXT'
                        elif name.endswith('-memory-usage-report.txt'):hint='COMPILER_MEMORY_REPORT_TEXT'
                        elif name.endswith('.txt'):hint='TEXT_UNKNOWN_SCHEMA'
                        else:hint='UNKNOWN_FORMAT_RETAINED'
                        files[name]={'sha256':digest.hexdigest(),'bytes':count,'format_hint':hint}
            finally:os.close(parent_fd)
    finally:
        for fd,_ in pending:os.close(fd)
    return {'status':'RAW_DUMP_SNAPSHOT_UNLINKED','files':dict(sorted(files.items())),'total_bytes':total,'entries':seen,'limits':{'entries':max_entries,'file_bytes':max_file_bytes,'total_bytes':max_total_bytes},'ownership':'REQUIRES_SEPARATE_REVIEWED_CLOSURE','compiler_stage_link':'UNKNOWN','selected_executable_link':'UNKNOWN','allocator_peak':'UNKNOWN','filename_hints_are_identity':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();result=capture(a.directory)
    with a.output.open('x') as target:target.write(json.dumps(result,sort_keys=True,indent=2)+'\n')

"""Reviewed bounded live observation. No hard aggregate quota."""
import os, stat

def observe(directory,identity,limits):
    current=os.lstat(directory)
    if not stat.S_ISDIR(current.st_mode) or (current.st_dev,current.st_ino)!=identity:raise ValueError('DUMP_ROOT_REPLACED')
    root_fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    opened=os.fstat(root_fd)
    if (opened.st_dev,opened.st_ino)!=identity:
        os.close(root_fd);raise ValueError('DUMP_ROOT_REPLACED')
    pending=[root_fd];entries=total=largest=0
    result={'entries':0,'total_bytes':0,'largest_file_bytes':0,'status':'WITHIN_OBSERVED_LIMITS'}
    try:
        while pending:
            fd=pending.pop()
            try:
                with os.scandir(fd) as scan:
                    for entry in scan:
                        entries+=1;result['entries']=entries
                        if entries>limits['entries']:result['status']='DUMP_ENTRY_LIMIT';return result
                        s=entry.stat(follow_symlinks=False)
                        if stat.S_ISDIR(s.st_mode):
                            child=os.open(entry.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);opened=os.fstat(child)
                            if (s.st_dev,s.st_ino)!=(opened.st_dev,opened.st_ino):os.close(child);raise ValueError('DUMP_CHANGED_DURING_OBSERVATION')
                            pending.append(child);continue
                        if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1:raise ValueError('DUMP_FILE_TYPE_OR_HARDLINK')
                        total+=s.st_size;largest=max(largest,s.st_size);result.update(total_bytes=total,largest_file_bytes=largest)
                        if s.st_size>limits['file_bytes']:result['status']='DUMP_FILE_SIZE_LIMIT';return result
                        if total>limits['total_bytes']:result['status']='DUMP_TOTAL_SIZE_LIMIT';return result
            finally:os.close(fd)
    finally:
        for fd in pending:os.close(fd)
    return result


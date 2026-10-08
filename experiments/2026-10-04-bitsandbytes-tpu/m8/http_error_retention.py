"""Retain bounded classifications only. Preserve the original exception."""
from contextlib import contextmanager
import hashlib,json,os,re,sys
from types import SimpleNamespace
from pathlib import Path
MAX_BODY=16*1024
KNOWN_TYPES={'HTTPError','ReadTimeout','ConnectTimeout','Timeout','ConnectionError','ValueError','HTTP_RESPONSE_OBSERVED'}
PATTERNS={
 'SOURCE_SIZE':r'(?:source|script|notebook).{0,80}(?:size|large|bytes|megabyte|megabytes|mb)|(?:size|large|bytes|megabyte|megabytes).{0,80}(?:source|script|notebook)',
 'SOURCE_LINE_LIMIT':r'(?:source|script|line).{0,80}(?:line length|too long|character limit)',
 'ACCELERATOR_CONSTRAINT':r'(?:accelerator|machine.?shape|tpu).{0,80}(?:invalid|available|quota|allowed|supported|limit)',
 'SESSION_TIMEOUT_CONSTRAINT':r'(?:session.?timeout|run.?time|timeout).{0,80}(?:invalid|maximum|limit|exceed|between)',
 'MODEL_SOURCE_CONSTRAINT':r'model.{0,40}(?:source|invalid|not found)',
}
def describe(error):
    response=getattr(error,'response',None)  # Do not use Response truthiness: 400 is false.
    status=getattr(response,'status_code',None)
    if type(status) is not int or not 100<=status<=599:status=None
    body=getattr(response,'_content',None)  # Inspect buffered bytes only. Never read a stream.
    result={'format':'m8.sanitized-http-error.v1','operation':'submit','endpoint':'SaveKernel',
      'exception_type':type(error).__name__ if type(error).__name__ in KNOWN_TYPES else 'OTHER',
      'http_status':status,'response_present':response is not None,'response_body':'NOT_BUFFERED',
      'buffered_bytes':None,'prefix_bytes':0,'prefix_sha256':None,'application_code':None,
      'constraint_classes':[],'remote_effect':'UNKNOWN','automatic_retry':False,'acceptance':'NOT_QUALIFIED'}
    if isinstance(body,bytes):
        prefix=body[:MAX_BODY];result.update(buffered_bytes=len(body),prefix_bytes=len(prefix),prefix_sha256=hashlib.sha256(prefix).hexdigest(),response_body='BOUNDED_CLASSIFICATION_ONLY' if len(body)<=MAX_BODY else 'OVERSIZE_NOT_PARSED')
        if len(body)<=MAX_BODY:
            text=body.decode('utf-8',errors='replace')
            try:
                parsed=json.loads(text)
                if type(parsed) is dict and type(parsed.get('code')) is int and 0<=parsed['code']<=599:result['application_code']=parsed['code']
            except (ValueError,RecursionError):pass
            result['constraint_classes']=[name for name,pattern in PATTERNS.items() if re.search(pattern,text,re.I|re.S)]
    return result

def diagnostic(failure):
    try:
        sys.stderr.write(json.dumps({'stage':'HTTP_ERROR_RETENTION','error_type':type(failure).__name__[:64]},sort_keys=True)+'\n')
    except BaseException:
        pass  # Observation must never replace the original SDK response/error.

def record_response_error(response,output,writer=None):
    try:
        record=describe(SimpleNamespace(response=response))
        if not ((record['http_status'] is not None and record['http_status']>=400) or
                (record['application_code'] is not None and record['application_code']>=400)):return False
        record['exception_type']='HTTP_RESPONSE_OBSERVED'
        (writer or write_new)(Path(output)/'http-error.sanitized.json',record)
        return True
    except BaseException as failure:
        diagnostic(failure)
        return False

def write_new(path,record):
    data=(json.dumps(record,sort_keys=True,allow_nan=False)+'\n').encode()
    with open(path,'xb') as f:os.chmod(path,0o600);f.write(data)

@contextmanager
def retained_errors(output,writer=write_new):
    try:yield
    except BaseException as original:
        try:
            path=Path(output)/'http-error.sanitized.json'
            if not path.exists():writer(path,describe(original))
        except BaseException as retention_error:
            # Preserve the primary exception and record only the retention error type.
            try:original.add_note('M8 error retention failed: '+type(retention_error).__name__[:64])
            except BaseException:pass
        raise

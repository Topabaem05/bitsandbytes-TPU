"""Strict pinned debug-log diagnostic. Does not qualify M6 or a loaded handle."""
import hashlib
import math
import re
from lazy_hash_v2 import computation_hash
from module_arity_v2 import module_arity, ArityUnsupported, MAX_FIELDS
VERSION = 'M6_DEBUG_HASH_TRACE_V2_ARITY_REV1'
MAX_ENTRIES, MAX_FILE, MAX_TOTAL = 1024, 16777216, 134217728
BAR, DASH = '=' * 80, '-' * 80
H = r'(?:0|[1-9a-f][0-9a-f]{0,15})[0-9a-f]{16}'

class TraceError(ValueError):
    pass

def require(ok, reason):
    if not ok:
        raise TraceError(reason)

def graph_block(lines, pos, kind, expected_name):
    prefix = kind + ' Analysis: '
    require(lines[pos] == prefix + BAR, 'BLOCK_OPEN')
    begin = pos
    pos += 1
    body = []
    while pos < len(lines) and lines[pos] != prefix + DASH:
        require(lines[pos].startswith(prefix), 'BLOCK_INTERLEAVING_OR_TRUNCATION')
        body.append(lines[pos][len(prefix):]); pos += 1
    require(pos + 1 < len(lines) and lines[pos+1] == prefix + BAR, 'BLOCK_CLOSE')
    cause = 'Compilation Cause' if kind == 'Compilation' else 'Execution Cause'
    require(len(body) >= 9 and body[0] == cause, 'BLOCK_CAUSE')
    require('Graph Info: ' in body and 'Python Frame Triggered Execution: ' in body, 'BLOCK_FIELDS')
    i = body.index('Graph Info: ')
    require(1 < i and i + 5 < len(body), 'BLOCK_ORDER')
    require(all(x.startswith('  ') and x.strip() for x in body[1:i]), 'CAUSE_LINES')
    require(sum('Graph Name:' in x for x in body) == 1 and sum('Graph Hash:' in x for x in body) == 1, 'DUPLICATE_GRAPH_FIELDS')
    require(all(sum(('Number of Graph '+label+':') in x for x in body) == 1 for label in ('Inputs','Outputs')), 'DUPLICATE_ARITY_FIELDS')
    require(body[i+1] == '  Graph Name: ' + expected_name, 'GRAPH_NAME')
    match = re.fullmatch('  Graph Hash: (' + H + ')', body[i+2])
    require(match is not None, 'GRAPH_HASH_FORMAT')
    arity = {}
    for item, label in zip(body[i+3:i+5], ('Inputs', 'Outputs')):
        value = re.fullmatch('  Number of Graph ' + label + r': (0|[1-9][0-9]{0,8})', item)
        require(value is not None, 'GRAPH_ARITY')
        arity[label.lower()] = int(value.group(1))
        require(arity[label.lower()] <= MAX_FIELDS, 'GRAPH_ARITY_BOUND')
    require(body[i+5] == 'Python Frame Triggered Execution: ', 'FRAME_ORDER')
    require(all(x.startswith('  ') and x.strip() for x in body[i+6:]), 'FRAME_LINES')
    require(body[i+6:] and not any('Graph Hash:' in x for x in body[i+6:]), 'FRAME_DUPLICATE')
    return {'kind': kind, 'graph_hash': match.group(1), 'line_start': begin, 'arity': arity}, pos+2

def post_block(lines, pos):
    match = re.fullmatch('Computation hash: (' + H + ')', lines[pos])
    require(match is not None, 'COMPUTATION_HASH_FORMAT')
    pos += 1
    require(pos < len(lines) and lines[pos] == '', 'POST_SEPARATOR'); pos += 1
    p = 'Post Compilation Analysis: '
    require(pos < len(lines) and lines[pos] == p + BAR, 'POST_OPEN'); pos += 1
    fields = ('Graph input size', 'Graph output size', 'Aliased Input size',
              'Intermediate tensor size', 'Compiled program size')
    for label in fields:
        require(pos < len(lines) and re.fullmatch(re.escape(p + label) + r': (?:[0-9]+\.[0-9]+|Unknown ) GB', lines[pos]), 'POST_FIELD')
        pos += 1
    require(pos+1 < len(lines) and lines[pos:pos+2] == [p+DASH,p+BAR], 'POST_CLOSE')
    return match.group(1), pos+2

def parse_window(raw, name, module, submitted_module, cache):
    require(type(raw) is bytes and raw.endswith(b'\n'), 'LOG_TRUNCATED')
    require(len(raw) <= MAX_FILE and type(module) is bytes and len(module) <= MAX_FILE, 'FILE_LIMIT')
    try:
        lines = raw.decode('utf-8', errors='strict').splitlines()
    except UnicodeError as e:
        raise TraceError('LOG_UTF8') from e
    pos, pending, rows = 0, None, []
    expect = computation_hash(module)
    try:
        optimized_arity = module_arity(module)
        submitted_arity = module_arity(submitted_module) if submitted_module is not None else None
    except ArityUnsupported as e:
        raise TraceError(str(e)) from e
    while pos < len(lines):
        if lines[pos] == '': pos += 1; continue
        if lines[pos].startswith('Compilation Analysis: '):
            require(pending is None, 'INTERLEAVED_COMPILATION')
            pending,pos = graph_block(lines,pos,'Compilation',name)
            require(submitted_arity is not None, 'UNSUPPORTED_SUBMITTED_MODULE_REQUIRED')
            require(pending['arity'] == submitted_arity, 'COMPILATION_ARITY_MISMATCH')
            key = pending['graph_hash']
            require(key not in cache, 'RECOMPILE_AMBIGUOUS_CACHE_KEY')
            while pos < len(lines) and lines[pos] == '': pos += 1
            require(pos < len(lines), 'MISSING_POST_COMPILATION')
            actual,pos = post_block(lines,pos)
            require(actual == expect, 'COMPUTATION_HASH_MISMATCH')
            pending['computation_hash'] = actual
        elif lines[pos].startswith('Execution Analysis: '):
            event,pos = graph_block(lines,pos,'Execution',name)
            key = event['graph_hash']
            if pending is not None:
                require(key == pending['graph_hash'], 'COLD_EXECUTION_HASH_MISMATCH')
                cache[key] = {'computation_hash':pending['computation_hash'],
                    'optimized_arity':optimized_arity, 'compilation_arity':pending['arity'],
                    'optimized_module_sha256':hashlib.sha256(module).hexdigest(),
                    'submitted_module_sha256':hashlib.sha256(submitted_module).hexdigest()}; pending = None
                mode = 'COLD_COMPILE_TO_EXECUTION_ANALYSIS'
            else:
                require(key in cache, 'ORPHAN_WARM_EXECUTION')
                mode = 'OBSERVED_WARM_EXECUTION_ANALYSIS'
            require(cache[key]['computation_hash'] == expect, 'WARM_COMPUTATION_HASH_MISMATCH')
            require(cache[key]['optimized_module_sha256'] == hashlib.sha256(module).hexdigest(), 'WARM_MODULE_SHA256_MISMATCH')
            require(event['arity'] == optimized_arity == cache[key]['optimized_arity'], 'EXECUTION_ARITY_MISMATCH')
            rows.append({'graph_hash':key,'computation_hash':expect,'mode':mode,
                'compilation_arity':cache[key]['compilation_arity'], 'execution_arity':event['arity'],
                'optimized_module_arity':optimized_arity,
                'optimized_module_sha256':cache[key]['optimized_module_sha256'],
                'submitted_module_sha256':cache[key]['submitted_module_sha256']})
        else:
            raise TraceError('UNKNOWN_OR_INTERLEAVED_LOG_LINE')
    require(pending is None and rows, 'INCOMPLETE_WINDOW')
    return rows

def verify(request, *, now_epoch):
    require(request.get('kind') == VERSION, 'TRACE_VERSION')
    require(request.get('scope') == {'fresh_process':True,'single_device':True,'serial':True,
        'spmd':False,'eager':False,'persistent_cache':False,'debug_level':2}, 'UNSUPPORTED_SCOPE')
    for key in ('deadline_epoch','parent_deadline_epoch','started_epoch','finished_epoch'):
        require(type(request.get(key)) in (float,int) and math.isfinite(request[key]), 'TIME_SCHEMA')
    require(type(now_epoch) in (float,int) and math.isfinite(now_epoch), 'NOW_SCHEMA')
    require(request['started_epoch'] <= request['finished_epoch'] <= request['deadline_epoch'] <=
        request['parent_deadline_epoch'] and request['finished_epoch'] <= now_epoch, 'ORIGINAL_DEADLINE')
    require(re.fullmatch('[0-9a-f]{32}',request.get('process_nonce','')) is not None, 'PROCESS_NONCE')
    windows = request.get('windows'); raw = request.get('raw_log')
    require(type(windows) is list and [w.get('phase') for w in windows] == ['prepare','target','reference'], 'COMPLETE_PHASE_WINDOWS')
    require(type(raw) is bytes and len(raw) <= MAX_FILE, 'LOG_BOUND')
    require(request.get('raw_log_sha256') == hashlib.sha256(raw).hexdigest(), 'LOG_SEAL')
    cache, result, end, total = {}, [], 0, len(raw)
    require(1 + len(windows) * 2 <= MAX_ENTRIES, 'ENTRY_LIMIT')
    for w in windows:
        require(type(w.get('start')) is int and type(w.get('end')) is int and
            w['start'] == end < w['end'] <= len(raw), 'COMPLETE_CONTIGUOUS_OFFSETS')
        require(w.get('sync_completed') is True and w.get('readback_completed') is True, 'SYNC_READBACK_REQUIRED')
        require(w.get('module_format') == 'DETERMINISTIC_FULL_HLO_MODULE_PROTO', 'MODULE_FORMAT')
        require(type(w.get('module_bytes')) is bytes and len(w['module_bytes']) <= MAX_FILE, 'MODULE_BOUND')
        require(w.get('module_sha256') == hashlib.sha256(w['module_bytes']).hexdigest(), 'MODULE_SEAL')
        require(re.fullmatch('[A-Za-z0-9_-]{1,128}',w.get('graph_name','')) is not None, 'GRAPH_NAME_SCHEMA')
        submitted = w.get('submitted_module_bytes')
        if submitted is not None:
            require(type(submitted) is bytes and len(submitted) <= MAX_FILE, 'SUBMITTED_MODULE_BOUND')
            require(w.get('submitted_module_format') == 'DETERMINISTIC_FULL_HLO_MODULE_PROTO', 'SUBMITTED_MODULE_FORMAT')
            require(w.get('submitted_module_sha256') == hashlib.sha256(submitted).hexdigest(), 'SUBMITTED_MODULE_SEAL')
        total += len(w['module_bytes']) + (len(submitted) if submitted is not None else 0)
        require(total <= MAX_TOTAL, 'TOTAL_LIMIT')
        events = parse_window(raw[w['start']:w['end']],w['graph_name'],w['module_bytes'],submitted,cache)
        result.append({'phase':w['phase'],'events':events}); end = w['end']
    require(end == len(raw), 'UNASSIGNED_LOG_SUFFIX')
    return {'kind':VERSION,'status':'DEBUG_SOURCE_LINK','actual_device_verification':'NOT_RUN',
        'evidence_origin':'CALLER_SUPPLIED_REQUIRES_INDEPENDENT_REVIEW','windows':result,
        'loaded_handle_link':'UNKNOWN','binary_identity':'UNKNOWN','allocator_peak':'UNKNOWN',
        'mosaic_internal_storage':'UNKNOWN','m6_status':'NOT_QUALIFIED'}

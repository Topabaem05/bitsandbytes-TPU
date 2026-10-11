"""Independent host verifier. It does not import recorder code."""
import hashlib
import json
from pathlib import Path

MIB = 1024 * 1024
SCHEMA = 'M6_DIRECT_PJRT_TRACE_V4'


def read_regular(path):
    assert not path.is_symlink() and path.is_file(), 'Non-regular member'
    return path.read_bytes()


def identity(path):
    body = read_regular(path)
    return {'file': path.name, 'bytes': len(body),
            'sha256': hashlib.sha256(body).hexdigest()}


def seal(root, finalizer_return):
    """Call only after the C finalizer returns and no wrapped calls remain."""
    root = Path(root)
    footer = json.loads(read_regular(root / 'footer.json'))
    assert footer['state'] == 'API_FINALIZED' and footer['inflight'] == 0
    assert finalizer_return in (-1, 0)
    assert (finalizer_return == 0) == (footer['capture_status'] == 'OK')
    header = json.loads(read_regular(root / 'journal.jsonl').splitlines()[0])
    names = {x['file'] for x in footer['inventory']} | {'journal.jsonl', 'footer.json'}
    assert {p.name for p in root.iterdir()} == names, 'Unlisted capture member'
    members = [identity(root / name) for name in sorted(names)]
    for ref in footer['inventory']:
        assert ref == identity(root / ref['file']), 'Payload readback mismatch'
    limits = header['limits']
    assert len(members) + 1 <= limits['entries']
    assert all(m['bytes'] <= limits['file'] for m in members)
    result = {'state': 'CLOSED', 'schema': SCHEMA, 'inflight_at_finalize': 0,
              'nonce': header['nonce'], 'members': members, 'finalizer_return': finalizer_return,
              'readback': 'FULL_SIZE_AND_SHA256'}
    body = (json.dumps(result, sort_keys=True, indent=2) + '\n').encode()
    assert len(body) <= MIB and len(body) <= limits['file']
    assert sum(m['bytes'] for m in members) + len(body) <= limits['total']
    with (root / 'closed.json').open('xb') as stream:
        stream.write(body)
    return result


def envelope(body):
    """Locate fields without protobuf reconstruction or normalization."""
    position = 0
    fields = []

    def varint():
        nonlocal position
        value = 0
        for shift in range(0, 70, 7):
            assert position < len(body), 'Truncated varint'
            byte = body[position]
            position += 1
            assert shift < 63 or byte < 2, 'Varint overflow'
            value |= (byte & 127) << shift
            if not byte & 128:
                return value
        raise AssertionError('Long varint')

    while position < len(body):
        tag = varint()
        number, wire = tag >> 3, tag & 7
        assert number > 0, 'Invalid field number'
        if wire == 0:
            varint()
        elif wire in (1, 5):
            position += 8 if wire == 1 else 4
        elif wire == 2:
            length = varint()
            offset = position
            position += length
            if number in (1, 2):
                fields.append({'field': number, 'offset': offset, 'bytes': length,
                               'sha256': hashlib.sha256(body[offset:position]).hexdigest()})
        else:
            raise AssertionError('Unsupported wire type')
        assert position <= len(body), 'Truncated field'
    assert sum(f['field'] == 1 for f in fields) == 1, 'Missing or duplicate HLO field'
    assert sum(f['field'] == 2 for f in fields) <= 1, 'Duplicate config field'
    return fields


def verify(root):
    root = Path(root)
    closed = json.loads(read_regular(root / 'closed.json'))
    assert closed['state'] == 'CLOSED' and closed['readback'] == 'FULL_SIZE_AND_SHA256'
    members = closed['members']
    assert len({x['file'] for x in members}) == len(members)
    names = {m['file'] for m in members}
    assert names | {'closed.json'} == {p.name for p in root.iterdir()}
    for item in members:
        assert '/' not in item['file'] and item['file'] not in ('.', '..')
        assert item == identity(root / item['file']), 'Closed inventory mismatch'
    records = [json.loads(line) for line in read_regular(root / 'journal.jsonl').splitlines()]
    assert records and records[0]['type'] == 'header'
    header, calls = records[0], records[1:]
    assert header['schema'] == SCHEMA and closed['schema'] == SCHEMA
    assert closed['nonce'] == header['nonce'] and header['extra_observer_probes'] is False
    assert header['claim'] == 'DIRECT_LOADED_HANDLE_EXECUTE_CALL'
    assert header['clock'] == 'CLOCK_MONOTONIC'
    assert len(header['provider_sha256']) == 64 and header['recorder_id']
    limits = header['limits']
    assert limits['entries'] <= 1024 and limits['total'] <= 128 * MIB
    assert limits['file'] <= 16 * MIB and limits['metadata_reserve'] == 3 * MIB
    assert len(members) + 1 <= limits['entries']
    assert sum(x['bytes'] for x in members) + (root / 'closed.json').stat().st_size <= limits['total']
    assert all(x['bytes'] <= limits['file'] for x in members)
    footer = json.loads(read_regular(root / 'footer.json'))
    assert footer['state'] == 'API_FINALIZED' and footer['inflight'] == 0
    assert (closed['finalizer_return'] == 0) == (footer['capture_status'] == 'OK')
    assert footer['last_seq'] == max([0] + [c['seq'] for c in calls])
    payload_names = {x['file'] for x in footer['inventory']}
    assert payload_names | {'journal.jsonl', 'footer.json'} == names
    for ref in footer['inventory']:
        assert ref == identity(root / ref['file'])
    assert all(c['type'] == 'call' for c in calls)
    assert len(calls) + 1 <= 1024
    assert [c['seq'] for c in calls] == sorted({c['seq'] for c in calls})
    steps = []
    call_ids, seqs = set(), set()
    for c in calls:
        assert c['call'] not in call_ids
        call_ids.add(c['call'])
        assert c['pre_seq'] < c['seq'] and c['pre_ns'] <= c['post_ns']
        for seq in (c['pre_seq'], c['seq']):
            assert seq not in seqs
            seqs.add(seq)
        steps.extend([(c['pre_seq'], 'begin', c), (c['seq'], 'end', c)])
    assert seqs == set(range(1, len(calls) * 2 + 1)), 'Missing journal sequence'
    handles, active, bad, queries, fills = {}, {}, set(), {}, {}
    epochs, query_snapshots, execute_snapshots, begin_valid = {}, {}, {}, {}
    claims, extracted = [], []
    operations = {'compile', 'deserialize', 'get_executable', 'optimized',
                  'execute', 'loaded_destroy', 'loaded_delete', 'executable_destroy'}
    for _, event, c in sorted(steps, key=lambda step: step[0]):
        hid, op = c['handle'], c['op']
        assert op in operations, 'Unknown creator or operation'
        h = handles.get(hid)
        if h:
            expected_kind = 'E' if op in ('optimized', 'executable_destroy') else 'L'
            assert h['kind'] == expected_kind, 'Handle kind mismatch'
        if event == 'begin':
            begin_valid[c['call']] = bool(h and h['alive'] and
                                          not h['deleted'] and hid not in bad)
            if hid:
                if active.get(hid):
                    bad.add(hid)
                active.setdefault(hid, set()).add(c['call'])
            if op == 'optimized':
                # Every BEGIN supersedes the preceding payload, even on error.
                assert c['optimized_epoch'] == c['pre_seq'], 'Invalid optimized epoch'
                fills.pop(hid, None)
                query_snapshots[c['call']] = queries.pop(hid, None)
                epochs[hid] = c['pre_seq']
            elif op == 'execute':
                # Evidence completed during Execute cannot qualify that call.
                execute_snapshots[c['call']] = {
                    exe: (fill, epochs[exe]) for exe, fill in fills.items()
                    if handles[exe]['parent'] == hid and exe not in bad}
            continue
        if hid:
            active[hid].remove(c['call'])
        replaced = c.get('replaced_alive')
        if replaced:
            bad.add(replaced)
            bad.add(c.get('created'))
        created = c.get('created')
        if created:
            assert op in ('compile', 'deserialize', 'get_executable'), 'Invalid creator operation'
            kind = 'L' if op in ('compile', 'deserialize') else 'E'
            assert created == header['nonce'] + '-' + kind + '-' + str(c['seq'])
            assert created not in handles
            assert not c['error_returned']
            if op == 'get_executable':
                assert c['parent'] == hid, 'GetExecutable parent differs from input handle'
                parent_valid = bool(h and h['kind'] == 'L' and
                                    begin_valid[c['call']] and h['alive'] and
                                    not h['deleted'] and hid not in bad and
                                    c['pre_valid'] and c['post_valid'])
                if not parent_valid:
                    bad.add(created)
            handles[created] = {'kind': kind, 'alive': True, 'deleted': False, 'origin': op,
                                'parent': c.get('parent'), 'source': c.get('submitted'),
                                'format_ref': c.get('format_ref'), 'creator_call': c['call'],
                                'creator_phase': c['phase']}
            if not c['created_valid']:
                bad.add(created)
        if op in ('loaded_destroy', 'executable_destroy', 'loaded_delete') and not c['error_returned'] and h:
            if op == 'loaded_delete':
                h['deleted'] = True
            else:
                h['alive'] = False
        valid = h and h['alive'] and not h['deleted'] and hid not in bad
        observed_valid = c['pre_valid'] and c['post_valid'] and begin_valid[c['call']]
        if op == 'optimized' and not c['error_returned'] and valid and observed_valid:
            assert c['mode'] in ('query', 'fill'), 'Unknown optimized mode'
            if epochs[hid] != c['pre_seq']:
                continue
            if c['mode'] == 'query' and not c['capture_failure']:
                queries[hid] = c
            elif c['mode'] == 'fill' and c['payload'] and not c['capture_failure']:
                query = query_snapshots.pop(c['call'], None)
                assert query and query['seq'] == c['query_seq']
                assert query['seq'] < c['pre_seq'], 'Query not completed before fill'
                assert query['returned_size'] == c['query_size'] == c['payload']['bytes']
                assert c['query_size'] <= c['capacity'] == c['returned_size']
                assert query['format'] == c['format']
                assert c['payload'] == identity(root / c['payload']['file'])
                fields = envelope(read_regular(root / c['payload']['file'])) if c['format'] == 'hlo_with_config' else []
                extracted.append({'executable': hid, 'payload': c['payload'],
                                  'format': c['format'], 'fields': fields})
                fills[hid] = c
        if op == 'execute' and valid and observed_valid and not c['error_returned']:
            eligible = []
            if h['origin'] == 'compile' and h['source'] and h['format_ref']:
                assert h['source'] == identity(root / h['source']['file'])
                assert h['format_ref'] == identity(root / h['format_ref']['file'])
                for exe, (fill, epoch) in execute_snapshots[c['call']].items():
                    eh = handles[exe]
                    # E destruction does not erase its historical link to live L.
                    # An optimized BEGIN during Execute invalidates this candidate.
                    if eh['parent'] == hid and exe not in bad and epochs[exe] == epoch:
                        assert fill['seq'] < c['pre_seq']
                        eligible.append(exe)
            if eligible:
                claims.append({'call': c['call'], 'phase': c['phase'], 'loaded': hid,
                               'compile_call': h['creator_call'], 'compile_phase': h['creator_phase'],
                               'submitted': h['source'], 'format_ref': h['format_ref'],
                               'executables': sorted(eligible)})
    # A later duplicate makes both generations ambiguous for this capture.
    claims = [x for x in claims if x['loaded'] not in bad and all(e not in bad for e in x['executables'])]
    status = 'PASS' if claims and footer['capture_status'] == 'OK' else 'UNKNOWN'
    return {'status': status, 'claim': header['claim'], 'claims': claims,
            'extracted_fields': extracted, 'calls': len(calls),
            'capture_status': footer['capture_status'], 'failure': footer['failure'],
            'device_completion': 'NOT_OBSERVED', 'TPU_execution': 'NOT_RUN'}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--seal', action='store_true')
    parser.add_argument('--finalizer-return', type=int, choices=(-1, 0), default=0)
    options = parser.parse_args()
    if options.seal:
        seal(options.root, options.finalizer_return)
    print(json.dumps(verify(options.root), sort_keys=True, indent=2))

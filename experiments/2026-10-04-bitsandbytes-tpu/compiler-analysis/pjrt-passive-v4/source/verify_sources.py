"""Verify pinned primary bodies and retained ranges without retaining bodies."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--offline', action='store_true')
options = parser.parse_args()
proof = Path(__file__).resolve().parent.parent / 'proof'
records = []
for path in sorted(proof.glob('*.excerpt.json')):
    source = json.loads(path.read_text())
    for section in source['sections']:
        assert hashlib.sha256(section['text'].encode()).hexdigest() == section['sha256']
    if not options.offline:
        body = urllib.request.urlopen(source['url'], timeout=30).read()
        assert len(body) == source['bytes']
        assert hashlib.sha256(body).hexdigest() == source['sha256']
        lines = body.decode().splitlines(keepends=True)
        for section in source['sections']:
            assert ''.join(lines[section['start'] - 1:section['end']]) == section['text']
        del body, lines
    records.append({'file': path.name, 'sections': len(source['sections']), 'result': 'PASS'})
print(json.dumps({'status': 'PASS', 'sources': records,
                  'scope': 'RETAINED_EXCERPTS_ONLY' if options.offline else 'FULL_PINNED_BODIES_VERIFIED_AND_DISCARDED'}, indent=2))

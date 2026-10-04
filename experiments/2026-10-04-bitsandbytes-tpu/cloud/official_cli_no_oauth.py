"""Use the official CLI. Disable new OAuth flows and redact proxy URL tokens."""
import re
import sys
import colab_cli.auth as auth
from colab_cli.cli import main

class RedactedStream:
    def __init__(self, wrapped):
        self.wrapped = wrapped
    def write(self, value):
        return self.wrapped.write(re.sub(r'(colab-runtime-proxy-token=)[^\s&]+', r'\1<REDACTED>', value))
    def __getattr__(self, name):
        return getattr(self.wrapped, name)

def deny_new_oauth(*args, **kwargs):
    raise RuntimeError('New OAuth flow disabled for this admitted experiment')

auth._run_remote_flow = deny_new_oauth
sys.stdout = RedactedStream(sys.stdout)
sys.stderr = RedactedStream(sys.stderr)
sys.argv = ['colab', *sys.argv[1:]]
main()

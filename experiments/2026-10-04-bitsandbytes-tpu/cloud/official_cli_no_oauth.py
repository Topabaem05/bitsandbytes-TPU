"""Official CLI with explicit opt-in allocation transport; no new OAuth flow."""
import re
import sys
import colab_cli.auth as auth
from colab_cli.cli import main as official_main


class RedactedStream:
    def __init__(self,wrapped): self.wrapped = wrapped
    def write(self,value):
        return self.wrapped.write(re.sub(r'(colab-runtime-proxy-token=)[^\s&]+',r'\1<REDACTED>',value))
    def __getattr__(self,name): return getattr(self.wrapped,name)


def deny_new_oauth(*args,**kwargs):
    raise RuntimeError('New OAuth flow disabled for this admitted experiment')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    auth._run_remote_flow = deny_new_oauth  # Same restriction as the original wrapper.
    sys.stdout = RedactedStream(sys.stdout); sys.stderr = RedactedStream(sys.stderr)
    if argv and argv[0] in ('--adopt-root-browser-v1','--remove-browser-provisional-v1'):
        import json
        import colab_cli.common as common
        from browser_adoption import command
        print(json.dumps(command(argv,common),sort_keys=True))
        return
    if '--allocation-transport-v1' not in argv:
        sys.argv = ['colab',*argv]
        return official_main()
    from allocation_transport import validate_opt_in, adapted_client_factory
    argv = validate_opt_in(argv)
    import colab_cli.common as common
    sys.argv = ['colab',*argv]
    with adapted_client_factory(common):
        return official_main()


if __name__ == '__main__': main()

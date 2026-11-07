"""Download pinned upstreams into ignored third_party/; never touch configured local clones."""
import argparse
import json
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / 'backends.lock.json').read_text())

def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backends', nargs='*', choices=list(LOCK), default=list(LOCK))
    args = parser.parse_args()
    for name in args.backends:
        spec = LOCK[name]
        path = ROOT / 'third_party' / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git','clone','--no-checkout',spec['url'],str(path)],check=True)
            git(path, 'checkout', '--detach', spec['commit'])
        if git(path,'rev-parse','HEAD') != spec['commit']:
            raise RuntimeError(f'{name}: revision mismatch; left untouched')
        if git(path,'status','--porcelain'):
            raise RuntimeError(f'{name}: modified checkout; left untouched')
        git(path,'submodule','update','--init','--recursive')
        print(name, spec['commit'])
if __name__ == '__main__': main()

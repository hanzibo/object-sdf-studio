"""Apply the bundled NVIDIA-licensed compatibility patch to a pinned checkout."""
import argparse
import json
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', nargs='?', type=Path, default=ROOT/'third_party/BundleSDF')
    args = parser.parse_args()
    path = args.checkout.expanduser().resolve()
    def git(*args, **kwargs):
        return subprocess.run(['git','-C',str(path),*args],text=True,capture_output=True,**kwargs)
    expected = json.loads((ROOT/'backends.lock.json').read_text())['BundleSDF']['commit']
    if git('rev-parse','HEAD',check=True).stdout.strip() != expected:
        raise RuntimeError('Checkout does not match pinned BundleSDF revision')
    patch = str(ROOT/'patches/bundlesdf/local-compatibility.patch')
    if git('apply','--reverse','--check',patch).returncode == 0:
        print('Patch already applied'); return
    if git('status','--porcelain',check=True).stdout.strip():
        raise RuntimeError('Refusing to patch a modified checkout')
    check = git('apply','--check',patch)
    if check.returncode:
        raise RuntimeError(check.stderr)
    git('apply',patch,check=True)
    print('Applied compatibility patch')
if __name__ == '__main__': main()

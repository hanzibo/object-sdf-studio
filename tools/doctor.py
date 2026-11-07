"""Check configured paths; --imports checks backend packages in separate processes."""
import argparse
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import config

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imports',action='store_true')
    args=parser.parse_args(); missing=[]
    for key in ['GUI_PYTHON','CAMERA_PYTHON','SAM_PYTHON','BUNDLE_PYTHON','POSE_PYTHON','SAM_PROJECT','SAM_CHECKPOINT','BUNDLE_PROJECT','POSE_PROJECT']:
        value=getattr(config,key);ok=value.exists()
        print(('OK  ' if ok else 'MISS'),key,value)
        if not ok:missing.append(key)
    if args.imports:
        probes=[('GUI_PYTHON','import app; from PySide6 import QtCore; print(QtCore.__version__)'),
                ('CAMERA_PYTHON','import cv2, numpy, pyorbbecsdk; print("camera imports OK")'),
                ('SAM_PYTHON','import torch; print(torch.__version__, torch.cuda.is_available())'),
                ('BUNDLE_PYTHON',f'import sys; sys.path.insert(0, {str(config.BUNDLE_PROJECT)!r}); sys.path.insert(0, {str(config.BUNDLE_PROJECT/"BundleTrack/build")!r}); import torch; from mycuda import common; import my_cpp; print(torch.__version__, torch.cuda.is_available())'),
                ('POSE_PYTHON','import torch; print(torch.__version__, torch.cuda.is_available())')]
        for key,code in probes:
            if getattr(config,key).exists():
                result=subprocess.run([str(getattr(config,key)),'-c',code],cwd=config.ROOT,timeout=60)
                if result.returncode:missing.append(key+' imports')
    if missing:raise SystemExit('Incomplete: '+', '.join(missing))
    print('Path/import checks passed; this does not validate hardware or inference.')
if __name__=='__main__':main()

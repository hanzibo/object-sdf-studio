"""Portable paths. Environment > ignored local JSON > repository defaults.

Relative paths in JSON resolve against the JSON file's directory.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOME = Path.home()
CONFIG_FILE = Path(os.environ.get('STUDIO_CONFIG', ROOT / 'config.local.json')).expanduser().resolve()
_settings = json.loads(CONFIG_FILE.read_text()) if CONFIG_FILE.exists() else {}

def _path(key, default):
    value = os.environ.get('STUDIO_' + key.upper(), _settings.get(key, str(default)))
    result = Path(os.path.expandvars(str(value))).expanduser()
    return (CONFIG_FILE.parent / result).resolve() if not result.is_absolute() else result

ENVS = _path('conda_envs', HOME / 'miniconda3/envs')
CAMERA_PYTHON = _path('camera_python', ENVS / 'object-studio/bin/python')
GUI_PYTHON = _path('gui_python', CAMERA_PYTHON)
SAM_PYTHON = _path('sam_python', ENVS / 'sam3/bin/python')
BUNDLE_PYTHON = _path('bundle_python', ENVS / 'bundlesdf/bin/python')
POSE_PYTHON = _path('pose_python', ENVS / 'foundationpose/bin/python')
SAM_PROJECT = _path('sam_project', ROOT / 'third_party/sam3')
SAM_CHECKPOINT = _path('sam_checkpoint', ROOT / 'checkpoints/sam3/sam3.pt')
BUNDLE_PROJECT = _path('bundle_project', ROOT / 'third_party/BundleSDF')
POSE_PROJECT = _path('pose_project', ROOT / 'third_party/FoundationPose')
DATASETS = ROOT / 'datasets'
EXPORTS = ROOT / 'exports'
RUNTIME = ROOT / 'runtime'

"""Start the configured GUI interpreter without activating a shell environment."""
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT, GUI_PYTHON
if not GUI_PYTHON.is_file():
    raise SystemExit('GUI Python missing. Create object-studio or configure config.local.json.')
target = 'trainer.py' if '--trainer' in sys.argv else 'app.py'
args = [arg for arg in sys.argv[1:] if arg != '--trainer']
for key in ['PYTHONPATH', 'QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_PLUGIN_PATH']:
    os.environ.pop(key, None)
from config import RUNTIME
RUNTIME.mkdir(parents=True, exist_ok=True)
log = os.open(RUNTIME / ('trainer_gui.log' if target == 'trainer.py' else 'gui.log'), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
os.dup2(log, 1)
os.dup2(log, 2)
os.close(log)
os.chdir(ROOT)
os.execv(str(GUI_PYTHON), [str(GUI_PYTHON), str(ROOT/target), *args])

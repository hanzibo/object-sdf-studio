"""Install user desktop/application launchers for this checkout."""
from pathlib import Path
import os
import subprocess
ROOT = Path(__file__).resolve().parents[1]
try:
    desktop = Path(subprocess.check_output(['xdg-user-dir','DESKTOP'],text=True).strip())
except (OSError, subprocess.CalledProcessError):
    desktop = Path.home()/'Desktop'
for folder in [Path.home()/'.local/share/applications', desktop]:
    folder.mkdir(parents=True, exist_ok=True)
    for suffix, title, script in [('','Object SDF Studio','launch.sh'),('-trainer','BundleSDF Training','launch_trainer.sh')]:
        target=folder/f'object-sdf-studio{suffix}.desktop'
        # Desktop Exec has its own quoting rules, independent of a shell.
        executable=str(ROOT/script).replace('\\','\\\\').replace('"','\\"').replace('`','\\`').replace('$','\\$').replace('%','%%')
        target.write_text(f'[Desktop Entry]\nType=Application\nName={title}\nExec="{executable}"\nIcon=camera-video\nTerminal=false\nCategories=Science;Graphics;\n')
        target.chmod(0o755)
        print(target)

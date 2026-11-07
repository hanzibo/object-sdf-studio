"""Locate final textured models without walking RGB-D frames or training caches."""
import json
from pathlib import Path
from config import ROOT, BUNDLE_PROJECT

BUNDLE_OUTPUTS = BUNDLE_PROJECT/'outputs'


def read_settings(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def mesh_browser_locations(current='', root=ROOT, bundle_outputs=BUNDLE_OUTPUTS):
    """Return a suggested file/directory and existing output-root shortcuts.

    Only final textured_mesh.obj files are auto-suggested. Intermediate normalized
    meshes and test/evidence artifacts are deliberately outside the search.
    """
    root = Path(root)
    roots = [root/'training_runs', root/'output', Path(bundle_outputs)]
    state = read_settings(root/'runtime/trainer_state.json')
    custom = state.get('output_root')
    if isinstance(custom, str) and custom.strip():
        roots.insert(0, Path(custom).expanduser())
    extras = read_settings(root/'runtime/history_roots.json').get('roots', [])
    if isinstance(extras, list):
        roots.extend(Path(p).expanduser() for p in extras if isinstance(p, str) and p.strip())

    shortcuts = []
    for path in roots:
        try:
            if path.is_dir() and path.resolve() not in shortcuts:
                shortcuts.append(path.resolve())
        except OSError:
            continue
    candidates = set()

    def add_models(folder):
        for relative in ['textured_mesh.obj', 'reconstruction/textured_mesh.obj',
                         'texture_recovery/textured_mesh.obj']:
            path = folder/relative
            try:
                if path.is_file() and path.stat().st_size:
                    candidates.add(path.resolve())
            except OSError:
                continue

    for folder in shortcuts:
        add_models(folder)
        try:
            for child in folder.iterdir():
                if child.is_dir() and not child.is_symlink():
                    add_models(child)
        except OSError:
            continue
    for folder in root.glob('recon_*'):
        if folder.is_dir():
            add_models(folder)
    # Files may disappear while history cleanup runs; ignore vanished entries.
    dated = []
    for path in candidates:
        try:
            dated.append((path.stat().st_mtime_ns, str(path), path))
        except OSError:
            continue
    models = [item[2] for item in sorted(dated, reverse=True)]
    selected = Path(current.strip()).expanduser() if current.strip() else None
    if selected is not None and (selected.is_file() or selected.is_dir()):
        initial = selected.resolve()
    elif models:
        initial = models[0]
    else:
        initial = shortcuts[0] if shortcuts else root
    # Recent model folders are also available in the location-history dropdown.
    return initial, shortcuts, [p.parent for p in models]

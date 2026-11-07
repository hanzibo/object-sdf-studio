"""Qt workspace transitions with a deterministic subprocess camera substitute.

No real camera, CUDA inference, or user files are used or modified.
"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import app as gui
from data_ops import stamp
from PySide6.QtGui import QImage,QColor
from PySide6.QtWidgets import QScrollArea

application=gui.QApplication([])
window=gui.Studio();window.auto_live=lambda:None
window.resize(1440,1000);window.show()
evidence=gui.ROOT/'evidence'/('workspace_reset_'+stamp());evidence.mkdir(parents=True)

def wait(condition):
    deadline=time.monotonic()+15
    while not condition():
        application.processEvents();time.sleep(.01)
        assert time.monotonic()<deadline,window.log.toPlainText()

def focused_click(button):
    # Mouse clicks focus buttons; QPushButton.click() alone does not reproduce
    # Qt's focus handoff when that button is subsequently disabled.
    button.setFocus();application.processEvents()
    sidebar=window.findChild(QScrollArea);sidebar.verticalScrollBar().setValue(0)
    application.processEvents();button.click()

def stable_sidebar():
    application.processEvents()
    assert window.findChild(QScrollArea).verticalScrollBar().value()==0
    assert not window.depth.hasFocus() and not window.depth.lineEdit().hasSelectedText()

def clean():
    assert window.dataset is None and not window.files and not window.prompts and not window.plan
    assert window.segmentlist.count()==0 and not window.segmented.isChecked()
    assert not window.canvas.points and not window.last_export
    assert window.slider.maximum()==0 and window.progress.value()==0
    assert not window.segment.isEnabled() and not window.export.isEnabled()
    assert not window.buildplan.isEnabled() and not window.overlay.isEnabled()

with tempfile.TemporaryDirectory(prefix='studio-workspace-') as temp:
    root=Path(temp);dataset=root/'dataset'
    (dataset/'rgb').mkdir(parents=True);(dataset/'metadata').mkdir();(dataset/'depth').mkdir()
    for i in range(3):
        image=QImage(640,480,QImage.Format_RGB32);image.fill(QColor(40+i*30,100,160))
        image.save(str(dataset/'rgb'/f'{i:03d}.png'))
    (dataset/'cam_K.txt').write_text('400 0 320\n0 400 240\n0 0 1\n')
    (dataset/'metadata/prompts.json').write_text(json.dumps({'000.png':[[200,200,1]]}))
    (dataset/'metadata/segmentation_plan.json').write_text(json.dumps(dict(enabled=True,interval=2,starts=[0,2])))
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in dataset.rglob('*') if p.is_file()}
    def loaded():
        window.load_dataset(dataset)
        assert window.segmentlist.count()==2 and window.canvas.points
        assert window.segment.isEnabled()
        window.last_export=[str(root/'old_export')];window.progress.setValue(90)
    loaded();window.pose_panel.set_dataset(dataset)
    window.clearbutton.click();clean()
    assert window.canvas.pix.isNull() and '已清空' in window.status.text()
    assert window.pose_panel.dataset==dataset # only this page is unloaded
    loaded() # reload retained prompt/plan metadata
    # Use actual QProcess IO lifecycle, with predictable image events and delayed stop.
    template=dataset/'rgb/000.png'
    script='''import json,sys,time,shutil
from pathlib import Path
job=sys.argv[1];target=Path(sys.argv[2])
def event(**kw):print('EVENT '+json.dumps(kw),flush=True)
if job=='live':
    event(type='live',image=TEMPLATE,capture_fps=30)
    for line in sys.stdin:
        if line.strip()=='stop':
            time.sleep(.1);event(type='live',image=TEMPLATE,capture_fps=30)
            event(type='done',live_preview=True);break
elif job in ['capture','import']:
    shutil.copytree(SOURCE,target)
    event(type='done',dataset=str(target))
elif job=='segment':
    event(type='done',dataset=str(target),preview=True)
else:time.sleep(.3);event(type='done')
'''.replace('TEMPLATE',repr(str(template))).replace('SOURCE',repr(str(dataset)))
    (root/'worker.py').write_text(script)
    real_root,real_python,real_sam=gui.ROOT,gui.CAMERA_PYTHON,gui.SAM_PYTHON
    gui.ROOT=root;gui.CAMERA_PYTHON=Path(sys.executable);gui.SAM_PYTHON=Path(sys.executable)
    try:
        focused_click(window.livebutton);wait(lambda:'实时彩色' in window.frameinfo.text());clean();stable_sidebar()
        assert not window.canvas.pix.isNull() and window.clearbutton.isEnabled()
        window.grab().save(str(evidence/'live_clean.png'))
        window.clearbutton.click();assert window.clear_pending
        wait(lambda:window.proc is None);clean();assert window.canvas.pix.isNull()
        # Loading a dataset while live must survive late events from the stopping preview.
        window.livebutton.click();wait(lambda:'实时彩色' in window.frameinfo.text())
        window.load_dataset(dataset);wait(lambda:window.proc is None)
        assert window.dataset==dataset and window.segmentlist.count()==2 and window.canvas.points
        assert window.segment.isEnabled()
        # Existing segmentation preview flag survives done->load->finished.
        window.run_job('segment',dataset,['--preview']);wait(lambda:window.proc is None)
        assert window.done and window.preview_active
        # Live->capture queued handoff starts clean, then loads the new recording.
        window.livebutton.click();wait(lambda:'实时彩色' in window.frameinfo.text())
        window.root.setText(str(root));window.name.setText('new_recording');focused_click(window.record)
        stable_sidebar()
        wait(lambda:window.proc is None)
        assert window.dataset==root/'new_recording' and window.segmentlist.count()==2
        stable_sidebar()
        # Non-preview work cannot be unloaded until it finishes.
        window.run_job('validate',window.dataset);assert not window.clearbutton.isEnabled()
        window.clear_workspace();assert window.dataset==root/'new_recording'
        wait(lambda:window.proc is None)
        window.clearbutton.click();clean()
        window.grab().save(str(evidence/'empty_workspace.png'))
    finally:
        gui.ROOT=real_root;gui.CAMERA_PYTHON=real_python;gui.SAM_PYTHON=real_sam
    after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in dataset.rglob('*') if p.is_file()}
    assert before==after
window.close();application.processEvents()
report=dict(clear_button=True,auto_unload_for_live=True,late_preview_events_ignored=True,
    preview_to_capture=True,segment_preview_state_preserved=True,busy_task_guard=True,
    saved_files_unchanged=True,other_tab_unchanged=True,focus_and_scroll_stable=True,real_camera=False,real_inference=False)
(evidence/'report.json').write_text(json.dumps(report,indent=2))
print('PASS workspace transitions; evidence:',evidence)

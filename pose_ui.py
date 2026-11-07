"""Qt Widgets workflow for SAM3 initialization and FoundationPose offline tracking."""
import json
import time
from pathlib import Path
from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QWidget, QLabel, QPushButton, QLineEdit, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGroupBox, QSlider, QSpinBox, QProgressBar, QPlainTextEdit,
    QScrollArea, QSplitter)
from config import ROOT, RUNTIME, SAM_PYTHON, POSE_PYTHON, POSE_PROJECT
from data_ops import atomic_json, rgb_files, stamp
from pose_data import validate_init, confirm_init
from mesh_browser import mesh_browser_locations


class PosePanel(QWidget):
    activity = Signal(bool)

    def __init__(self, studio, canvas_factory):
        super().__init__()
        self.studio = studio
        self.dataset = None
        self.files = []
        self.prompts = {}
        self.init_metadata = None
        self.proc = None
        self.job = ''
        self.buffer = ''
        self.failed = False
        self.done = False
        self.paused = False
        self.reinit_pending = False
        self.current_frame = 0
        self.run_dir = None
        self.parent_run = None
        self.display_time = None
        self.last_image = None
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('已有模型 → 新录制 RGB-D → 当前起始帧 SAM3 点选 → FoundationPose 跟踪'))
        body = QHBoxLayout();layout.addLayout(body, 1)
        scroll = QScrollArea();scroll.setWidgetResizable(True);scroll.setMinimumWidth(335);scroll.setMaximumWidth(400)
        content = QWidget();left = QVBoxLayout(content);scroll.setWidget(content);body.addWidget(scroll)
        right = QVBoxLayout();body.addLayout(right,1)
        def group(title):
            box = QGroupBox(title);v = QVBoxLayout(box);left.addWidget(box);return v
        g = group('01 已有物体模型')
        try: saved = json.loads((RUNTIME/'pose_settings.json').read_text())
        except (OSError,ValueError): saved = {}
        self.mesh = QLineEdit(saved.get('mesh',''))
        self.mesh.setPlaceholderText('选择米制 textured_mesh.obj');g.addWidget(self.mesh)
        self.mesh_button = QPushButton('选择带纹理 OBJ 模型…');self.mesh_button.clicked.connect(self.choose_mesh);g.addWidget(self.mesh_button)
        self.mesh_button.setToolTip('优先定位当前/上次使用的有效模型，否则定位最近生成的 textured_mesh.obj。\n左侧可直达训练结果、自定义输出及旧 demo 目录。')
        self.mesh.editingFinished.connect(self.save_settings)
        self.mesh.textChanged.connect(lambda _:self.update_controls() if hasattr(self,'start') else None)
        note=QLabel('保留 OBJ 同目录的 MTL 与贴图。\n新序列必须是建模时的同一个物体。');note.setWordWrap(True);g.addWidget(note)
        g = group('02 新序列 / 数据目录')
        self.open_button = QPushButton('打开 RGB-D 数据集…');self.open_button.clicked.connect(self.choose_dataset);g.addWidget(self.open_button)
        self.use_capture = QPushButton('载入采集页的当前数据');self.use_capture.clicked.connect(self.from_studio);g.addWidget(self.use_capture)
        self.capture_button = QPushButton('采集新序列（切换到采集页）');self.capture_button.clicked.connect(lambda:studio.tabs.setCurrentIndex(0));g.addWidget(self.capture_button)
        self.location = QLabel('尚未载入数据');self.location.setWordWrap(True);g.addWidget(self.location)
        g = group('03 初始化 / 重新定位')
        note=QLabel('拖动到清晰帧，左键添加前景点，右键添加背景点。\n该帧作为起点，后续不要求逐帧掩码。');note.setWordWrap(True);g.addWidget(note)
        self.undo = QPushButton('撤销当前帧最后一个点');self.undo.clicked.connect(self.undo_point);g.addWidget(self.undo)
        self.segment = QPushButton('SAM3 分割当前起始帧');self.segment.clicked.connect(self.segment_frame);g.addWidget(self.segment)
        self.confirm = QPushButton('确认当前初始化掩码');self.confirm.clicked.connect(self.confirm_mask);g.addWidget(self.confirm)
        self.init_status = QLabel('初始化掩码：未确认');self.init_status.setWordWrap(True);g.addWidget(self.init_status)
        g = group('04 离线跟踪')
        self.limit = QSpinBox();self.limit.setRange(0,1000000);self.limit.setSpecialValueText('从起始帧处理到结尾');self.limit.setSuffix(' 帧');g.addWidget(self.limit)
        self.start = QPushButton('▶ 开始 FoundationPose 跟踪');self.start.clicked.connect(self.start_tracking);g.addWidget(self.start)
        self.pause = QPushButton('暂停');self.pause.clicked.connect(self.toggle_pause);g.addWidget(self.pause)
        self.reinitialize = QPushButton('在当前帧重新点选目标');self.reinitialize.clicked.connect(self.prepare_reinit);g.addWidget(self.reinitialize)
        self.stop = QPushButton('■ 安全停止并保存结果');self.stop.clicked.connect(self.stop_job);g.addWidget(self.stop)
        self.results = QPushButton('打开最近结果目录');self.results.clicked.connect(self.open_results);g.addWidget(self.results)
        self.video = QPushButton('播放最近跟踪视频');self.video.clicked.connect(self.open_video);g.addWidget(self.video)
        left.addStretch()
        self.canvas = canvas_factory();self.canvas.clicked.connect(self.add_point);right.addWidget(self.canvas,1)
        self.slider = QSlider(Qt.Horizontal);self.slider.valueChanged.connect(self.show_frame);right.addWidget(self.slider)
        self.frameinfo = QLabel('帧 0 / 0');right.addWidget(self.frameinfo)
        self.performance = QLabel('推理耗时：—    预览刷新：—');right.addWidget(self.performance)
        controls=QHBoxLayout()
        for button in [self.start,self.pause,self.reinitialize,self.stop]:
            g.removeWidget(button);controls.addWidget(button)
        layout.addLayout(controls)
        self.status = QLabel('选择同一物体的模型和新 RGB-D 序列。无需完整分割或重新建模。');self.status.setWordWrap(True);layout.addWidget(self.status)
        self.progress = QProgressBar();layout.addWidget(self.progress)
        self.log = QPlainTextEdit();self.log.setReadOnly(True);self.log.setMaximumHeight(130);self.log.setMaximumBlockCount(500);layout.addWidget(self.log)
        self.update_controls()

    def message(self, text):
        self.status.setText(text);self.log.appendPlainText(text)

    def save_settings(self):
        atomic_json(RUNTIME/'pose_settings.json',dict(mesh=self.mesh.text()))
        self.update_controls()

    def update_controls(self):
        busy = self.proc is not None
        cleaning=bool(getattr(getattr(self.studio,'history_panel',None),'deleting',False))
        other = self.studio.proc is not None or cleaning
        for widget in [self.mesh,self.mesh_button,self.open_button,self.use_capture,self.capture_button,
                       self.slider,self.undo,self.limit]:widget.setEnabled(not busy and not cleaning)
        self.segment.setEnabled(not busy and not other and bool(self.files))
        confirmed = False
        if self.init_metadata and self.files:
            try:
                validate_init(self.init_metadata,self.dataset,self.slider.value(),self.points())
                confirmed = True
            except (ValueError,OSError,KeyError): pass
        self.confirm.setEnabled(not busy and self.init_metadata is not None and not confirmed)
        self.start.setEnabled(not busy and not other and confirmed and Path(self.mesh.text()).is_file())
        self.pause.setEnabled(busy and self.job=='tracking' and not self.reinit_pending)
        self.pause.setText('继续' if self.paused else '暂停')
        self.reinitialize.setEnabled(busy and self.job=='tracking' and self.paused)
        self.stop.setEnabled(busy)
        self.results.setEnabled(self.run_dir is not None)
        self.video.setEnabled(self.run_dir is not None and (self.run_dir/'tracking.mp4').is_file())

    def choose_mesh(self):
        initial,shortcuts,recent = mesh_browser_locations(self.mesh.text())
        dialog = QFileDialog(self,'选择米制带纹理模型')
        dialog.setOption(QFileDialog.DontUseNativeDialog,True)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter('OBJ 模型 (*.obj *.OBJ)')
        dialog.setViewMode(QFileDialog.Detail)
        for label,text in [(QFileDialog.LookIn,'查找位置：'),(QFileDialog.FileName,'模型文件：'),
                           (QFileDialog.FileType,'文件类型：'),(QFileDialog.Accept,'选择模型'),
                           (QFileDialog.Reject,'取消')]:
            dialog.setLabelText(label,text)
        dialog.resize(1040,680)
        splitter = dialog.findChild(QSplitter)
        if splitter:splitter.setSizes([220,800])
        dialog.setSidebarUrls([QUrl.fromLocalFile(str(p)) for p in dict.fromkeys([Path.home(),*shortcuts])])
        dialog.setHistory([str(p) for p in dict.fromkeys(recent)][:20])
        dialog.setDirectory(str(initial if initial.is_dir() else initial.parent))
        if initial.is_file():dialog.selectFile(str(initial))
        accepted = dialog.exec()
        paths = dialog.selectedFiles() if accepted else []
        dialog.deleteLater()
        if paths:self.mesh.setText(paths[0]);self.save_settings();self.update_controls()

    def choose_dataset(self):
        path = QFileDialog.getExistingDirectory(self,'选择包含 rgb、depth、cam_K.txt 的目录',str(ROOT/'datasets'))
        if path:
            try:self.set_dataset(path)
            except Exception as exc:self.message(str(exc))

    def from_studio(self):
        if self.studio.dataset:
            try:self.set_dataset(self.studio.dataset)
            except Exception as exc:self.message(str(exc))
        else:self.message('请先在采集页录制或打开数据集')

    def set_dataset(self, dataset):
        if self.proc:raise ValueError('请先停止当前跟踪页任务')
        root=Path(dataset).resolve();files=rgb_files(root)
        if not (root/'cam_K.txt').is_file() or not (root/'depth').is_dir():
            raise ValueError('数据集需要 depth 目录和 cam_K.txt')
        self.dataset=root;self.files=files;self.init_metadata=None;self.parent_run=None
        point_file=root/'metadata/pose_prompts.json'
        self.prompts=json.loads(point_file.read_text()) if point_file.exists() else {}
        self.location.setText(str(root));self.slider.setRange(0,len(files)-1);self.slider.setValue(0)
        self.show_frame(0);self.init_status.setText('初始化掩码：未确认')
        self.message(f'已载入 {len(files)} 帧 RGB-D，请在起始帧点选目标');self.update_controls()

    def points(self):
        return self.prompts.get(self.files[self.slider.value()].name,[]) if self.files else []

    def show_frame(self, i):
        if not self.files or self.proc:return
        self.current_frame=i;self.canvas.load(self.files[i],self.points())
        self.frameinfo.setText(f'起始帧 {i+1} / {len(self.files)} · 提示点 {len(self.points())}')
        self.update_controls()

    def save_points(self):
        atomic_json(self.dataset/'metadata/pose_prompts.json',self.prompts)
        self.init_status.setText('提示点已修改，请重新分割并确认')
        self.show_frame(self.slider.value())

    def add_point(self,x,y,label):
        if self.proc or not self.files:return
        self.prompts.setdefault(self.files[self.slider.value()].name,[]).append([round(x,2),round(y,2),label])
        self.save_points()

    def undo_point(self):
        if self.proc or not self.files:return
        if self.points():self.points().pop();self.save_points()

    def segment_frame(self):
        if not self.files:return
        if not any(p[2]==1 for p in self.points()):self.message('请先用左键在目标内部添加前景点');return
        atomic_json(self.dataset/'metadata/pose_prompts.json',self.prompts)
        self.init_metadata=None
        self.launch('initialization',SAM_PYTHON,[str(ROOT/'worker.py'),'pose_init',str(self.dataset),
                    '--frame-index',str(self.slider.value()),'--prompts-file',str(self.dataset/'metadata/pose_prompts.json')])

    def confirm_mask(self):
        try:
            if not self.init_metadata:raise ValueError('请先运行 SAM3 初始化分割')
            confirm_init(self.init_metadata,self.dataset,self.slider.value(),self.points())
            self.init_status.setText(f'初始化掩码已确认 · 第 {self.slider.value()+1} 帧')
            self.message('掩码已确认，SAM3 已退出，可以开始 FoundationPose 跟踪')
        except Exception as exc:self.message(str(exc))
        self.update_controls()

    def start_tracking(self):
        try:
            validate_init(self.init_metadata,self.dataset,self.slider.value(),self.points())
            mesh=Path(self.mesh.text()).expanduser().resolve()
            if not mesh.is_file():raise ValueError('请选择有效模型文件')
            self.run_dir=ROOT/'foundationpose_runs'/('run_'+stamp())
            argv=[str(ROOT/'scripts/run_foundationpose_offline.py'),'--dataset',str(self.dataset),
                  '--mesh',str(mesh),'--output',str(self.run_dir),'--frames',str(self.limit.value()),
                  '--start-frame',str(self.slider.value()),'--init-metadata',str(self.init_metadata),
                  '--events','--controls']
            if self.parent_run:argv+=['--parent-run',str(self.parent_run)]
            self.save_settings();self.launch('tracking',POSE_PYTHON,argv)
        except Exception as exc:self.message(str(exc))

    def launch(self, job, python, argv):
        if self.proc:return
        if getattr(getattr(self.studio,'history_panel',None),'deleting',False):self.message('历史数据正在清理，请稍候');return
        if self.studio.proc:self.message('请先停止采集页的相机预览、录制或分割任务，再启动跟踪页任务');return
        self.job=job;self.failed=False;self.done=False;self.buffer='';self.paused=False;self.reinit_pending=False
        self.display_time=None;self.last_image=None
        p=QProcess(self);self.proc=p;p.setProcessChannelMode(QProcess.MergedChannels)
        env=QProcessEnvironment.systemEnvironment()
        for key in ['PYTHONPATH','QT_QPA_PLATFORM_PLUGIN_PATH','QT_PLUGIN_PATH']:env.remove(key)
        env.insert('PYTHONUNBUFFERED','1');p.setProcessEnvironment(env);p.setWorkingDirectory(str(ROOT))
        p.readyReadStandardOutput.connect(self.read_output);p.finished.connect(self.finished)
        p.errorOccurred.connect(lambda error:self.process_error(p,error))
        p.start(str(python),argv);self.message('启动 '+('SAM3 初始化分割' if job=='initialization' else 'FoundationPose 跟踪'))
        self.progress.setRange(0,0);self.activity.emit(True);self.update_controls()

    def process_error(self,p,error):
        if error==QProcess.FailedToStart and self.proc is p:
            self.failed=True;self.message(p.errorString());self.proc=None;p.deleteLater()
            self.activity.emit(False);self.progress.setRange(0,100);self.update_controls()

    def read_output(self):
        if not self.proc:return
        chunk=bytes(self.proc.readAllStandardOutput()).decode('utf-8',errors='replace')
        if self.job=='tracking' and self.run_dir and self.run_dir.exists():
            with (self.run_dir/'worker.log').open('a') as handle:handle.write(chunk)
        self.buffer+=chunk
        while '\n' in self.buffer:
            line,self.buffer=self.buffer.split('\n',1)
            if line.startswith('EVENT '):
                try:self.handle(json.loads(line[6:]))
                except Exception as exc:self.message('事件处理错误：'+str(exc))
            elif line:self.log.appendPlainText(line)

    def handle(self,value):
        kind=value['type']
        if value.get('message'):self.message(value['message'])
        if kind=='error':self.failed=True
        if value.get('image'):
            self.last_image=value['image'];self.canvas.load(self.last_image)
        if kind=='progress' and 'frame_index' in value:
            self.current_frame=value['frame_index']
            total=value.get('total',0);self.progress.setRange(0,total);self.progress.setValue(value.get('index',0))
            self.frameinfo.setText(f'原序列帧 {self.current_frame+1} / {len(self.files)}')
            now=time.monotonic();fps=1/(now-self.display_time) if self.display_time else 0.;self.display_time=now
            self.performance.setText(f'推理耗时：{value["inference_ms"]:.1f} ms    预览刷新：{fps:.1f} FPS')
        if kind=='paused':self.paused=True;self.update_controls()
        if kind=='resumed':self.paused=False;self.update_controls()
        if kind=='done':
            self.done=True
            if value.get('init_metadata'):
                self.init_metadata=Path(value['init_metadata'])
                self.init_status.setText('SAM3 预览完成，请检查后确认掩码')

    def finished(self,code,status):
        if not self.proc:return
        self.read_output();self.proc.deleteLater();self.proc=None;self.paused=False;self.activity.emit(False)
        self.progress.setRange(0,100);self.progress.setValue(100 if self.done else 0)
        if code or not self.done:
            self.failed=True;self.message(f'任务退出（{code}），请查看日志；已输出结果保留')
        if self.reinit_pending:
            self.reinit_pending=False;self.parent_run=self.run_dir;self.init_metadata=None
            self.slider.setValue(self.current_frame);self.show_frame(self.current_frame)
            self.init_status.setText('重新定位：请在当前帧点选并分割')
            self.message('跟踪进程已退出并释放 GPU。请在当前帧重新点选、分割、确认，再开始跟踪。')
        self.update_controls()

    def toggle_pause(self):
        if self.proc and self.job=='tracking':
            self.proc.write(b'resume\n' if self.paused else b'pause\n')
            self.pause.setEnabled(False)

    def prepare_reinit(self):
        if not self.proc or not self.paused:return
        self.reinit_pending=True;self.proc.write(b'stop\n');self.stop.setEnabled(False)
        self.message('保存当前结果并退出跟踪，随后可重新点选…');self.update_controls()

    def stop_job(self):
        if self.proc:
            self.proc.write(b'stop\n');self.stop.setEnabled(False);self.pause.setEnabled(False)
            self.message('正在安全停止，等待当前计算和视频保存完成…')

    def open_results(self):
        if self.run_dir and self.run_dir.exists():QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.run_dir)))

    def open_video(self):
        if self.run_dir and (self.run_dir/'tracking.mp4').is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.run_dir/'tracking.mp4')))

#!/usr/bin/env python3
"""Native Qt workflow; SDK and CUDA run in isolated QProcess workers."""
import sys,os,json,time
from collections import deque
from pathlib import Path
import importlib.util
try:
    _SYSTEM_QT = importlib.util.find_spec('PySide6.QtCore') is None
except ModuleNotFoundError:
    _SYSTEM_QT = True
if _SYSTEM_QT:
    sys.modules.pop('PySide6', None)
    sys.path.insert(0,'/usr/lib/python3/dist-packages')
from PySide6.QtCore import Qt,QProcess,QTimer,QUrl,Signal,QPointF
from PySide6.QtGui import QPixmap,QImage,QPainter,QColor,QPen,QDesktopServices
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QLabel,QPushButton,QVBoxLayout,QHBoxLayout,QGroupBox,QLineEdit,QFileDialog,QPlainTextEdit,QSlider,QSpinBox,QDoubleSpinBox,QProgressBar,QMessageBox,QSplitter,QRadioButton,QCheckBox,QScrollArea,QListWidget,QComboBox,QTabWidget)
if _SYSTEM_QT:sys.path.remove('/usr/lib/python3/dist-packages')
from config import *
from segment_plan import read_plan,ranges,cached_result,write_json,PLAN_FILE

def stamp():return time.strftime('%Y%m%d_%H%M%S')+f'_{time.time_ns()%1000000:06d}'
class Canvas(QWidget):
    clicked=Signal(float,float,int)
    def __init__(self):
        super().__init__();self.pix=QPixmap();self.points=[];self.mode=1;self.setMinimumSize(640,360);self.setCursor(Qt.CrossCursor);self.setFocusPolicy(Qt.ClickFocus)
    def load(self,path,points=None):
        self.pix=QPixmap(str(path));self.points=points or [];self.update()
    def rect_image(self):
        if self.pix.isNull():return None
        size=self.pix.size();size.scale(self.size(),Qt.KeepAspectRatio)
        from PySide6.QtCore import QRectF
        return QRectF((self.width()-size.width())/2,(self.height()-size.height())/2,size.width(),size.height())
    def paintEvent(self,e):
        p=QPainter(self);p.fillRect(self.rect(),QColor('#101824'));r=self.rect_image()
        if r is None:
            p.setPen(QColor('#9eafc2'));p.drawText(self.rect(),Qt.AlignCenter,'录制 RGB-D 或导入 RGB 目录后，在这里选择目标');return
        p.drawPixmap(r,self.pix,self.pix.rect())
        for x,y,label in self.points:
            pt=QPointF(r.x()+x/self.pix.width()*r.width(),r.y()+y/self.pix.height()*r.height())
            p.setPen(QPen(QColor('#ffffff'),2));p.setBrush(QColor('#39e58c' if label else '#ff536a'));p.drawEllipse(pt,6,6)
            p.setPen(QColor('#152030'));p.drawText(pt+QPointF(-4,4),'+' if label else '−')
    def mousePressEvent(self,e):
        r=self.rect_image()
        if r and r.contains(e.position()):
            x=(e.position().x()-r.x())/r.width()*self.pix.width();y=(e.position().y()-r.y())/r.height()*self.pix.height()
            self.clicked.emit(x,y,0 if e.button()==Qt.RightButton else self.mode)
class Studio(QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle('BundleSDF Studio · Gemini 336L × SAM3');self.resize(1380,920)
        self.dataset=None;self.files=[];self.prompts={};self.proc=None;self.buffer='';self.job='';self.failed=False;self.done=False;self.closing=False;self.pending_job=None;self.live_stopping=False;self.clear_pending=False;self.plan={}
        base=QWidget();self.setCentralWidget(base);layout=QVBoxLayout(base)
        title=QLabel('BundleSDF Studio');title.setStyleSheet('font-size:26px;font-weight:700;color:#65dcc9');layout.addWidget(title)
        layout.addWidget(QLabel('① 同步采集 RGB-D + 内参    →    ② 首帧点选与 SAM3 跟踪    →    ③ 验收并导出训练片段'))
        middle=QHBoxLayout();layout.addLayout(middle,1)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setMinimumWidth(365);scroll.setMaximumWidth(405);panel=QWidget();left=QVBoxLayout(panel);scroll.setWidget(panel);middle.addWidget(scroll);right=QVBoxLayout();middle.addLayout(right,1)
        def group(name):
            box=QGroupBox(name);v=QVBoxLayout(box);left.addWidget(box);return v
        g=group('01 采集 / 数据目录')
        self.root=QLineEdit(str(DATASETS));g.addWidget(QLabel('数据保存根目录'));g.addWidget(self.root)
        self.name=QLineEdit('object_'+time.strftime('%m%d_%H%M%S'));g.addWidget(self.name)
        self.maxframes=QSpinBox();self.maxframes.setRange(0,100000);self.maxframes.setSpecialValueText('手动停止');self.maxframes.setSuffix(' 帧');g.addWidget(self.maxframes)
        g.addWidget(QLabel('预览目标刷新率'))
        self.previewrate=QComboBox()
        for fps in [10,20,30]:self.previewrate.addItem(f'{fps} FPS',fps)
        try:stored=json.loads((RUNTIME/'settings.json').read_text()).get('preview_fps',10)
        except (OSError,ValueError):stored=10
        self.previewrate.setCurrentIndex(max(0,self.previewrate.findData(stored)));self.previewrate.currentIndexChanged.connect(self.preview_rate_changed);g.addWidget(self.previewrate)
        self.livebutton=QPushButton('相机实时预览（不录制）');self.livebutton.clicked.connect(self.start_live);g.addWidget(self.livebutton)
        self.record=QPushButton('● 开始录制 RGB + Depth');self.record.clicked.connect(self.start_capture);g.addWidget(self.record)
        self.stop=QPushButton('■ 停止录制 / 取消分割');self.stop.clicked.connect(self.cancel);self.stop.setEnabled(False);g.addWidget(self.stop)
        self.importbutton=QPushButton('选择 RGB 目录 → 创建工作副本');self.importbutton.clicked.connect(self.import_folder);g.addWidget(self.importbutton)
        self.openbutton=QPushButton('打开已有 Studio 数据集');self.openbutton.clicked.connect(self.open_dataset);g.addWidget(self.openbutton)
        self.clearbutton=QPushButton('清空工作区（不删除文件）');self.clearbutton.setToolTip('卸载当前数据集、提示点和分段状态；已保存的数据与其他页面不受影响')
        self.clearbutton.clicked.connect(self.clear_workspace);g.addWidget(self.clearbutton)
        g=group('02 目标点选与分割')
        g.addWidget(QLabel('绿点：物体前景；红点：背景\n右键始终添加背景点。先回到首帧点选。'))
        row=QHBoxLayout();self.positive=QRadioButton('前景 +');self.negative=QRadioButton('背景 −');self.positive.setChecked(True);row.addWidget(self.positive);row.addWidget(self.negative);g.addLayout(row)
        self.negative.toggled.connect(lambda checked:setattr(self.canvas,'mode',0 if checked else 1))
        self.undo=QPushButton('撤销当前帧最后一个点');self.undo.clicked.connect(self.undo_point);g.addWidget(self.undo)
        self.preview=QPushButton('预览首帧分割');self.preview.clicked.connect(lambda:self.start_segment(True));g.addWidget(self.preview)
        self.segment=QPushButton('SAM3 完整逐帧分割');self.segment.clicked.connect(lambda:self.start_segment(False));g.addWidget(self.segment)
        g.addWidget(QLabel('后续帧可补点；分段模式可单段重跑。\n不会跳帧；中断不会覆盖完整掩码。'))
        g=group('分段分割 · 断点管理')
        self.segmented=QCheckBox('启用分段分割（各段独立跟踪）');g.addWidget(self.segmented)
        self.interval=QSpinBox();self.interval.setRange(1,100000);self.interval.setValue(100);self.interval.setSuffix(' 帧 / 段');g.addWidget(self.interval)
        self.buildplan=QPushButton('按间隔生成断点');self.buildplan.clicked.connect(self.build_plan);g.addWidget(self.buildplan)
        self.segmentlist=QListWidget();self.segmentlist.setMaximumHeight(145);self.segmentlist.currentRowChanged.connect(self.select_segment);g.addWidget(self.segmentlist)
        nav=QHBoxLayout();self.prevsegment=QPushButton('上一断点');self.nextsegment=QPushButton('下一断点');nav.addWidget(self.prevsegment);nav.addWidget(self.nextsegment);g.addLayout(nav)
        self.prevsegment.clicked.connect(lambda:self.move_segment(-1));self.nextsegment.clicked.connect(lambda:self.move_segment(1))
        self.addbreak=QPushButton('将当前帧插入为新断点');self.addbreak.clicked.connect(self.add_breakpoint);g.addWidget(self.addbreak)
        self.previewsegment=QPushButton('预览所选段起始帧');self.previewsegment.clicked.connect(lambda:self.start_segment_job(True));g.addWidget(self.previewsegment)
        self.runsegment=QPushButton('重跑所选段');self.runsegment.clicked.connect(lambda:self.start_segment_job(False));g.addWidget(self.runsegment)
        self.resume=QPushButton('处理 / 继续全部待完成段');self.resume.clicked.connect(lambda:self.start_segment(False));g.addWidget(self.resume)
        self.segmented.toggled.connect(self.toggle_segmented)
        g.addWidget(QLabel('各段应选择同一个物体。\n完整段可复用；全部完成才合并 masks。'))
        g=group('03 验收与导出')
        self.depth=QDoubleSpinBox();self.depth.setRange(.2,5);self.depth.setValue(1);self.depth.setSingleStep(.1);self.depth.setSuffix(' m');g.addWidget(QLabel('有效深度上限（BundleSDF 默认 1 m）'));g.addWidget(self.depth)
        self.export=QPushButton('检查并导出 BundleSDF 数据集');self.export.clicked.connect(self.start_export);g.addWidget(self.export)
        exported=QPushButton('打开最近导出目录');exported.clicked.connect(self.show_export);g.addWidget(exported)
        folder=QPushButton('打开当前数据文件夹');folder.clicked.connect(self.show_folder);g.addWidget(folder)
        left.addStretch()
        self.location=QLabel('尚未载入数据');self.location.setWordWrap(True);right.addWidget(self.location)
        self.canvas=Canvas();self.canvas.clicked.connect(self.add_point);right.addWidget(self.canvas,1)
        self.slider=QSlider(Qt.Horizontal);self.slider.valueChanged.connect(self.frame_changed);right.addWidget(self.slider)
        self.frameinfo=QLabel('帧 0 / 0');right.addWidget(self.frameinfo)
        self.fpsinfo=QLabel('采集帧率：— FPS    ·    预览刷新：— FPS');right.addWidget(self.fpsinfo);self.display_times=deque()
        self.overlay=QCheckBox('叠加已完成的掩码（绿色）');self.overlay.setChecked(True);self.overlay.toggled.connect(lambda _:self.frame_changed(self.slider.value()));right.addWidget(self.overlay)
        self.status=QLabel('就绪。建议物体距离 0.2–0.9 m，缓慢绕物体拍摄，避免遮挡与运动模糊。');self.status.setWordWrap(True);layout.addWidget(self.status)
        self.progress=QProgressBar();layout.addWidget(self.progress)
        self.log=QPlainTextEdit();self.log.setReadOnly(True);self.log.setMaximumHeight(145);self.log.setMaximumBlockCount(500);layout.addWidget(self.log)
        self.setStyleSheet('QMainWindow,QWidget {background:#172231;color:#e5edf5;font-size:14px;} QGroupBox{border:1px solid #40546b;border-radius:8px;margin-top:12px;padding:12px;min-width:275px;} QGroupBox::title{subcontrol-origin:margin;left:12px;color:#65dcc9;} QPushButton{background:#294358;border:1px solid #46667b;border-radius:5px;padding:9px;} QPushButton:hover{background:#355e73;} QPushButton:disabled{color:#75808e;background:#222d3b;} QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox,QPlainTextEdit{background:#101824;border:1px solid #40546b;padding:5px;} QProgressBar{border:1px solid #40546b;text-align:center;} QProgressBar::chunk{background:#258c82;}')
        self.setStyleSheet(self.styleSheet()+f'''
            QComboBox {{min-height:28px;padding-right:48px;}}
            QComboBox::drop-down {{subcontrol-origin:border;subcontrol-position:top right;width:42px;
                background:#294358;border-left:1px solid #5a7d94;}}
            QComboBox::drop-down:hover {{background:#3b647b;}}
            QComboBox::down-arrow {{image:url("{ROOT/'assets/chevron-down.svg'}");width:18px;height:18px;}}
        ''')
        from pose_ui import PosePanel
        self.takeCentralWidget();self.tabs=QTabWidget();self.tabs.addTab(base,'采集 / 建模数据')
        self.pose_panel=PosePanel(self,Canvas);self.tabs.addTab(self.pose_panel,'已有模型 · FoundationPose 跟踪')
        from history_ui import HistoryPanel
        self.history_panel=HistoryPanel(self);self.tabs.addTab(self.history_panel,'历史数据管理')
        self.history_panel.activity.connect(self.history_activity);self.history_panel.deleted.connect(self.history_deleted)
        self.setCentralWidget(self.tabs);self.pose_panel.activity.connect(self.pose_activity)
        self.tabs.currentChanged.connect(self.tab_changed)
        self.busy(False)
        QTimer.singleShot(300,self.auto_live)
    def pose_activity(self,active):
        self.busy(bool(self.proc) or active)
        if self.closing and not active:QTimer.singleShot(0,self.close)
    def history_activity(self,active):
        self.busy(bool(self.proc) or bool(self.pose_panel.proc) or active)
        self.pose_panel.update_controls()
    def history_deleted(self,paths):
        def removed(path):
            if not path:return False
            p=Path(path).resolve()
            return any(p==Path(root) or Path(root) in p.parents for root in paths)
        if removed(self.dataset):
            self.reset_workspace();self.frameinfo.setText('数据已从历史页删除')
        elif self.dataset:
            self.refresh_segments();self.frame_changed(self.slider.value())
        panel=self.pose_panel
        if removed(panel.dataset):
            panel.dataset=None;panel.files=[];panel.prompts={};panel.init_metadata=None
            panel.canvas.pix=QPixmap();panel.canvas.update();panel.slider.setRange(0,0)
            panel.location.setText('数据已从历史页删除');panel.frameinfo.setText('帧 0 / 0')
            panel.init_status.setText('初始化掩码：未确认')
        if removed(panel.init_metadata):panel.init_metadata=None;panel.init_status.setText('初始化掩码已删除，请重新分割')
        if removed(panel.mesh.text()):panel.mesh.clear();panel.save_settings()
        if removed(panel.run_dir):panel.run_dir=None
        if removed(panel.parent_run):panel.parent_run=None
        if getattr(self,'last_export',None):self.last_export=[p for p in self.last_export if not removed(p)]
        panel.update_controls()
    def tab_changed(self,index):
        if index in [1,2] and self.proc and self.job=='live':self.cancel()
        self.pose_panel.update_controls()
        if index==2 and self.history_panel.catalog is None:self.history_panel.scan()
    def preview_rate_changed(self,index):
        fps=self.previewrate.currentData();write_json(RUNTIME/'settings.json',dict(preview_fps=fps));self.display_times.clear()
        if self.proc and self.job in ['live','capture']:self.proc.write(f'fps {fps}\n'.encode())
    def auto_live(self):
        if not self.files and not self.proc and not self.closing and self.tabs.currentIndex()==0 and not self.pose_panel.proc:self.start_live()
    def start_live(self):
        if self.proc or self.pose_panel.proc or self.history_panel.deleting:return
        self.run_job('live',RUNTIME/('preview_'+stamp()))
    def message(self,text):self.status.setText(text);self.log.appendPlainText(text)
    def reset_workspace(self):
        """Unload capture-page state only. Never write or delete dataset files."""
        self.dataset=None;self.files=[];self.prompts={};self.plan={};self.last_export=[]
        self.preview_active=False;self.failed=False;self.done=False
        self.slider.blockSignals(True);self.slider.setRange(0,0);self.slider.setValue(0);self.slider.blockSignals(False)
        self.segmentlist.blockSignals(True);self.segmentlist.clear();self.segmentlist.blockSignals(False)
        self.segmented.blockSignals(True);self.segmented.setChecked(False);self.segmented.blockSignals(False)
        self.interval.setValue(100);self.positive.setChecked(True)
        self.canvas.points=[];self.canvas.pix=QPixmap();self.canvas.update()
        self.location.setText('尚未载入数据');self.frameinfo.setText('帧 0 / 0')
        self.display_times.clear();self.fpsinfo.setText('采集帧率：— FPS    ·    预览刷新：— FPS')
        self.progress.setRange(0,100);self.progress.setValue(0);self.log.clear()
        self.busy(bool(self.proc))
    def clear_workspace(self):
        if self.pose_panel.proc or self.history_panel.deleting or (self.proc and self.job!='live'):
            self.message('请先停止当前任务，再清空工作区');return
        self.pending_job=None
        if self.proc:
            self.clear_pending=True;self.cancel();self.busy(True);return
        self.reset_workspace();self.message('工作区已清空，已保存文件保留。可开启相机预览或打开数据集。')
    def busy(self,value):
        value=value or bool(getattr(getattr(self,'pose_panel',None),'proc',None))
        value=value or bool(getattr(getattr(self,'history_panel',None),'deleting',False))
        data_ready=not value and bool(self.files) and not (self.proc and self.job=='live')
        states=[(b,not value) for b in [self.livebutton,self.record,self.importbutton,self.openbutton,self.clearbutton]]
        states += [(b,data_ready) for b in [self.preview,self.segment,self.export,self.undo,self.positive,self.negative,self.slider,self.segmented,self.interval,self.buildplan,self.segmentlist,self.prevsegment,self.nextsegment,self.addbreak,self.previewsegment,self.runsegment,self.resume,self.overlay]]
        states.append((self.stop,bool(self.proc) and value and self.job in ['capture','segment','live']))
        # Disabling the focused control makes Qt tab forward and scroll to its
        # replacement (often the depth spin box). Move focus outside the sidebar
        # first, so task transitions preserve the user's scroll position.
        focused=QApplication.focusWidget()
        if focused and any(not enabled and (widget==focused or widget.isAncestorOf(focused)) for widget,enabled in states):
            self.canvas.setFocus(Qt.OtherFocusReason)
        for widget,enabled in states:widget.setEnabled(enabled)
    def run_job(self,job,dataset,extra=None):
        if self.history_panel.deleting:self.message('历史数据正在清理，请稍候');return
        if self.pose_panel.proc:self.message('请先停止跟踪页任务');return
        if self.proc:
            if self.job=='live':
                self.pending_job=(job,dataset,extra);self.live_stopping=True;self.busy(True);self.proc.write(b'stop\n')
            return
        if job in ['live','capture','import']:self.reset_workspace()
        self.live_stopping=False;self.display_times.clear();self.fpsinfo.setText('采集帧率：测量中    ·    预览刷新：测量中')
        self.job=job;self.failed=False;self.done=False;self.preview_active=False;self.buffer='';self.busy(True)
        p=QProcess(self);self.proc=p;p.setProcessChannelMode(QProcess.MergedChannels)
        from PySide6.QtCore import QProcessEnvironment
        env=QProcessEnvironment.systemEnvironment()
        for k in ['PYTHONPATH','QT_QPA_PLATFORM_PLUGIN_PATH','QT_PLUGIN_PATH']:env.remove(k)
        env.insert('PYTHONUNBUFFERED','1');p.setProcessEnvironment(env)
        p.readyReadStandardOutput.connect(self.read_output);p.finished.connect(self.finished);p.errorOccurred.connect(lambda e:self.process_error(p,e))
        interpreter=SAM_PYTHON if job=='segment' else CAMERA_PYTHON
        options=list(extra or [])
        if job in ['live','capture']:options+=['--preview-fps',str(self.previewrate.currentData())]
        p.start(str(interpreter),[str(ROOT/'worker.py'),job,str(dataset)]+options);self.busy(True);self.message('启动任务：'+job)
        self.pose_panel.update_controls()
        if job=='live':self.busy(False);self.stop.setEnabled(True);self.livebutton.setEnabled(False)
    def process_error(self,p,error):
        self.message('进程状态：'+p.errorString())
        if error==QProcess.FailedToStart:
            self.failed=True;self.proc=None;self.busy(False);p.deleteLater()
    def read_output(self):
        self.buffer+=bytes(self.proc.readAllStandardOutput()).decode('utf-8',errors='replace')
        while '\n' in self.buffer:
            line,self.buffer=self.buffer.split('\n',1)
            if line.startswith('EVENT '):
                try:self.handle(json.loads(line[6:]))
                except Exception as e:self.log.appendPlainText('事件处理错误：'+str(e))
            elif line:self.log.appendPlainText(line)
    def handle(self,e):
        typ=e['type']
        if e.get('message') and typ!='live':self.message(e['message'])
        if typ=='live':
            if not self.live_stopping:
                self.update_fps(e);self.status.setText(e.get('message','相机实时预览'));self.canvas.load(e['image']);self.frameinfo.setText('实时彩色画面 · 640×480 · 尚未录制')
                self.location.setText('相机预览：调整物体摆放后点击开始录制')
            return
        if typ=='progress':
            n=e.get('total',0);self.progress.setRange(0,n or 0);self.progress.setValue(e.get('index',0))
            if e.get('image'):
                self.canvas.load(e['image'])
                if self.job=='capture':self.update_fps(e)
        if typ=='segment_done':self.refresh_segments()
        if typ=='error':self.failed=True
        if typ=='done':
            self.done=True;self.preview_active=e.get('preview',False)
            if e.get('dataset'):
                frame=self.slider.value();segment=self.segmentlist.currentRow();same=self.dataset==Path(e['dataset'])
                self.load_dataset(e['dataset'])
                if self.job in ['capture','import'] and not self.pose_panel.proc:self.pose_panel.set_dataset(e['dataset'])
                if same and self.job=='segment':
                    self.segmentlist.blockSignals(True);self.segmentlist.setCurrentRow(segment);self.segmentlist.blockSignals(False);self.slider.setValue(frame);self.frame_changed(frame)
            if e.get('exported'):
                self.last_export=e['exported'];self.message('导出完成：\n'+'\n'.join(e['exported']))
    def update_fps(self,event):
        now=time.monotonic();self.display_times.append(now)
        while len(self.display_times)>2 and now-self.display_times[0]>1.:self.display_times.popleft()
        times=self.display_times;fps=(len(times)-1)/(times[-1]-times[0]) if len(times)>1 else 0.
        capture=event.get('capture_fps',0.)
        self.fpsinfo.setText(f'采集帧率：{capture:.1f} FPS    ·    预览刷新：{fps:.1f} FPS')
    def finished(self,code,status):
        self.read_output();self.proc.deleteLater();self.proc=None;self.busy(False);self.progress.setRange(0,100);self.progress.setValue(100 if self.done and self.job!='live' else 0)
        if code and not self.failed:self.message(f'任务退出，代码 {code}，请查看日志')
        self.pose_panel.update_controls()
        if self.clear_pending:
            self.clear_pending=False;self.reset_workspace()
            self.message('工作区已清空，相机已停止；已保存文件保留。')
        if self.dataset:
            self.refresh_segments()
            if not(self.done and self.preview_active):self.frame_changed(self.slider.value())
            if self.job=='live':self.location.setText(str(self.dataset))
        if self.closing:self.pending_job=None;self.close()
        elif self.pending_job:
            pending=self.pending_job;self.pending_job=None;self.run_job(*pending)
    def start_capture(self):
        name=self.name.text().strip()
        if not name or Path(name).name!=name:self.message('名称只能是单个文件夹名');return
        target=Path(self.root.text()).expanduser()/name
        if target.exists():self.message('名称已存在，请更换名称，防止覆盖');return
        self.run_job('capture',target,['--frames',str(self.maxframes.value())])
    def cancel(self):
        if self.proc:self.live_stopping=True;self.proc.write(b'stop\n');self.stop.setEnabled(False);self.message('正在安全停止并保存已接受的帧，请稍候…')
    def import_folder(self):
        folder=QFileDialog.getExistingDirectory(self,'选择 RGB 图像目录')
        if folder:self.run_job('import',Path(self.root.text()).expanduser()/('import_'+stamp()),['--source',folder])
    def open_dataset(self):
        folder=QFileDialog.getExistingDirectory(self,'选择包含 rgb / depth / cam_K.txt 的工作目录')
        if folder:
            try:self.load_dataset(folder)
            except Exception as e:self.message(str(e))
    def load_dataset(self,folder):
        if self.proc and self.job=='live':
            self.live_stopping=True;self.proc.write(b'stop\n')
        root=Path(folder);files=sorted(p for p in (root/'rgb').glob('*.png'))
        if not files:raise ValueError('工作目录中没有 RGB PNG')
        p=root/'metadata/prompts.json';prompts=json.loads(p.read_text()) if p.exists() else {}
        if not self.proc or self.job=='live':
            self.progress.setRange(0,100);self.progress.setValue(0);self.log.clear()
            self.preview_active=False;self.failed=False;self.done=False
        self.dataset=root;self.files=files;self.prompts=prompts
        self.plan=read_plan(root)
        self.last_export=[];self.canvas.points=[]
        self.segmented.blockSignals(True);self.segmented.setChecked(self.plan.get('enabled',False));self.segmented.blockSignals(False);self.interval.setValue(self.plan.get('interval',100))
        self.refresh_segments()
        self.location.setText(str(root));self.slider.setRange(0,len(files)-1);self.slider.setValue(0);self.frame_changed(0)
        self.busy(bool(self.proc) and self.job!='live')
    def frame_changed(self,i):
        if not self.files:return
        self.fpsinfo.setText('静态帧回看 · FPS：—')
        p=self.files[i];self.canvas.load(p,self.prompts.get(p.name,[]));self.frameinfo.setText(f'帧 {i+1} / {len(self.files)} · {p.name} · 当前提示点 {len(self.prompts.get(p.name,[]))}')
        mask=self.dataset/'masks'/p.name
        if self.segmented.isChecked():
            for start,end in ranges(self.files,self.plan):
                if start<=i<end:
                    cached=cached_result(self.dataset,self.files,self.prompts,start,end)
                    if cached:mask=self.dataset/cached['masks']/p.name
                    else:mask=Path('/nonexistent/studio_pending_mask')
                    break
        if self.overlay.isChecked() and mask.exists():
            alpha=QImage(str(mask)).convertToFormat(QImage.Format_Grayscale8)
            tint=QImage(alpha.size(),QImage.Format_ARGB32);tint.fill(QColor('#39e58c'));tint.setAlphaChannel(alpha)
            painter=QPainter(self.canvas.pix);painter.setOpacity(.4);painter.drawImage(0,0,tint);painter.end();self.canvas.update()
    def save_prompts(self):
        p=self.dataset/'metadata/prompts.json';p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(self.prompts,indent=2));os.replace(tmp,p);self.refresh_segments()
    def add_point(self,x,y,label):
        if self.proc or not self.files:return
        i=self.slider.value();name=self.files[i].name;self.prompts.setdefault(name,[]).append([round(x,2),round(y,2),label]);self.save_prompts();self.frame_changed(i)
    def undo_point(self):
        if not self.files:return
        name=self.files[self.slider.value()].name
        if self.prompts.get(name):self.prompts[name].pop()
        if not self.prompts.get(name):self.prompts.pop(name,None)
        self.save_prompts();self.frame_changed(self.slider.value())
    def save_plan(self):
        write_json(self.dataset/PLAN_FILE,self.plan);self.refresh_segments();self.frame_changed(self.slider.value())
    def toggle_segmented(self,enabled):
        if not self.files:return
        self.plan['enabled']=enabled;self.save_plan()
    def build_plan(self):
        if not self.files:self.message('请先打开数据集');return
        self.plan=dict(enabled=True,interval=self.interval.value(),starts=list(range(0,len(self.files),self.interval.value())))
        self.segmented.blockSignals(True);self.segmented.setChecked(True);self.segmented.blockSignals(False)
        self.save_plan();self.segmentlist.setCurrentRow(0)
        self.message(f'已生成 {len(self.plan["starts"])} 段，请逐个标注起始帧。相同范围和提示点的已完成段仍可复用。')
    def add_breakpoint(self):
        if not self.files:return
        self.plan['starts']=sorted(set(self.plan.get('starts',[0])+[self.slider.value()]))
        self.plan['enabled']=True;self.segmented.blockSignals(True);self.segmented.setChecked(True);self.segmented.blockSignals(False)
        self.save_plan();self.segmentlist.setCurrentRow(self.plan['starts'].index(self.slider.value()))
    def refresh_segments(self):
        if not self.files:return
        current=self.segmentlist.currentRow();self.segmentlist.blockSignals(True);self.segmentlist.clear()
        for j,(start,end) in enumerate(ranges(self.files,self.plan)):
            annotated=any(p[2]==1 for p in self.prompts.get(self.files[start].name,[]))
            result=cached_result(self.dataset,self.files,self.prompts,start,end)
            state='已完成' if result else ('待处理' if annotated else '待标注')
            self.segmentlist.addItem(f'{j+1:02d} · 帧 {start+1}–{end} · {state}')
        self.segmentlist.setCurrentRow(max(0,min(current,self.segmentlist.count()-1)));self.segmentlist.blockSignals(False)
    def select_segment(self,index):
        if self.files and index>=0:self.slider.setValue(ranges(self.files,self.plan)[index][0])
    def move_segment(self,delta):
        self.segmentlist.setCurrentRow(max(0,min(self.segmentlist.count()-1,self.segmentlist.currentRow()+delta)))
        self.select_segment(self.segmentlist.currentRow())
    def start_segment_job(self,preview):
        if not self.files:self.message('请先打开数据集');return
        if not self.segmented.isChecked():self.message('请先按间隔生成断点或启用分段分割');return
        index=self.segmentlist.currentRow()
        start,end=ranges(self.files,self.plan)[index]
        if not any(p[2]==1 for p in self.prompts.get(self.files[start].name,[])):
            self.slider.setValue(start);self.message(f'请标注第 {start+1} 帧的目标前景点');return
        self.slider.setValue(start);self.frame_changed(start)
        self.save_prompts();self.run_job('segment',self.dataset,['--segment-index',str(index)]+(['--preview'] if preview else ['--force']))
    def start_segment(self,preview=False):
        if not self.files:self.message('请先录制或导入 RGB 目录');return
        if self.segmented.isChecked() and preview:self.start_segment_job(True);return
        starts=self.plan['starts'] if self.segmented.isChecked() else [0]
        missing=[s for s in starts if not any(p[2]==1 for p in self.prompts.get(self.files[s].name,[]))]
        if missing:
            self.slider.setValue(missing[0]);self.message(f'还有 {len(missing)} 个起始帧未标注，请先在第 {missing[0]+1} 帧添加绿色前景点');return
        if preview:self.slider.setValue(0);self.frame_changed(0)
        self.save_prompts();self.run_job('segment',self.dataset,['--preview'] if preview else [])
    def start_export(self):
        if not self.dataset:self.message('请先选择数据');return
        target=EXPORTS/('dataset_'+stamp());self.run_job('validate',self.dataset,['--export',str(target),'--max-depth',str(self.depth.value())])
    def show_export(self):
        if getattr(self,'last_export',None):QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.last_export[0]).parent)))
        else:self.message('尚无导出结果，请先完成分割并验收导出')
    def show_folder(self):
        if self.dataset:QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.dataset)))
    def closeEvent(self,e):
        if self.history_panel.running:
            e.ignore();self.closing=True;self.history_panel.status.setText('等待历史数据操作完成后关闭窗口…')
        elif self.pose_panel.proc:
            e.ignore();self.closing=True;self.pose_panel.stop_job()
        elif self.proc:
            e.ignore();self.closing=True;self.cancel();self.message('等待当前任务安全退出后关闭窗口…')
        else:e.accept()
def main():
    app=QApplication(sys.argv);app.setApplicationName('BundleSDF Studio');w=Studio()
    if len(sys.argv)>1:w.load_dataset(sys.argv[1])
    w.show();sys.exit(app.exec())
if __name__=='__main__':main()

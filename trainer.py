#!/usr/bin/env python3
"""Desktop entry point for validated BundleSDF reconstruction jobs."""
import json
import os
from pathlib import Path
import sys

import importlib.util
try:
    _SYSTEM_QT = importlib.util.find_spec('PySide6.QtCore') is None
except ModuleNotFoundError:
    _SYSTEM_QT = True
if _SYSTEM_QT:sys.path.insert(0, '/usr/lib/python3/dist-packages')
from PySide6.QtCore import QProcess, QProcessEnvironment, QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QPlainTextEdit, QProgressBar, QGroupBox)
if _SYSTEM_QT:sys.path.remove('/usr/lib/python3/dist-packages')

from config import ROOT, RUNTIME, CAMERA_PYTHON

STATE = RUNTIME / 'trainer_state.json'
DEFAULT_DATASET = ROOT/'exports'
PHASES = {'validating': '验收数据', 'tracking': '位姿跟踪与在线重建',
          'global': '全局神经场优化', 'texturing': '模型贴图', 'stopping': '正在停止',
          'complete': '全部完成', 'canceled': '已停止', 'failed': '任务失败'}


def duration(seconds):
    if seconds is None:
        return '尚在估算'
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        return f'{seconds//3600} 小时 {(seconds%3600)//60} 分钟'
    if seconds >= 60:
        return f'{seconds//60} 分 {seconds%60:02d} 秒'
    return f'{seconds} 秒'


class Trainer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.proc = None
        self.buffer = bytearray()
        self.job_dir = None
        self.output = None
        self.total = 0
        self.completed = False
        self.failed = False
        self.canceled = False
        self.stopping = False
        self.selected_dataset = None
        self.setWindowTitle('BundleSDF 训练启动器')
        self.resize(1120, 830)
        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)
        title = QLabel('BundleSDF 训练启动器')
        title.setStyleSheet('font-size:26px;font-weight:700;color:#65dcc9')
        layout.addWidget(title)
        layout.addWidget(QLabel('选择数据集 → 自动验收 → 位姿跟踪 / 在线重建 → 全局优化 → 贴图模型'))
        box = QGroupBox('数据与结果')
        fields = QVBoxLayout(box)
        layout.addWidget(box)
        fields.addWidget(QLabel('数据集目录：请选择包含 rgb、depth、masks 和 cam_K.txt 的 segment 目录'))
        row = QHBoxLayout()
        self.dataset = QLineEdit()
        self.browse = QPushButton('选择数据集…')
        self.browse.clicked.connect(self.choose_dataset)
        row.addWidget(self.dataset, 1)
        row.addWidget(self.browse)
        fields.addLayout(row)
        self.dataset.editingFinished.connect(self.inspect_dataset)
        self.overview = QLabel('尚未选择数据集')
        self.overview.setWordWrap(True)
        fields.addWidget(self.overview)
        fields.addWidget(QLabel('训练结果保存根目录（每次自动创建新目录）'))
        row = QHBoxLayout()
        self.output_root = QLineEdit(str(ROOT/'training_runs'))
        self.browse_output = QPushButton('选择保存位置…')
        self.browse_output.clicked.connect(self.choose_output)
        row.addWidget(self.output_root, 1)
        row.addWidget(self.browse_output)
        fields.addLayout(row)
        row = QHBoxLayout()
        self.start = QPushButton('▶ 开始完整训练')
        self.start.clicked.connect(self.start_training)
        self.stop = QPushButton('■ 停止训练')
        self.stop.clicked.connect(self.stop_training)
        self.stop.setEnabled(False)
        self.results = QPushButton('打开结果目录')
        self.results.clicked.connect(self.open_results)
        self.results.setEnabled(False)
        self.openlog = QPushButton('打开日志')
        self.openlog.clicked.connect(self.open_log)
        self.openlog.setEnabled(False)
        for button in [self.start, self.stop, self.results, self.openlog]:
            row.addWidget(button)
        layout.addLayout(row)
        self.status = QLabel('就绪。默认逐帧运行全部输入，使用已有 SAM3 掩码。')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.output_label = QLabel('结果目录：尚未开始')
        self.output_label.setWordWrap(True)
        self.output_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.output_label)
        self.frames_label = QLabel('位姿输出：0 / 0 帧')
        layout.addWidget(self.frames_label)
        self.frames_bar = QProgressBar()
        layout.addWidget(self.frames_bar)
        self.stage_label = QLabel('当前阶段：待开始')
        layout.addWidget(self.stage_label)
        self.stage_bar = QProgressBar()
        layout.addWidget(self.stage_bar)
        self.timing = QLabel('耗时：—    输入阶段剩余时间：—')
        layout.addWidget(self.timing)
        hint = QLabel('帧进度完成后仍会继续全局优化和贴图；显示“全部完成”才表示训练结束。\n'
                      '8GB 显存运行时，请结束 SAM3 分割及其他占用 GPU 的任务。停止后保留已有结果；重新开始会创建新目录。')
        hint.setWordWrap(True)
        hint.setStyleSheet('color:#b6c6d7')
        layout.addWidget(hint)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(700)
        layout.addWidget(self.log, 1)
        self.setStyleSheet('QMainWindow,QWidget{background:#172231;color:#e5edf5;font-size:14px;}'
            'QGroupBox{border:1px solid #40546b;border-radius:8px;margin-top:12px;padding:12px;}'
            'QGroupBox::title{subcontrol-origin:margin;left:12px;color:#65dcc9;}'
            'QPushButton{background:#294358;border:1px solid #46667b;border-radius:5px;padding:10px;}'
            'QPushButton:hover{background:#355e73;}QPushButton:disabled{color:#75808e;background:#222d3b;}'
            'QLineEdit,QPlainTextEdit{background:#101824;border:1px solid #40546b;padding:6px;}'
            'QProgressBar{border:1px solid #40546b;text-align:center;}QProgressBar::chunk{background:#258c82;}')
        try:
            state = json.loads(STATE.read_text())
        except (OSError, ValueError):
            state = {}
        self.dataset.setText(state.get('dataset', str(DEFAULT_DATASET)))
        self.output_root.setText(state.get('output_root', str(ROOT/'training_runs')))
        old = Path(state['job_dir']) if state.get('job_dir') else None
        if old and (old/'job.json').exists():
            try:
                report = json.loads((old/'job.json').read_text())
                self.job_dir = old
                self.output = Path(report['output'])
                self.output_label.setText('上次结果：' + str(self.output))
                self.results.setEnabled(True)
                self.openlog.setEnabled((old/'train.log').exists())
            except (OSError, ValueError, KeyError):
                pass
        self.inspect_dataset()

    def append(self, message):
        self.log.appendPlainText(message)

    def inspect_dataset(self):
        if self.proc:
            return
        root = Path(self.dataset.text()).expanduser()
        counts = {sub: len(list((root/sub).glob('*.png'))) for sub in ['rgb', 'depth', 'masks']}
        self.total = counts['rgb']
        valid = self.total >= 5 and len(set(counts.values())) == 1 and (root/'cam_K.txt').is_file()
        self.overview.setText(f'RGB {counts["rgb"]} 帧 · 深度 {counts["depth"]} 帧 · 掩码 {counts["masks"]} 帧 · '
                              + ('内参已找到；开始后将逐帧完整验收' if valid else '请检查目录、同名图像数量和内参'))
        self.start.setEnabled(valid)
        self.frames_label.setText(f'位姿输出：0 / {self.total} 帧')
        self.frames_bar.setRange(0, max(self.total, 1))
        self.frames_bar.setValue(0)
        return valid

    def choose_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, '选择 BundleSDF 数据集（segment 目录）',
                                                  str(ROOT/'exports'))
        if folder:
            self.dataset.setText(folder)
            self.inspect_dataset()

    def choose_output(self):
        folder = QFileDialog.getExistingDirectory(self, '选择训练结果保存根目录', self.output_root.text())
        if folder:
            self.output_root.setText(folder)

    def save_state(self):
        RUNTIME.mkdir(parents=True, exist_ok=True)
        path = STATE.with_suffix('.tmp')
        path.write_text(json.dumps(dict(dataset=self.dataset.text(), output_root=self.output_root.text(),
                        job_dir=str(self.job_dir) if self.job_dir else None), ensure_ascii=False, indent=2))
        os.replace(path, STATE)

    def busy(self, value):
        for item in [self.dataset, self.output_root, self.browse, self.browse_output, self.start]:
            item.setEnabled(not value)
        self.stop.setEnabled(value and not self.stopping)

    def start_training(self):
        if self.proc or not self.inspect_dataset():
            return
        self.completed = self.failed = self.canceled = self.stopping = False
        self.buffer.clear()
        self.job_dir = self.output = None
        self.results.setEnabled(False)
        self.openlog.setEnabled(False)
        self.selected_dataset = self.dataset.text()
        self.busy(True)
        self.status.setText('正在启动数据验收…')
        self.output_label.setText('结果目录：创建中…')
        self.stage_bar.setRange(0, 0)
        self.append('开始任务：' + self.dataset.text())
        p = QProcess(self)
        self.proc = p
        p.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        for name in ['PYTHONPATH', 'QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_PLUGIN_PATH']:
            env.remove(name)
        env.insert('PYTHONUNBUFFERED', '1')
        p.setProcessEnvironment(env)
        p.readyReadStandardOutput.connect(self.read_output)
        p.finished.connect(self.finished)
        p.errorOccurred.connect(self.process_error)
        p.start(str(CAMERA_PYTHON), [str(ROOT/'train_job.py'), self.dataset.text(),
                                   '--output-root', self.output_root.text()])

    def process_error(self, error):
        self.append('启动器进程错误：' + self.proc.errorString())
        if error == QProcess.FailedToStart:
            p = self.proc
            self.proc = None
            self.failed = True
            self.busy(False)
            self.status.setText('启动失败，请检查解释器路径')
            self.stage_bar.setRange(0, 100)
            self.stage_bar.setValue(0)
            p.deleteLater()

    def read_output(self):
        self.buffer.extend(bytes(self.proc.readAllStandardOutput()))
        while b'\n' in self.buffer:
            raw, _, rest = self.buffer.partition(b'\n')
            self.buffer = bytearray(rest)
            line = raw.decode('utf-8', errors='replace')
            if line.startswith('EVENT '):
                try:
                    self.handle(json.loads(line[6:]))
                except (ValueError, KeyError, TypeError) as exc:
                    self.append('事件解析失败：' + str(exc))
            elif line:
                self.append(line)

    def handle(self, value):
        kind = value['type']
        if kind == 'job':
            self.job_dir = Path(value['job_dir'])
            self.output = Path(value['output'])
            self.output_label.setText('结果目录：' + str(self.output))
            self.results.setEnabled(True)
            self.save_state()
        elif kind == 'started':
            self.openlog.setEnabled(True)
            self.append('BundleSDF 已启动，PID ' + str(value['pid']))
        elif kind == 'error':
            self.failed = True
            self.status.setText('任务失败：' + value['message'])
            self.append(self.status.text())
        elif kind == 'canceled':
            self.canceled = True
            self.status.setText(value.get('message', '已停止'))
            self.append(self.status.text())
        elif kind == 'done':
            self.completed = True
            self.status.setText(value['message'])
            self.frames_label.setText(f'位姿输出：{value["saved_poses"]} / {value["frames"]} 帧')
            self.frames_bar.setRange(0, value['frames'])
            self.frames_bar.setValue(value['saved_poses'])
            self.stage_label.setText('全部完成：textured_mesh.obj、材质与纹理图片已生成')
            self.stage_bar.setRange(0, 100)
            self.stage_bar.setValue(100)
            self.timing.setText('总耗时：' + duration(value['elapsed_seconds']))
            self.append(value['message'])
        if kind in ['status', 'progress']:
            phase = value.get('phase', 'validating')
            label = PHASES.get(phase, phase)
            self.status.setText(value.get('message', label))
            if phase == 'validating':
                self.stage_label.setText('数据验收：' + str(value.get('index', 0)) + ' / ' + str(value.get('total', self.total)))
                if value.get('total'):
                    self.stage_bar.setRange(0, value['total'])
                    self.stage_bar.setValue(value.get('index', 0))
            else:
                count = value.get('index', 0)
                self.frames_label.setText(f'位姿输出：{count} / {self.total} 帧，尚未输出 {max(self.total-count,0)} 帧')
                self.frames_bar.setRange(0, self.total)
                self.frames_bar.setValue(count)
                step, steps = value.get('iteration', 0), value.get('iterations', 0)
                self.stage_label.setText('当前阶段：' + label + (f' · {step}/{steps}' if steps else ''))
                self.stage_bar.setRange(0, steps if steps else 0)
                if steps:
                    self.stage_bar.setValue(step)
                rate = value.get('seconds_per_frame')
                eta = duration(value.get('eta_input_seconds')) if phase == 'tracking' else '输入阶段已结束 / 当前阶段不估算'
                self.timing.setText(f'耗时：{duration(value.get("elapsed", 0))}    输入阶段剩余：{eta}'
                                    + (f'    约 {rate:.2f} 秒/帧' if rate and phase == 'tracking' else ''))
        if value.get('lines'):
            self.append('\n'.join(value['lines']))

    def finished(self, code, status):
        self.read_output()
        p = self.proc
        self.proc = None
        p.deleteLater()
        self.busy(False)
        self.stopping = False
        if not (self.completed or self.failed or self.canceled):
            self.failed = True
            self.status.setText(f'任务异常退出，代码 {code}。请查看日志。')
        if not self.completed:
            self.stage_bar.setRange(0, 100)
            self.stage_bar.setValue(0)
            self.stage_label.setText('当前阶段：' + ('已停止' if self.canceled else '任务失败'))
        self.save_state()

    def stop_training(self):
        if self.proc and not self.stopping:
            self.stopping = True
            self.stop.setEnabled(False)
            self.proc.write(b'stop\n')
            self.status.setText('正在停止全部训练子进程，请等待…')

    def open_results(self):
        if self.output:
            target = self.output if self.output.exists() else self.job_dir
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def open_log(self):
        if self.job_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.job_dir/'train.log')))

    def closeEvent(self, event):
        if self.proc:
            event.ignore()
            self.status.setText('训练仍在运行，可最小化窗口。若要关闭，请先点击“停止训练”并等待任务退出。')
        else:
            self.save_state()
            event.accept()


def main():
    app = QApplication(sys.argv)
    window = Trainer()
    if len(sys.argv) > 1:
        window.dataset.setText(sys.argv[1])
        window.inspect_dataset()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

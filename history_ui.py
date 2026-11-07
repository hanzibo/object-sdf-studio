"""Qt history library: previews, filtering, inode-aware totals and batch deletion."""
from datetime import datetime
from pathlib import Path
import shutil
import time
from PySide6.QtCore import Qt,QThread,Signal,QTimer,QUrl,QSize
from PySide6.QtGui import QPixmap,QIcon,QDesktopServices
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,
    QLineEdit,QComboBox,QTreeWidget,QTreeWidgetItem,QAbstractItemView,QSplitter,
    QProgressBar,QFileDialog,QMessageBox,QScrollArea,QDialog,QListWidget)
from config import ROOT
from data_ops import atomic_json
from history_store import Catalog,CATEGORIES,accounting,fmt_size,relative_time,read_json,inside


class HistoryTask(QThread):
    result=Signal(object)
    error=Signal(str)
    progress=Signal(str)

    def __init__(self,fn,parent=None):
        super().__init__(parent);self.fn=fn

    def run(self):
        try:self.result.emit(self.fn(self.progress.emit))
        except Exception as exc:self.error.emit(str(exc))


class HistoryPanel(QWidget):
    activity=Signal(bool)
    deleted=Signal(object)

    def __init__(self,studio,root=ROOT):
        super().__init__();self.studio=studio;self.root=Path(root).resolve()
        self.catalog=None;self.worker=None;self.deleting=False;self.refresh_after=False
        self.items={};self.last_report=None;self.preview_paths=[]
        self.settings=self.root/'runtime/history_roots.json'
        self.extra_roots=read_json(self.settings).get('roots',[])
        layout=QVBoxLayout(self)
        title=QLabel('历史数据管理');title.setStyleSheet('font-size:24px;font-weight:700;color:#65dcc9');layout.addWidget(title)
        layout.addWidget(QLabel('按用途查看历史数据、预览内容并清理空间。原目录保持不动；勾选后可批量永久删除。'))
        diskrow=QHBoxLayout();self.disk_label=QLabel();diskrow.addWidget(self.disk_label,1)
        self.disk_bar=QProgressBar();self.disk_bar.setRange(0,100);self.disk_bar.setMaximumWidth(260);diskrow.addWidget(self.disk_bar);layout.addLayout(diskrow)
        tools=QHBoxLayout()
        self.category=QComboBox();self.category.addItem('全部用途');self.category.addItems(CATEGORIES);self.category.currentTextChanged.connect(self.apply_filter);tools.addWidget(self.category)
        self.search=QLineEdit();self.search.setPlaceholderText('搜索名称、路径或来源数据集');self.search.textChanged.connect(self.apply_filter);tools.addWidget(self.search,1)
        self.refresh=QPushButton('刷新 / 重新统计');self.refresh.clicked.connect(self.scan);tools.addWidget(self.refresh)
        self.locations=QPushButton('扫描位置…');self.locations.clicked.connect(self.manage_roots);tools.addWidget(self.locations)
        layout.addLayout(tools)
        split=QSplitter(Qt.Horizontal);layout.addWidget(split,1)
        self.tree=QTreeWidget();self.tree.setColumnCount(6)
        self.tree.setHeaderLabels(['数据 / 结果（勾选删除）','录制 / 保存时间','距今','帧数','文件量','预计单独释放'])
        self.tree.setColumnWidth(0,300);self.tree.setColumnWidth(1,152);self.tree.setColumnWidth(2,90)
        self.tree.setColumnWidth(3,58);self.tree.setColumnWidth(4,100);self.tree.setColumnWidth(5,120)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection);self.tree.setAlternatingRowColors(True)
        self.tree.setIndentation(28)
        assets=Path(__file__).resolve().parent/'assets'
        self.tree.setStyleSheet(f'''
            QTreeWidget{{background:#101824;alternate-background-color:#19293a;}}
            QTreeWidget::item{{padding-top:4px;padding-bottom:4px;min-height:30px;}}
            QTreeWidget::item:selected{{background:#31576c;}}
            QTreeWidget::indicator{{width:26px;height:26px;border:2px solid #8faabd;border-radius:5px;background:#101824;}}
            QTreeWidget::indicator:hover{{border-color:#65dcc9;background:#294358;}}
            QTreeWidget::indicator:checked{{background:#258c82;border-color:#65dcc9;image:url("{assets/'check.svg'}");}}
            QTreeWidget::indicator:disabled{{border-color:#506172;background:#222d3b;}}
            QTreeWidget::branch:has-children:closed{{image:url("{assets/'chevron-right.svg'}");}}
            QTreeWidget::branch:has-children:open{{image:url("{assets/'chevron-down.svg'}");}}
        ''')
        self.tree.currentItemChanged.connect(self.show_entry);self.tree.itemChanged.connect(self.selection_changed)
        split.addWidget(self.tree)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setMinimumWidth(290)
        preview=QWidget();details=QVBoxLayout(preview)
        self.detail_title=QLabel('选择一条历史记录查看预览');self.detail_title.setWordWrap(True);self.detail_title.setStyleSheet('font-size:17px;color:#65dcc9;font-weight:700');details.addWidget(self.detail_title)
        self.photos=[]
        for _ in range(3):
            label=QLabel();label.setAlignment(Qt.AlignCenter);label.setMinimumHeight(125);label.setStyleSheet('background:#101824;');details.addWidget(label);self.photos.append(label)
        self.detail=QLabel('可按 Ctrl / Shift 多选行，再点击“勾选高亮行”。');self.detail.setWordWrap(True);self.detail.setTextInteractionFlags(Qt.TextSelectableByMouse);details.addWidget(self.detail)
        self.folder=QPushButton('打开目录');self.folder.clicked.connect(self.open_folder);details.addWidget(self.folder)
        self.video=QPushButton('播放结果视频');self.video.clicked.connect(self.open_video);details.addWidget(self.video)
        self.load_capture=QPushButton('载入采集 / 建模数据页');self.load_capture.clicked.connect(lambda:self.load_dataset(False));details.addWidget(self.load_capture)
        self.load_pose=QPushButton('载入 FoundationPose 跟踪页');self.load_pose.clicked.connect(lambda:self.load_dataset(True));details.addWidget(self.load_pose)
        self.load_model=QPushButton('使用此重建模型跟踪');self.load_model.clicked.connect(self.use_model);details.addWidget(self.load_model)
        details.addStretch();scroll.setWidget(preview);split.addWidget(scroll);split.setSizes([950,340])
        selection=QHBoxLayout()
        self.check_visible=QPushButton('勾选全部可见项');self.check_visible.clicked.connect(self.select_visible);selection.addWidget(self.check_visible)
        self.check_highlight=QPushButton('勾选高亮行');self.check_highlight.clicked.connect(self.select_highlighted);selection.addWidget(self.check_highlight)
        self.clear=QPushButton('清空勾选');self.clear.clicked.connect(self.clear_checked);selection.addWidget(self.clear);selection.addStretch()
        self.remove=QPushButton('永久删除勾选项…');self.remove.setStyleSheet('QPushButton{background:#763b46;} QPushButton:disabled{background:#33333a;}');self.remove.clicked.connect(self.confirm_delete);selection.addWidget(self.remove);layout.addLayout(selection)
        self.totals=QLabel('尚未扫描');self.totals.setWordWrap(True);layout.addWidget(self.totals)
        self.last_feedback=QLabel('');self.last_feedback.setWordWrap(True);self.last_feedback.setStyleSheet('color:#65dcc9;');layout.addWidget(self.last_feedback)
        self.status=QLabel('打开本页后扫描历史目录。扫描只读取文件信息，不加载 GPU 模型。');self.status.setWordWrap(True);layout.addWidget(self.status)
        self.progress=QProgressBar();self.progress.setRange(0,100);layout.addWidget(self.progress)
        self.selection_timer=QTimer(self);self.selection_timer.setSingleShot(True);self.selection_timer.timeout.connect(self.update_selection)
        self.update_disk();self.set_controls()

    @property
    def running(self):return self.worker is not None

    def update_disk(self):
        usage=shutil.disk_usage(self.root)
        self.disk_label.setText(f'当前数据卷：可用 {fmt_size(usage.free)} / 总计 {fmt_size(usage.total)} · {self.root}')
        self.disk_bar.setValue(round(usage.used/usage.total*100))
        self.disk_bar.setFormat('已用 %p%')

    def set_controls(self):
        busy=self.running
        for widget in [self.refresh,self.locations,self.category,self.search,self.check_visible,self.check_highlight,self.clear,self.tree]:widget.setEnabled(not busy)
        self.remove.setEnabled(not busy and bool(self.checked()))
        self.show_entry(self.tree.currentItem(),None)

    def start_task(self,fn,callback,deleting=False):
        if self.worker:return
        self.deleting=deleting;self.worker=HistoryTask(fn,self)
        self.worker.progress.connect(self.status.setText)
        self.worker.result.connect(callback);self.worker.error.connect(self.task_error)
        self.worker.finished.connect(self.task_finished)
        self.progress.setRange(0,0)
        if deleting:self.activity.emit(True)
        self.set_controls();self.worker.start()

    def task_error(self,message):
        self.status.setText('操作未完成：'+message)
        if self.deleting:QMessageBox.warning(self,'清理未完成',message)

    def task_finished(self):
        task=self.worker;self.worker=None
        if task:task.deleteLater()
        was_deleting=self.deleting;self.deleting=False
        if was_deleting:self.activity.emit(False)
        self.progress.setRange(0,100);self.progress.setValue(100)
        self.update_disk();self.set_controls()
        if getattr(self.studio,'closing',False):QTimer.singleShot(0,self.studio.close)
        elif self.refresh_after:
            self.refresh_after=False;QTimer.singleShot(0,self.scan)

    def scan(self):
        if self.running:return
        self.status.setText('后台扫描历史数据…')
        self.start_task(lambda progress:Catalog(self.root,self.extra_roots).scan(progress),self.scanned)

    def scanned(self,catalog):
        old_checked=set(self.checked());self.catalog=catalog
        self.tree.blockSignals(True);self.tree.clear();self.items={}
        groups={}
        for category in CATEGORIES:
            parent=QTreeWidgetItem(self.tree,[category]);parent.setFlags(Qt.ItemIsEnabled);parent.setExpanded(True);groups[category]=parent
        for entry in sorted(catalog.entries.values(),key=lambda e:e.saved,reverse=True):
            size=accounting(entry.nodes)
            item=QTreeWidgetItem(groups[entry.category],[entry.title,datetime.fromtimestamp(entry.saved).strftime('%Y-%m-%d %H:%M'),
                relative_time(entry.saved),str(entry.frames) if entry.frames else '—',fmt_size(size['logical']),fmt_size(size['reclaim'])])
            item.setData(0,Qt.UserRole,str(entry.path));item.setFlags(item.flags()|Qt.ItemIsUserCheckable)
            item.setCheckState(0,Qt.Checked if str(entry.path) in old_checked else Qt.Unchecked)
            item.setToolTip(0,str(entry.path));item.setToolTip(5,'按单独删除此目录计算；批量选择后的值以底部合并统计为准')
            if entry.previews:
                pix=QPixmap(entry.previews[0]);item.setIcon(0,QIcon(pix.scaled(90,60,Qt.KeepAspectRatio,Qt.SmoothTransformation)))
            if entry.active or entry.errors:
                item.setText(0,entry.title+(' · 使用中' if entry.active else ' · 扫描异常'))
                item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable);item.setCheckState(0,Qt.Unchecked)
            self.items[str(entry.path)]=item
        self.tree.setIconSize(QSize(80,54))
        self.tree.blockSignals(False);self.apply_filter();self.update_selection()
        summary=catalog.selected(list(catalog.entries))
        self.status.setText(f'已索引 {len(catalog.entries)} 项 · 去重后占用 {fmt_size(summary["allocated"])} · 扫描 {catalog.scan_seconds:.1f} 秒。来源 / 缓存目录可能重叠，勾选时自动合并。')

    def apply_filter(self,*_):
        if not self.catalog:return
        category=self.category.currentText();query=self.search.text().strip().casefold()
        for path,item in self.items.items():
            entry=self.catalog.entries[Path(path)]
            match=(category=='全部用途' or entry.category==category) and query in (entry.title+' '+path+' '+entry.source).casefold()
            item.setHidden(not match)
        for i in range(self.tree.topLevelItemCount()):
            parent=self.tree.topLevelItem(i)
            parent.setHidden(not any(not parent.child(j).isHidden() for j in range(parent.childCount())))

    def checked(self):
        return [path for path,item in self.items.items() if item.checkState(0)==Qt.Checked]

    def selection_changed(self,*_):self.selection_timer.start(100)

    def update_selection(self):
        paths=self.checked()
        if not self.catalog:self.totals.setText('尚未扫描');return
        size=self.catalog.selected(paths)
        self.totals.setText(f'已勾选 {len(paths)} 行 → 合并为 {size["items"]} 个删除目录 · 文件量 {fmt_size(size["logical"])} · 预计可释放 {fmt_size(size["reclaim"])}\n硬链接仍被其他目录引用的文件不计入释放量。筛选隐藏的已勾选项仍会保留。')
        self.remove.setEnabled(not self.running and bool(paths))

    def select_visible(self):
        self.tree.blockSignals(True)
        for item in self.items.values():
            if not item.isHidden() and item.flags() & Qt.ItemIsUserCheckable:item.setCheckState(0,Qt.Checked)
        self.tree.blockSignals(False);self.update_selection()

    def select_highlighted(self):
        self.tree.blockSignals(True)
        for item in self.tree.selectedItems():
            if item.flags() & Qt.ItemIsUserCheckable:item.setCheckState(0,Qt.Checked)
        self.tree.blockSignals(False);self.update_selection()

    def clear_checked(self):
        self.tree.blockSignals(True)
        for item in self.items.values():item.setCheckState(0,Qt.Unchecked)
        self.tree.blockSignals(False);self.update_selection()

    def entry(self):
        item=self.tree.currentItem()
        path=item.data(0,Qt.UserRole) if item else None
        return self.catalog.entries.get(Path(path)) if path and self.catalog else None

    def show_entry(self,current,previous):
        entry=self.entry()
        for button in [self.folder,self.video,self.load_capture,self.load_pose,self.load_model]:button.setEnabled(False)
        if not entry:
            self.preview_paths=[];self.detail_title.setText('选择一条历史记录查看预览')
            self.detail.setText('可按 Ctrl / Shift 多选行，再点击“勾选高亮行”。')
            for label in self.photos:label.clear();label.hide()
            return
        size=accounting(entry.nodes);dependents=self.catalog.dependents([entry.path])
        self.detail_title.setText(entry.title)
        self.preview_paths=entry.previews[:3]
        for i,label in enumerate(self.photos):
            if i<len(self.preview_paths):
                pix=QPixmap(self.preview_paths[i]);label.setPixmap(pix.scaled(360,205,Qt.KeepAspectRatio,Qt.SmoothTransformation));label.show()
            else:label.clear();label.setText('暂无图像预览' if i==0 else '');label.setVisible(i==0)
        lines=[entry.category,datetime.fromtimestamp(entry.saved).strftime('%Y-%m-%d %H:%M:%S')+' · '+relative_time(entry.saved)+' · '+entry.time_basis,
               f'帧数：{entry.frames or "—"}',f'文件量：{fmt_size(size["logical"])}',f'引用的数据块：{fmt_size(size["allocated"])}',f'单独删除预计释放：{fmt_size(size["reclaim"])}',str(entry.path)]
        if entry.source:lines+=['来源：'+entry.source]
        if dependents:lines+=['其他关联记录：'+str(len(dependents))+' 项']+[e['title'] for e in dependents[:4]]
        lines+=entry.notes+entry.errors
        if entry.active:lines+=['任务正在使用，不能删除']
        self.detail.setText('\n\n'.join(lines))
        busy=self.running
        self.folder.setEnabled(not busy);self.video.setEnabled(not busy and bool(entry.video))
        self.load_capture.setEnabled(not busy and bool(entry.dataset));self.load_pose.setEnabled(not busy and bool(entry.dataset))
        self.load_model.setEnabled(not busy and bool(entry.model))

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if hasattr(self,'photos'):
            for path,label in zip(self.preview_paths,self.photos):
                pix=QPixmap(path);label.setPixmap(pix.scaled(max(240,label.width()-8),190,Qt.KeepAspectRatio,Qt.SmoothTransformation))

    def open_folder(self):
        entry=self.entry()
        if entry:QDesktopServices.openUrl(QUrl.fromLocalFile(str(entry.path)))

    def open_video(self):
        entry=self.entry()
        if entry and entry.video:QDesktopServices.openUrl(QUrl.fromLocalFile(entry.video))

    def load_dataset(self,pose):
        entry=self.entry()
        if not entry or not entry.dataset:return
        try:
            if self.studio.proc or self.studio.pose_panel.proc:raise ValueError('请先停止当前采集或推理任务')
            if pose:self.studio.pose_panel.set_dataset(entry.dataset)
            else:self.studio.load_dataset(entry.dataset)
            self.studio.tabs.setCurrentIndex(1 if pose else 0)
        except Exception as exc:self.status.setText(str(exc))

    def use_model(self):
        entry=self.entry()
        if not entry or not entry.model:return
        if self.studio.pose_panel.proc:self.status.setText('请先停止当前跟踪');return
        self.studio.pose_panel.mesh.setText(entry.model);self.studio.pose_panel.save_settings();self.studio.tabs.setCurrentIndex(1)

    def confirm_delete(self):
        paths=self.checked()
        if not paths or not self.catalog:return
        if self.studio.proc or self.studio.pose_panel.proc:
            self.status.setText('请先停止采集或推理任务，再清理历史数据');return
        totals=self.catalog.selected(paths);refs=self.catalog.dependents(paths)
        dialog=QMessageBox(self);dialog.setIcon(QMessageBox.Warning);dialog.setWindowTitle('永久删除历史数据')
        dialog.setText(f'永久删除 {totals["items"]} 个目录？\n文件量：{fmt_size(totals["logical"])}\n预计释放：{fmt_size(totals["reclaim"])}')
        info='不会放入回收站，删除后无法撤销。只删除勾选范围，不自动删除关联数据。'
        if refs:info+=f'\n另有 {len(refs)} 条记录引用这些数据或模型，删除后相关来源可能失效。'
        dialog.setInformativeText(info)
        dialog.setDetailedText('\n'.join(totals['paths'])+'\n\n关联记录：\n'+'\n'.join(r['title'] for r in refs))
        delete_button=dialog.addButton('永久删除',QMessageBox.DestructiveRole)
        cancel=dialog.addButton('取消',QMessageBox.RejectRole);dialog.setDefaultButton(cancel)
        dialog.exec()
        if dialog.clickedButton() is delete_button:self.perform_delete(paths)

    def perform_delete(self,paths):
        """Invoked after confirmation; also used by isolated integration tests."""
        if self.studio.proc or self.studio.pose_panel.proc:
            self.status.setText('请先停止采集或推理任务');return
        catalog=self.catalog
        self.start_task(lambda progress:catalog.delete(paths,progress),self.deletion_done,True)

    def deletion_done(self,report):
        self.last_report=report
        self.tree.setCurrentItem(None)
        self.deleted.emit(report['deleted'])
        self.refresh_after=True
        delta=report['observed_free_delta']
        text=f'已删除 {len(report["deleted"])} 个目录；失败 {len(report["failed"])} 个。\n删除前预计释放：{fmt_size(report["selected"]["reclaim"])}\n当前磁盘可用空间变化：{"+" if delta>=0 else ""}{fmt_size(delta)}\n可用空间变化也会受到同时运行的其他程序影响。'
        if report['failed']:text+='\n\n'+'\n'.join(f['path']+'：'+f['message'] for f in report['failed'])
        self.status.setText(text)
        self.totals.setText(text)
        self.last_feedback.setText(f'上次清理：删除 {len(report["deleted"])} 个目录，失败 {len(report["failed"])} 个；可用空间变化 {"+" if delta>=0 else ""}{fmt_size(delta)}')
        QMessageBox.information(self,'历史数据清理结果',text)

    def manage_roots(self):
        dialog=QDialog(self);dialog.setWindowTitle('历史数据扫描位置');dialog.resize(740,370)
        layout=QVBoxLayout(dialog);layout.addWidget(QLabel('默认扫描 Studio 的采集、导出、重建、跟踪及预览目录。\n可增加外部数据目录或其父目录；移除扫描位置不会删除数据。'))
        listing=QListWidget();listing.addItems(self.extra_roots);layout.addWidget(listing)
        row=QHBoxLayout();add=QPushButton('添加目录…');remove=QPushButton('移除选中的扫描位置');close=QPushButton('保存并关闭')
        row.addWidget(add);row.addWidget(remove);row.addWidget(close);layout.addLayout(row)
        def add_root():
            path=QFileDialog.getExistingDirectory(dialog,'选择外部数据或结果目录')
            if path:
                path=str(Path(path).resolve())
                if path not in [listing.item(i).text() for i in range(listing.count())]:listing.addItem(path)
        add.clicked.connect(add_root);remove.clicked.connect(lambda:listing.takeItem(listing.currentRow()) if listing.currentRow()>=0 else None);close.clicked.connect(dialog.accept)
        if dialog.exec()==QDialog.Accepted:
            self.extra_roots=[listing.item(i).text() for i in range(listing.count())]
            atomic_json(self.settings,dict(roots=self.extra_roots));self.scan()

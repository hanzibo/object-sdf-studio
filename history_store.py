"""Read-only catalog and explicit, guarded deletion of recognized Studio data.

No content hashing and no recursive symlink following. Inode/link accounting avoids
claiming space for RGB-D files still linked from exports or another dataset.
"""
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import stat
import time
from config import ROOT
from data_lock import DataAccess

CATEGORIES = ['采集 / 导入 RGB-D','BundleSDF 导出数据','BundleSDF 重建结果',
              'FoundationPose 跟踪结果','SAM3 掩码 / 缓存','相机预览缓存']


def read_json(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}


def inside(path, parent):
    return path == parent or parent in path.parents


def related(a, b):
    return inside(a,b) or inside(b,a)


def safe_dirs(path):
    try:return sorted(p for p in path.iterdir() if p.is_dir() and not p.is_symlink())
    except OSError:return []


def pictures(path):
    try:return sorted(p for p in path.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg'} and p.is_file() and not p.is_symlink())
    except OSError:return []


def fmt_size(value):
    sign='−' if value<0 else '';n=abs(value)
    for suffix in ['B','KiB','MiB','GiB','TiB']:
        if n<1024 or suffix=='TiB':return f'{sign}{n:.1f} {suffix}' if suffix!='B' else f'{sign}{n:.0f} B'
        n/=1024


def relative_time(timestamp, now=None):
    seconds=max(0,(now or time.time())-timestamp)
    if seconds<60:return '刚刚'
    if seconds<3600:return f'{int(seconds/60)} 分钟前'
    if seconds<86400:return f'{int(seconds/3600)} 小时前'
    if seconds<86400*30:return f'{int(seconds/86400)} 天前'
    return f'{int(seconds/(86400*30))} 个月前'


def timestamp(path, metadata=None):
    value=metadata or {}
    if isinstance(value.get('started_at'),(int,float)):return value['started_at'], '任务记录'
    # Directory names are host wall time; image names are device timestamps.
    for name in [path.name,path.parent.name]:
        match=re.search(r'(20\d{6})_(\d{6})',name)
        if match:
            try:return datetime.strptime(''.join(match.groups()),'%Y%m%d%H%M%S').timestamp(),'目录时间'
            except ValueError:pass
    for hint in ['metadata/frames.jsonl','metadata/source_frames.jsonl']:
        try:
            with (path/hint).open() as handle:first=json.loads(handle.readline())
            if first.get('host_time_ns'):return first['host_time_ns']/1e9,'首帧采集时间'
        except (OSError,ValueError):pass
    for hint in ['metadata/capture.json','cam_K.txt']:
        p=path/hint
        if p.is_file():return p.stat().st_mtime,'文件时间估计'
    return path.stat().st_mtime,'文件时间估计'


@dataclass
class Node:
    path: Path
    inode: tuple
    size: int
    blocks: int
    links: int
    directory: bool


@dataclass
class Entry:
    path: Path
    category: str
    title: str
    identity: tuple
    saved: float
    time_basis: str
    frames: int = 0
    previews: list = field(default_factory=list)
    source: str = ''
    references: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    dataset: str = ''
    model: str = ''
    video: str = ''
    nodes: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    active: bool = False


def accounting(nodes):
    unique={str(n.path):n for n in nodes}
    by_inode={};counts=Counter()
    logical=0
    for node in unique.values():
        by_inode[node.inode]=node;counts[node.inode]+=1
        if not node.directory:logical+=node.size
    allocated=sum(n.blocks for n in by_inode.values())
    reclaim=sum(n.blocks for inode,n in by_inode.items() if n.directory or counts[inode]>=n.links)
    return dict(logical=logical,allocated=allocated,reclaim=reclaim,files=sum(not n.directory for n in unique.values()),
                shared_inodes=sum(not n.directory and counts[key]<n.links for key,n in by_inode.items()))


def active_paths():
    """Also recognizes older CLI workers that predate the shared data lease."""
    result=[]
    scripts={'worker.py','train_job.py','run_custom.py','run_foundationpose_offline.py'}
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args=[a.decode(errors='replace') for a in (proc/'cmdline').read_bytes().split(b'\0') if a]
            script=next((i for i,a in enumerate(args) if Path(a).name in scripts),None)
            if script is None:continue
            for arg in args[script+1:]:
                if arg.startswith('/'):
                    result.append((Path(arg).resolve(),int(proc.name)))
        except (OSError,ValueError):pass
    return result


class Catalog:
    def __init__(self, root=ROOT, extra_roots=None):
        self.root=Path(root).resolve()
        self.runtime=self.root/'runtime'
        self.extra_roots=[Path(p).expanduser().resolve() for p in (extra_roots or [])]
        self.entries={}
        self.scopes=[]
        self.scan_seconds=0.

    def add(self,path,category,title=None,source='',note=''):
        path=Path(path)
        if not path.is_dir() or path.is_symlink() or path.resolve()!=path:return None
        if path in self.entries:return self.entries[path]
        info=path.stat();saved,basis=timestamp(path,read_json(path/'job.json'))
        entry=Entry(path,category,title or path.name,(info.st_dev,info.st_ino),saved,basis,source=source)
        if note:entry.notes.append(note)
        self.entries[path]=entry
        return entry

    def add_dataset(self,path,export=False):
        entry=self.add(path,CATEGORIES[1] if export else CATEGORIES[0])
        if not entry:return
        datasets=[path] if (path/'rgb').is_dir() else [p for p in safe_dirs(path) if (p/'rgb').is_dir()]
        entry.frames=sum(len(pictures(p/'rgb')) for p in datasets)
        if datasets:
            entry.dataset=str(datasets[0]);photos=pictures(datasets[0]/'rgb')
            if photos:entry.previews=[str(photos[i]) for i in sorted({0,len(photos)//2,len(photos)-1})]
        details=read_json(path/'metadata/export.json') or read_json(path/'export_report.json')
        if not details and datasets:details=read_json(datasets[0]/'metadata/export.json')
        entry.source=details.get('source',details.get('dataset',''))
        if len(datasets)>1:entry.notes.append(f'包含 {len(datasets)} 个 segment；删除整个导出批次')
        if not export:
            for name,label in [('segmentation_runs','分割运行缓存'),('pose_initializations','位姿初始化掩码'),('backups','旧掩码备份')]:
                for child in safe_dirs(path/'metadata'/name):
                    cache=self.add(child,CATEGORIES[4],f'{path.name} · {label} · {child.name}',str(path),
                        '删除缓存后相关分段可能需要重新分割；原始 RGB-D 保留')
                    if cache:
                        cache.previews=[str(child/'preview.jpg')] if (child/'preview.jpg').is_file() else entry.previews[:1]
            if (path/'masks').is_dir():
                mask=self.add(path/'masks',CATEGORIES[4],f'{path.name} · 已发布完整掩码',str(path),
                    '删除后，此工作数据集需重新分割才能再次导出训练数据')
                if mask:mask.frames=len(pictures(path/'masks'));mask.previews=entry.previews[:1]

    def add_result(self,path,pose=False):
        entry=self.add(path,CATEGORIES[3] if pose else CATEGORIES[2])
        if not entry:return
        metadata=read_json(path/'job.json') or read_json(path/'summary.json')
        entry.source=metadata.get('dataset','')
        entry.references=[str(metadata[key]) for key in ['dataset','mesh','init_metadata','parent_run'] if metadata.get(key)]
        output=path if pose or (path/'textured_mesh.obj').is_file() else path/'reconstruction'
        entry.frames=len(list((output/'ob_in_cam').glob('*.txt')))
        if (output/'textured_mesh.obj').is_file():entry.model=str(output/'textured_mesh.obj')
        if metadata.get('status'):entry.notes.append('任务状态：'+metadata['status'])
        photos=pictures(path/'frames')
        if photos:entry.previews=[str(photos[i]) for i in sorted({0,len(photos)//2,len(photos)-1})]
        for preview in [path/'contact_sheet.jpg',output/'tracking_demo/random_samples/contact_sheet.jpg']:
            if not entry.previews and preview.is_file():entry.previews=[str(preview)]
        if not entry.previews and entry.source:
            photos=pictures(Path(entry.source)/'rgb')
            if photos:entry.previews=[str(photos[i]) for i in sorted({0,len(photos)//2,len(photos)-1})]
        for video in [path/'tracking.mp4',output/'tracking_demo/pose_tracking.mp4']:
            if video.is_file():entry.video=str(video);break
        if metadata.get('mesh') and not Path(metadata['mesh']).exists():entry.notes.append('引用的模型已不存在')
        if entry.source and not Path(entry.source).exists():entry.notes.append('来源数据已不存在；已有视频和位姿仍可查看')

    def discover(self):
        self.entries={}
        self.scopes=[self.root/'datasets',self.root/'exports',self.root/'training_runs',self.root/'foundationpose_runs',self.root/'output',self.root/'runtime']+self.extra_roots
        for path in safe_dirs(self.root/'datasets'):self.add_dataset(path)
        for path in safe_dirs(self.root/'exports'):self.add_dataset(path,True)
        for path in safe_dirs(self.root/'training_runs'):self.add_result(path)
        for path in safe_dirs(self.root/'foundationpose_runs'):self.add_result(path,True)
        for path in safe_dirs(self.root/'output'):
            if (path/'reconstruction').is_dir() or (path/'textured_mesh.obj').is_file():self.add_result(path)
        for path in safe_dirs(self.root):
            if path.name.startswith('recon_') and ((path/'ob_in_cam').is_dir() or (path/'final').is_dir()):
                self.scopes.append(path);self.add_result(path)
        for path in safe_dirs(self.runtime):
            if path.name.startswith('preview_'):
                entry=self.add(path,CATEGORIES[5])
                if entry and (path/'metadata/live.jpg').exists():entry.previews=[str(path/'metadata/live.jpg')]
        for root in self.extra_roots:
            if root==self.root:continue
            candidates=[root]+safe_dirs(root)
            for path in candidates:
                if (path/'rgb').is_dir() and (path/'cam_K.txt').is_file():self.add_dataset(path)
                elif (path/'export_report.json').is_file():self.add_dataset(path,True)
                elif (path/'summary.json').is_file() and (path/'ob_in_cam').is_dir():self.add_result(path,True)
                elif (path/'reconstruction').is_dir() or (path/'textured_mesh.obj').is_file():self.add_result(path)
        # A custom training output root is already remembered by the launcher.
        custom=read_json(self.runtime/'trainer_state.json').get('output_root')
        if custom:
            custom=Path(custom).expanduser().resolve()
            if custom!=self.root/'training_runs':
                self.scopes.append(custom)
                for path in safe_dirs(custom):
                    if (path/'job.json').is_file():self.add_result(path)
        for entry in self.entries.values():
            if entry.source and entry.source not in entry.references:entry.references.append(entry.source)
        return self

    def scan(self, progress=lambda _:None):
        started=time.monotonic();self.discover()
        roots=self.normalized(list(self.entries))
        nodes={};errors=[]
        def record(path):
            try:
                s=path.lstat()
                node=Node(path,(s.st_dev,s.st_ino),s.st_size,s.st_blocks*512,s.st_nlink,stat.S_ISDIR(s.st_mode))
                nodes[path]=node
            except OSError as exc:errors.append((path,str(exc)))
        for root in roots:
            progress('扫描 '+root.name);record(root)
            def walk_error(exc):errors.append((Path(exc.filename or root),str(exc)))
            for folder,dirs,files in os.walk(root,followlinks=False,onerror=walk_error):
                current=Path(folder)
                kept=[]
                for name in dirs:
                    path=current/name;record(path)
                    if path.is_symlink():continue
                    try:
                        if path.stat().st_dev!=root.stat().st_dev or os.path.ismount(path):
                            errors.append((path,'包含挂载点，不允许删除'));continue
                    except OSError:continue
                    kept.append(name)
                dirs[:]=kept
                for name in files:record(current/name)
        for path,node in nodes.items():
            for ancestor in [path]+list(path.parents):
                if ancestor in self.entries:self.entries[ancestor].nodes.append(node)
        active=active_paths()
        for entry in self.entries.values():
            entry.errors=[text for path,text in errors if inside(path,entry.path)]
            entry.active=any(related(entry.path,path) for path,_ in active)
            if entry.category==CATEGORIES[4] and not entry.frames:
                entry.frames=sum(not n.directory and n.path.suffix.lower()=='.png' for n in entry.nodes)
        self.scan_seconds=time.monotonic()-started
        return self

    @staticmethod
    def normalized(paths):
        result=[]
        for path in sorted(set(map(Path,paths)),key=lambda p:(len(p.parts),str(p))):
            if not any(inside(path,parent) for parent in result):result.append(path)
        return result

    def selected(self,paths):
        normalized=self.normalized(paths)
        entries=[self.entries[p] for p in normalized]
        totals=accounting(n for entry in entries for n in entry.nodes)
        totals.update(items=len(entries),selected_rows=len(set(paths)),paths=[str(e.path) for e in entries])
        return totals

    def dependents(self,paths):
        paths=self.normalized(paths);found=[]
        for entry in self.entries.values():
            if any(inside(entry.path,p) for p in paths):continue
            refs=[ref for ref in entry.references if any(related(Path(ref),p) for p in paths)]
            if refs:found.append(dict(path=str(entry.path),title=entry.title,references=refs))
        return found

    def delete(self,paths,progress=lambda _:None):
        # Call only after explicit UI confirmation. No automatic cleanup policy.
        requested=self.normalized(paths)
        if not requested:raise ValueError('尚未选择数据')
        with DataAccess(exclusive=True,runtime=self.runtime):
            live=active_paths()
            for path in requested:
                if path not in self.entries:raise ValueError('目录不属于本次历史数据索引')
                entry=self.entries[path]
                if path.resolve()!=path or path.is_symlink():raise ValueError('目录已变成符号链接，拒绝删除')
                if not path.is_dir():raise ValueError('目录已经不存在，请刷新列表')
                s=path.stat()
                if (s.st_dev,s.st_ino)!=entry.identity:raise ValueError('目录已被替换，请刷新后重新选择')
                if path==self.root or not any(inside(path,scope) for scope in self.scopes):
                    raise ValueError('目录超出已识别的数据范围')
                if any(related(path,p) for p,_ in live):raise ValueError('目录正被任务使用，请先停止任务')
                if entry.errors:raise ValueError('扫描存在错误，请先处理：'+entry.errors[0])
            # Rescan while new cooperative writers are locked out; recompute hardlinks.
            fresh=Catalog(self.root,self.extra_roots).scan(progress)
            for path in requested:
                new=fresh.entries.get(path)
                if not new or new.identity!=self.entries[path].identity or new.errors:
                    raise ValueError('目录已变化或含无法扫描的内容，请刷新列表')
            before=fresh.selected(requested)
            devices={}
            for path in requested:
                s=path.stat();v=os.statvfs(path)
                devices.setdefault(s.st_dev,dict(path=str(path.parent),free=v.f_bavail*v.f_frsize))
            deleted=[];failed=[]
            for path in requested:
                progress('删除 '+path.name)
                try:
                    shutil.rmtree(path)
                    deleted.append(str(path))
                except OSError as exc:failed.append(dict(path=str(path),message=str(exc)))
            changes=[]
            for device,value in devices.items():
                v=os.statvfs(value['path']);free=v.f_bavail*v.f_frsize
                changes.append(dict(device=device,before=value['free'],after=free,delta=free-value['free']))
            report=dict(timestamp=time.time(),selected=before,deleted=deleted,failed=failed,
                        volume_changes=changes,observed_free_delta=sum(v['delta'] for v in changes))
            self.runtime.mkdir(parents=True,exist_ok=True)
            journal=self.runtime/'history_deletions.jsonl'
            with journal.open('a') as handle:handle.write(json.dumps(report,ensure_ascii=False)+'\n')
            return report

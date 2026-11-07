"""Dataset IO and conservative BundleSDF-ready export; no models or camera."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import time

import cv2
import numpy as np


def stamp():
    return time.strftime('%Y%m%d_%H%M%S') + f'_{time.time_ns()%1000000:06d}'


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def atomic_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + '.tmp' + path.suffix)
    if not cv2.imwrite(str(temp), image):
        raise IOError(f'图像写入失败：{path}')
    os.replace(temp, path)


def rgb_files(dataset):
    files = sorted(p for p in (Path(dataset)/'rgb').iterdir()
                   if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg'})
    if not files or len({p.stem for p in files}) != len(files):
        raise ValueError('RGB 目录为空或存在重复帧名')
    return files


def copy_file(src, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def prompt_digest(prompts):
    return hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest()


def import_rgb(source, target, limit=None):
    source, target = Path(source).resolve(), Path(target).resolve()
    if target.exists():
        raise ValueError('工作副本目录已存在，请选择新目录')
    if target == source or target in source.parents or source in target.parents:
        raise ValueError('工作副本必须与原目录分开')
    files = sorted(p for p in source.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg'})
    if limit:
        files = files[:limit]
    if not files or len({p.stem for p in files}) != len(files):
        raise ValueError('未找到图像，或文件名去扩展后重复')
    target.mkdir(parents=True)
    (target/'rgb').mkdir()
    (target/'depth').mkdir()
    rows = []
    for i, p in enumerate(files):
        name = p.stem+'.png'
        if p.suffix.lower()=='.png':
            copy_file(p, target/'rgb'/name)
        else:
            im=cv2.imread(str(p))
            if im is None:
                raise ValueError(f'图像无法读取：{p}')
            atomic_image(target/'rgb'/name, im)
        depth=source.parent/'depth'/name
        if depth.exists():
            copy_file(depth,target/'depth'/name)
        rows.append(dict(index=i,file=name,source=str(p)))
    for name in ['cam_K.txt']:
        src=source.parent/name
        if src.exists():
            copy_file(src,target/name)
    # Preserve original timing/calibration metadata when supplied by our recorder.
    for name in ['capture.json','frames.jsonl']:
        src=source.parent/'metadata'/name
        if src.exists():
            copy_file(src,target/'metadata'/('source_'+name))
    atomic_json(target/'metadata/import.json',dict(source=str(source),frames=rows,mask_source='SAM3 pending'))
    return dict(dataset=str(target),frames=len(files),has_intrinsics=(target/'cam_K.txt').exists())


def validate(dataset, max_depth_m=1.0, export_to=None, event=lambda **kw: None):
    root=Path(dataset).resolve()
    files=rgb_files(root)
    K=np.loadtxt(root/'cam_K.txt').reshape(3,3)
    if not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0 or not np.allclose(K[2],[0,0,1]):
        raise ValueError('cam_K.txt 内参不合法')
    prompts_path=root/'metadata/prompts.json'
    status_path=root/'metadata/segmentation.json'
    if prompts_path.exists():
        status=json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get('prompts_sha256') != prompt_digest(json.loads(prompts_path.read_text())):
            raise ValueError('提示点已更改，需完成重新分割后再导出')
    from segment_plan import read_plan,plan_signature
    plan=read_plan(root)
    if plan.get('enabled'):
        status=json.loads(status_path.read_text()) if status_path.exists() else {}
        prompts=json.loads(prompts_path.read_text()) if prompts_path.exists() else {}
        if status.get('plan_sha256')!=plan_signature(files,prompts,plan):
            raise ValueError('分段计划或提示点已变化，需完成全部分段并发布 masks 后再导出')
    shape=None; rows=[]; runs=[]; current=[]; issues=[]
    expected={p.stem+'.png' for p in files}
    for folder in ['depth','masks']:
        if not (root/folder).exists():
            raise ValueError(f'缺少 {folder} 目录')
        actual={p.name for p in (root/folder).glob('*.png')}
        if actual-expected:
            issues.append(f'{folder} 存在 {len(actual-expected)} 个多余文件')
    previous_ns=None
    for i,p in enumerate(files):
        rgb=cv2.imread(str(p)); name=p.stem+'.png'
        depth=cv2.imread(str(root/'depth'/name),-1)
        mask=cv2.imread(str(root/'masks'/name),-1)
        reason=[]
        if rgb is None: reason.append('RGB 无法读取')
        elif shape is None: shape=rgb.shape[:2]
        elif rgb.shape[:2]!=shape: reason.append('RGB 分辨率变化')
        if depth is None or depth.dtype!=np.uint16 or depth.ndim!=2 or depth.shape!=shape:
            reason.append('深度缺失、类型或尺寸不符')
        if mask is None or mask.dtype!=np.uint8 or mask.ndim!=2 or mask.shape!=shape:
            reason.append('掩码缺失、类型或尺寸不符')
        elif not np.isin(mask,[0,255]).all():
            reason.append('掩码必须为 0/255')
        valid_count=0; ratio=0.; area=0
        if not reason:
            eroded=cv2.erode(mask,np.ones((3,3),np.uint8))>0
            area=int(eroded.sum())
            valid=(depth>=100)&(depth<max_depth_m*1000)&eroded
            valid_count=int(valid.sum());ratio=valid_count/max(area,1)
            if area<100: reason.append('目标为空或腐蚀后不足 100 像素')
            if valid_count<100: reason.append('目标内有效深度不足 100 像素')
        # Timestamp-named datasets split across large gaps; simple frame counters do not imply nanoseconds.
        ns=int(p.stem) if p.stem.isdigit() and len(p.stem)>=16 else None
        gap=ns is not None and previous_ns is not None and (ns-previous_ns)>250_000_000
        if gap and current:
            runs.append(current);current=[]
        previous_ns=ns
        row=dict(index=i,file=name,usable=not reason,reasons=reason,mask_pixels_after_erosion=area,
                 valid_depth_pixels=valid_count,valid_depth_ratio=ratio,large_gap=gap)
        rows.append(row)
        if reason:
            if current: runs.append(current);current=[]
        else: current.append(name)
        if i%100==0: event(type='progress',index=i,total=len(files),message='检查数据完整性')
    if current:runs.append(current)
    report=dict(dataset=str(root),total_frames=len(files),usable_frames=sum(r['usable'] for r in rows),
                image_size=list(shape[::-1]) if shape else None,max_depth_m=max_depth_m,
                segments=[dict(frames=len(r),first=r[0],last=r[-1]) for r in runs],
                issues=issues,frames=rows,depth_unit='millimeter',
                note='导出检查不等于重建质量保证；长遮挡或无效帧切分为连续片段。')
    atomic_json(root/'metadata/validation.json',report)
    exported=[]
    if export_to:
        base=Path(export_to).resolve()
        if base.exists(): raise ValueError('导出目录已存在，禁止覆盖')
        useful=[r for r in runs if len(r)>=5]
        if not useful: raise ValueError('没有至少 5 帧的有效连续片段，请检查掩码和深度')
        base.mkdir(parents=True)
        for j,run in enumerate(useful,1):
            dest=base/f'segment_{j:03d}'
            for sub in ['rgb','depth','masks']:(dest/sub).mkdir(parents=True,exist_ok=True)
            for name in run:
                rgb=next(p for p in files if p.stem==Path(name).stem)
                copy_file(rgb,dest/'rgb'/name)
                for sub in ['depth','masks']:copy_file(root/sub/name,dest/sub/name)
            copy_file(root/'cam_K.txt',dest/'cam_K.txt')
            atomic_json(dest/'metadata/export.json',dict(source=str(root),frames=run,max_depth_m=max_depth_m,
                        contract='RGB PNG; aligned uint16 millimeter depth; uint8 0/255 masks; output-image K'))
            exported.append(str(dest))
        report['exported']=exported
        atomic_json(base/'export_report.json',report)
    return dict(report=str(root/'metadata/validation.json'),frames=len(files),usable=report['usable_frames'],
                segments=len(runs),exported=exported)

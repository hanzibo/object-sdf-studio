"""Initialization-mask provenance and RGB-D checks; no model imports."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from data_ops import atomic_json, rgb_files


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_rgbd(dataset, start=0, count=0):
    root = Path(dataset).resolve()
    files = rgb_files(root)
    if not 0 <= start < len(files):
        raise ValueError('起始帧超出序列范围')
    if any(p.suffix != '.png' for p in files):
        raise ValueError('请先通过 Studio 导入为 PNG 工作副本')
    K = np.loadtxt(root / 'cam_K.txt').reshape(3, 3)
    if not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0 or not np.allclose(K[2], [0, 0, 1]):
        raise ValueError('相机内参无效')
    selected = files[start:start+count] if count else files[start:]
    shape = None
    for p in selected:
        rgb = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(str(root / 'depth' / p.name), cv2.IMREAD_UNCHANGED)
        if rgb is None or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
            raise ValueError(f'RGB 不是 uint8 彩色图：{p.name}')
        if depth is None or depth.dtype != np.uint16 or depth.ndim != 2 or depth.shape != rgb.shape[:2]:
            raise ValueError(f'深度必须是同尺寸的 uint16 毫米 PNG：{p.name}')
        if shape is not None and shape != depth.shape:
            raise ValueError('序列中的图像尺寸发生变化')
        shape = depth.shape
    return files, K, shape


def validate_init(metadata, dataset, frame_index, hints=None, require_confirmed=True):
    metadata = Path(metadata).resolve()
    value = json.loads(metadata.read_text())
    root = Path(dataset).resolve()
    files = rgb_files(root)
    if value.get('dataset') != str(root) or value.get('frame_index') != frame_index:
        raise ValueError('初始化掩码不属于当前数据集或起始帧，请重新分割')
    source = files[frame_index]
    if value.get('rgb_sha256') != digest(source):
        raise ValueError('初始化源图已更改，请重新分割')
    if hints is not None and value.get('points') != hints:
        raise ValueError('提示点已更改，请重新分割并确认')
    if require_confirmed and not value.get('confirmed'):
        raise ValueError('请先确认初始化掩码')
    mask_path = metadata.parent / value['mask']
    if value.get('mask_sha256') != digest(mask_path):
        raise ValueError('初始化掩码文件已更改，请重新生成')
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(str(root / 'depth' / source.name), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.dtype != np.uint8 or mask.ndim != 2 or depth is None or mask.shape != depth.shape:
        raise ValueError('初始化掩码尺寸或格式错误')
    if not np.isin(mask, [0, 255]).all() or np.count_nonzero((mask > 0) & (depth > 1)) < 4:
        raise ValueError('初始化掩码没有足够的有效前景深度')
    return value, mask_path


def confirm_init(metadata, dataset, frame_index, hints):
    value, mask = validate_init(metadata, dataset, frame_index, hints, require_confirmed=False)
    value['confirmed'] = True
    atomic_json(metadata, value)
    return mask

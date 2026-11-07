"""Render saved tracking poses over RGB images; no tracking or training."""
import argparse
import itertools
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out_folder', type=Path, required=True)
    ap.add_argument('--video_dir', type=Path, required=True)
    ap.add_argument('--preview_only', action='store_true')
    ap.add_argument('--dest', type=Path, required=True)
    args = ap.parse_args()
    root, source = args.out_folder.resolve(), args.video_dir.resolve()
    dest = args.dest.resolve()
    frames = dest / 'frames'
    frames.mkdir(parents=True, exist_ok=True)
    pose_files = sorted((root / 'ob_in_cam').glob('*.txt'))
    if len(pose_files) < 2:
        raise ValueError('Need at least two poses')
    stamps = [int(p.stem) for p in pose_files]
    times = (np.array(stamps, dtype=np.int64) - stamps[0]) / 1e9
    durations = np.r_[np.diff(times), np.median(np.diff(times))]
    if not (durations > 0).all():
        raise ValueError('Frame timestamps must increase')
    mesh_path = root / 'textured_mesh.obj'
    mesh = trimesh.load(mesh_path, force='mesh')
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    K = np.loadtxt(root / 'cam_K.txt').reshape(3, 3)
    corners = np.array(list(itertools.product([-1, 1], repeat=3))) * extents / 2
    edges = [(a, b) for a in range(8) for b in range(a+1, 8)
             if np.count_nonzero(corners[a] != corners[b]) == 1]
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font = ImageFont.truetype(font_path, 23)
    small = ImageFont.truetype(font_path, 19)
    n = len(pose_files)
    if len(list((source/'rgb').glob('*.png'))) != n:
        raise ValueError('Pose count and source RGB count differ')
    samples = [0, n//3, 2*n//3, n-1]
    selected = sorted(set(list(range(min(90, n))) + samples)) if args.preview_only else range(n)
    for i in selected:
        image = cv2.imread(str(source / 'rgb' / f'{pose_files[i].stem}.png'))
        if image is None:
            raise FileNotFoundError(pose_files[i].stem)
        h, w = image.shape[:2]
        pose = np.loadtxt(pose_files[i]).reshape(4, 4) @ np.linalg.inv(to_origin)
        if not np.isfinite(pose).all():
            raise ValueError(f'Invalid pose: {pose_files[i]}')
        overlay = image.copy()

        def project(points):
            xyz = points @ pose[:3, :3].T + pose[:3, 3]
            if (xyz[:, 2] <= 0).any():
                raise ValueError(f'Overlay behind camera at frame {i}')
            uv = xyz @ K.T
            return np.round(uv[:, :2] / uv[:, 2:]).astype(int)

        uv = project(corners)
        for a, b in edges:
            cv2.line(overlay, tuple(uv[a]), tuple(uv[b]), (0, 220, 255), 2, cv2.LINE_AA)
        axis = project(np.vstack([np.zeros(3), np.eye(3)*min(extents)*0.55]))
        for j, (label, color) in enumerate([('X',(30,30,255)), ('Y',(40,230,40)), ('Z',(255,130,30))], 1):
            cv2.arrowedLine(overlay, tuple(axis[0]), tuple(axis[j]), color, 3, cv2.LINE_AA, tipLength=0.12)
            cv2.putText(overlay, label, tuple(axis[j]+[5,-5]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        canvas = np.full((h+104, w*2, 3), (27,23,18), dtype=np.uint8)
        canvas[64:64+h,:w] = image
        canvas[64:64+h,w:] = overlay
        pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        draw.text((18,14), '原始 RGB 图像', font=font, fill='#edf2f8')
        draw.text((w+18,14), 'BundleSDF 位姿跟踪 · 三维包围盒 / 坐标轴', font=font, fill='#edf2f8')
        draw.text((18,h+70), f'帧 {i+1:04d} / {n}    时间 {times[i]:05.2f} 秒', font=small, fill='#d0d8e4')
        draw.text((w+18,h+70), '已有位姿离线回放 · 不代表实时运行速度', font=small, fill='#aab8cc')
        cv2.imwrite(str(frames/f'{i:06d}.png'), cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR))
        if i % 200 == 0:
            print(f'Rendered {i+1}/{n}', flush=True)
    thumbs = []
    for i in samples:
        with Image.open(frames / f'{i:06d}.png') as im:
            thumbs.append(im.resize((640,292)))
    sheet = Image.new('RGB', (1280,584))
    for j, im in enumerate(thumbs):
        sheet.paste(im, ((j%2)*640,(j//2)*292))
    sheet.save(dest/'contact_sheet.png')
    count = min(90,n) if args.preview_only else n
    manifest = dest / ('preview.ffconcat' if args.preview_only else 'tracking.ffconcat')
    lines = ['ffconcat version 1.0']
    for i in range(count):
        lines.extend([f"file 'frames/{i:06d}.png'", 'option framerate 1000', f'duration {durations[i]:.9f}'])
    lines.extend([f"file 'frames/{count-1:06d}.png'", 'option framerate 1000'])
    manifest.write_text('\n'.join(lines)+'\n')
    video = dest / ('preview.mp4' if args.preview_only else 'pose_tracking.mp4')
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','warning','-safe','0','-f','concat','-i',str(manifest),
                    '-t',str(float(durations[:count].sum())), '-an','-c:v','libx264','-preset','medium',
                    '-crf','19','-pix_fmt','yuv420p','-fps_mode','vfr','-movflags','+faststart',str(video)],check=True)
    report = dict(source_rgb=str(source/'rgb'), poses=str(root/'ob_in_cam'), mesh=str(mesh_path),
                  frames=count, total_saved_poses=n, duration_seconds=float(durations[:count].sum()),
                  timing='Original nanosecond filename timestamps; variable frame rate',
                  layout='Original RGB left; OBB and axes overlay right; no audio',
                  box_extents_m=extents.tolist(), mesh_to_box=to_origin.tolist(), output=str(video))
    (dest / ('preview.json' if args.preview_only else 'edit_decisions.json')).write_text(
        json.dumps(report,indent=2,ensure_ascii=False))
    print(f'EXPORTED {video}',flush=True)


if __name__ == '__main__':
    main()

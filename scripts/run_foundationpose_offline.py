"""Offline adapter: metric BundleSDF mesh -> FoundationPose RGB-D tracking."""
import argparse
import csv
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
import threading
import signal
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import POSE_PROJECT
EVENTS = False
OUTPUT = None


def event(**value):
    if EVENTS:
        print('EVENT '+json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


class Controls:
    def __init__(self, enabled):
        self.stop = threading.Event()
        self.pause = threading.Event()
        self.paused_s = 0.
        if enabled:
            threading.Thread(target=self.read, daemon=True).start()
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, lambda *_: self.stop.set())

    def read(self):
        for line in sys.stdin:
            command = line.strip()
            if command == 'stop': self.stop.set()
            elif command == 'pause': self.pause.set()
            elif command == 'resume': self.pause.clear()
        # A closed supervisor pipe must not leave a paused GPU worker behind.
        self.stop.set()

    def wait(self, frame_index):
        if self.pause.is_set() and not self.stop.is_set():
            event(type='paused',frame_index=frame_index,message='已暂停，可继续跟踪或停止进程后重新点选')
            started = time.perf_counter()
            while self.pause.is_set() and not self.stop.wait(.05):
                pass
            self.paused_s += time.perf_counter()-started
            if not self.stop.is_set(): event(type='resumed',message='继续跟踪')
        return self.stop.is_set()


def main():
    global EVENTS, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--foundationpose-dir', type=Path, default=POSE_PROJECT)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=150)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--preview', action='store_true', help='Show live overlay; Q/Esc or closing window finishes early')
    parser.add_argument('--events', action='store_true', help='Studio JSON event protocol')
    parser.add_argument('--controls', action='store_true', help='Read pause/resume/stop from stdin')
    parser.add_argument('--start-frame', type=int, default=0, help='Zero-based source frame index')
    parser.add_argument('--init-metadata', type=Path, help='Confirmed Studio initialization JSON')
    parser.add_argument('--parent-run', type=Path, help='Previous run before manual reinitialization')
    args = parser.parse_args()
    EVENTS = args.events
    if args.frames < 0 or args.fps <= 0 or args.start_frame < 0:
        parser.error('frames/start-frame must be nonnegative; fps must be positive (frames=0 means all)')
    from pose_data import check_rgbd, validate_init
    from data_ops import atomic_json
    event(type='status',message='检查 RGB-D、内参和初始化掩码…')
    files, _, _ = check_rgbd(args.dataset, args.start_frame, args.frames)
    init_mask_path = None
    if args.init_metadata:
        _, init_mask_path = validate_init(args.init_metadata, args.dataset, args.start_frame)
    elif not (args.dataset/'masks'/files[args.start_frame].name).is_file():
        raise ValueError('请选择并确认初始化掩码')
    # Refuse reuse: neither source datasets nor previous results can be overwritten.
    args.output.mkdir(parents=True, exist_ok=False)
    OUTPUT = args.output
    atomic_json(args.output/'job.json',dict(status='loading',dataset=str(args.dataset.resolve()),
                mesh=str(args.mesh.resolve()),start_frame=args.start_frame,
                init_metadata=str(args.init_metadata) if args.init_metadata else None,
                parent_run=str(args.parent_run) if args.parent_run else None))
    for name in ['ob_in_cam', 'frames']:
        (args.output / name).mkdir()
    sys.path.insert(0, str(args.foundationpose_dir.resolve()))
    import cv2
    import numpy as np
    import torch
    import trimesh
    import nvdiffrast.torch as dr
    from estimater import FoundationPose
    from datareader import YcbineoatReader
    from Utils import (set_seed, make_mesh_tensors, nvdiffrast_render,
                       draw_posed_3d_box, draw_xyz_axis)

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)
    set_seed(0)
    started = time.perf_counter()
    # Avoid upstream whole-path rgb->depth replacement for arbitrary user paths.
    class Reader(YcbineoatReader):
        def get_depth(self, i):
            return cv2.imread(str(args.dataset/'depth'/Path(self.color_files[i]).name),-1).astype(np.float32)/1000.
        def get_mask(self, i):
            return cv2.imread(str(args.dataset/'masks'/Path(self.color_files[i]).name),-1)>0
    reader = Reader(str(args.dataset), shorter_side=None, zfar=np.inf)
    remaining = len(reader)-args.start_frame
    n = min(args.frames, remaining) if args.frames else remaining
    requested_frames = n
    if n == 0:
        raise ValueError('No RGB frames')
    for path in reader.color_files[args.start_frame:args.start_frame+n]:
        if not (args.dataset / 'depth' / Path(path).name).is_file():
            raise ValueError(f'Missing paired depth: {path}')
    event(type='status',message='加载 FoundationPose 模型与 GPU 渲染器…')
    mesh = trimesh.load(args.mesh)
    if not isinstance(mesh,trimesh.Trimesh) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError('模型必须是有效的单个三角网格')
    from pose_data import digest
    texture=getattr(getattr(mesh.visual,'material',None),'image',None)
    atomic_json(args.output/'asset.json',dict(mesh=str(args.mesh.resolve()),mesh_sha256=digest(args.mesh),
                units='meters',frame='original OBJ mesh coordinates',vertices=len(mesh.vertices),
                faces=len(mesh.faces),extents_m=mesh.extents.tolist(),visual=mesh.visual.kind,
                texture_size=list(texture.size) if texture is not None else None))
    origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2])
    context = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                         mesh=mesh, glctx=context, debug=0,
                         debug_dir=str(args.output / 'estimator_debug'))
    # Estimator uses a centered mesh internally; full-frame evaluation uses original mesh.
    original_tensors = make_mesh_tensors(mesh)
    torch.cuda.synchronize()
    setup_s = time.perf_counter() - started
    rows = []
    controls = Controls(args.controls)
    window = 'FoundationPose | RGB + tracked mesh | Q / Esc to finish'
    if args.preview:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1280, 522)
    pipeline_start = time.perf_counter()
    with (args.output / 'metrics.csv').open('w', newline='') as handle:
        writer = None
        for i in range(n):
            source_i = args.start_frame+i
            if controls.wait(max(args.start_frame,source_i-1)):
                break
            frame_start = time.perf_counter()
            rgb, depth = reader.get_color(source_i), reader.get_depth(source_i)
            torch.cuda.synchronize()
            inference_start = time.perf_counter()
            with torch.inference_mode():
                if i == 0:
                    event(type='status',message=f'初始化第 {source_i+1} 帧位姿…')
                    mask = cv2.imread(str(init_mask_path),-1)>0 if init_mask_path else reader.get_mask(source_i).astype(bool)
                    if ((depth > .001) & mask).sum() < 4:
                        raise ValueError('Insufficient valid foreground depth for registration')
                    pose = est.register(K=reader.K, rgb=rgb, depth=depth,
                                        ob_mask=mask, iteration=5)
                else:
                    pose = est.track_one(rgb=rgb, depth=depth, K=reader.K, iteration=2)
            torch.cuda.synchronize()
            inference_s = time.perf_counter() - inference_start
            if not np.isfinite(pose).all():
                raise ValueError(f'Nonfinite pose at frame {i}')
            np.savetxt(args.output / 'ob_in_cam' / f'{reader.id_strs[source_i]}.txt', pose)
            with torch.inference_mode():
                rendered, rendered_depth, _ = nvdiffrast_render(
                    K=reader.K, H=reader.H, W=reader.W,
                    ob_in_cams=torch.as_tensor(pose, device='cuda', dtype=torch.float)[None],
                    glctx=context, mesh_tensors=original_tensors)
            render_rgb = (rendered[0].cpu().numpy() * 255).astype(np.uint8)
            render_depth = rendered_depth[0].cpu().numpy()
            silhouette = render_depth > .001
            vis = rgb.copy()
            vis[silhouette] = (.55 * vis[silhouette] + .45 * render_rgb[silhouette]).astype(np.uint8)
            center_pose = pose @ np.linalg.inv(origin)
            vis = draw_posed_3d_box(reader.K, img=vis, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=.04, K=reader.K,
                                thickness=2, transparency=0, is_input_rgb=True)
            # SAM masks are evaluation-only after frame 0, never fed into track_one.
            mask_path = args.dataset / 'masks' / Path(reader.color_files[source_i]).name
            iou, residual_mm = None, None
            if mask_path.is_file():
                target = reader.get_mask(source_i).astype(bool)
                intersection = target & silhouette
                union = target | silhouette
                iou = float(intersection.sum() / max(1, union.sum()))
                valid = intersection & (depth > .001)
                if valid.any():
                    residual_mm = float(np.median(np.abs(depth[valid] - render_depth[valid])) * 1000)
                contours, _ = cv2.findContours(target.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, (255, 255, 0), 1)
            panel = np.concatenate([rgb, vis], axis=1)
            panel = cv2.copyMakeBorder(panel, 42, 0, 0, 0, cv2.BORDER_CONSTANT, value=(18, 24, 35))
            label = f'RGB | FoundationPose mesh + axes | source {source_i+1}/{len(reader)} | infer {inference_s*1000:.1f} ms'
            cv2.putText(panel, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .60, (230, 240, 245), 1, cv2.LINE_AA)
            cv2.imwrite(str(args.output / 'frames' / f'{i:06d}.png'), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            stop_requested = False
            if args.preview:
                cv2.imshow(window, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xff
                stop_requested = key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
            row = dict(frame=i, source_frame=source_i, frame_id=reader.id_strs[source_i], kind='register' if i==0 else 'track',
                       inference_s=inference_s, frame_s=time.perf_counter()-frame_start,
                       mask_iou=iou, median_depth_residual_mm=residual_mm)
            rows.append(row)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=row.keys())
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            event(type='progress',index=i+1,total=n,frame_index=source_i,
                  inference_ms=inference_s*1000,image=str(args.output/'frames'/f'{i:06d}.png'),
                  message=f'跟踪 {i+1}/{n} · 原序列第 {source_i+1} 帧 · {inference_s*1000:.1f} ms')
            atomic_json(args.output/'progress.json',dict(completed=i+1,total=n,frame_index=source_i,
                        pose_file=str(args.output/'ob_in_cam'/f'{reader.id_strs[source_i]}.txt')))
            if i == 0 or (i+1) % 10 == 0:
                print(json.dumps(row), flush=True)
            if stop_requested:
                print('Preview stopped by user; saving completed frames and video.', flush=True)
                break
    pipeline_s = time.perf_counter() - pipeline_start
    if args.preview:
        cv2.destroyAllWindows()
    n = len(rows)
    if not n:
        atomic_json(args.output/'job.json',dict(status='canceled',frames=0))
        event(type='done',status='canceled',output=str(args.output),message='已停止，尚未处理图像')
        return
    tracks = np.array([x['inference_s'] for x in rows[1:]])
    warmed = tracks[5:] if len(tracks) > 5 else tracks
    ious = [r['mask_iou'] for r in rows if r['mask_iou'] is not None]
    residuals = [r['median_depth_residual_mm'] for r in rows if r['median_depth_residual_mm'] is not None]
    summary = dict(dataset=str(args.dataset), mesh=str(args.mesh), frames=n,
                   start_frame=args.start_frame,last_source_frame=rows[-1]['source_frame'],paused_seconds=controls.paused_s,
                   parent_run=str(args.parent_run) if args.parent_run else None,
                   init_metadata=str(args.init_metadata) if args.init_metadata else None,
                   requested_frames=requested_frames, stopped_early=n < requested_frames,
                   dataset_frames=len(reader), initialization_s=rows[0]['inference_s'], setup_s=setup_s,
                   tracking_mean_ms=float(tracks.mean()*1000) if len(tracks) else None,
                   tracking_median_ms=float(np.median(tracks)*1000) if len(tracks) else None,
                   tracking_p95_ms=float(np.percentile(tracks,95)*1000) if len(tracks) else None,
                   tracking_throughput_fps=float(1/tracks.mean()) if len(tracks) else None,
                   warm_tracking_mean_ms=float(warmed.mean()*1000) if len(warmed) else None,
                   pipeline_s=pipeline_s, pipeline_fps=n/pipeline_s,
                   peak_torch_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                   peak_torch_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                   mask_iou_median=float(np.median(ious)) if ious else None,
                   median_depth_residual_mm=float(np.median(residuals)) if residuals else None,
                   pose_convention='OpenCV camera_from_original_mesh, translation in meters',
                   evaluation='SAM masks only measure agreement; no ground-truth poses available',
                   settings=dict(register_iterations=5, tracking_iterations=2, stride=1, seed=0,
                                 rgb_size=[reader.W,reader.H], visualization_fps=args.fps),
                   torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0))
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    samples = np.linspace(0, n-1, min(6,n)).astype(int)
    images = [cv2.imread(str(args.output/'frames'/f'{j:06d}.png')) for j in samples]
    thumbs = [cv2.resize(im, (640,261)) for im in images]
    while len(thumbs)%2:
        thumbs.append(np.zeros_like(thumbs[0]))
    sheet = np.concatenate([np.concatenate(thumbs[j:j+2],axis=1) for j in range(0,len(thumbs),2)],axis=0)
    cv2.imwrite(str(args.output/'contact_sheet.jpg'),sheet)
    event(type='status',message='保存已完成位姿，正在生成演示视频…')
    subprocess.run(['ffmpeg','-v','error','-n','-framerate',str(args.fps),'-i',
                    str(args.output/'frames'/'%06d.png'),'-c:v','libx264','-crf','18',
                    '-pix_fmt','yuv420p','-movflags','+faststart',str(args.output/'tracking.mp4')],check=True)
    print('COMPLETE '+json.dumps(summary,ensure_ascii=False),flush=True)
    atomic_json(args.output/'job.json',dict(status='stopped' if n<requested_frames else 'complete',**summary))
    event(type='done',status='stopped' if n<requested_frames else 'complete',output=str(args.output),
          frames=n,image=str(args.output/'frames'/f'{n-1:06d}.png'),message=f'已保存 {n} 帧位姿及视频：{args.output}')


if __name__ == '__main__':
    from gpu_lock import GpuLease
    from data_lock import DataAccess
    try:
        with DataAccess(), GpuLease('FoundationPose tracking'):
            main()
    except Exception as exc:
        if '--events' in sys.argv:
            EVENTS = True
            event(type='error',message=str(exc))
        if OUTPUT:
            from data_ops import atomic_json
            atomic_json(OUTPUT/'error.json',dict(message=str(exc),traceback=traceback.format_exc()))
            job_path=OUTPUT/'job.json'
            report=json.loads(job_path.read_text()) if job_path.exists() else {}
            atomic_json(job_path,dict(report,status='failed',message=str(exc)))
        traceback.print_exc()
        sys.exit(1)

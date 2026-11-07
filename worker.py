"""Isolated camera/model jobs. Protocol: EVENT {json} on stdout; stop on stdin."""
import argparse, json, sys, threading, traceback
from collections import deque
from pathlib import Path
from config import *
from data_ops import *
from segment_plan import read_plan,ranges,signature,plan_signature,cache_file,cached_result
STOP=threading.Event()
PREVIEW_FPS=10
def event(**kw): print('EVENT '+json.dumps(kw,ensure_ascii=False),flush=True)
def controls():
    global PREVIEW_FPS
    for line in sys.stdin:
        parts=line.strip().split()
        if parts==['stop']:STOP.set()
        elif len(parts)==2 and parts[0]=='fps' and parts[1] in ['10','20','30']:PREVIEW_FPS=int(parts[1])

def pose_init(root, frame_index, prompts_file=None):
    """SAM3 at one chosen frame; never publishes training masks."""
    from pose_data import check_rgbd, digest
    root = Path(root).resolve()
    files, _, _ = check_rgbd(root, frame_index, 1)
    p = files[frame_index]
    hints = json.loads(Path(prompts_file or root/'metadata/prompts.json').read_text()).get(p.name, [])
    image = cv2.imread(str(p)); h, w = image.shape[:2]
    if not any(pt[2] == 1 for pt in hints):
        raise ValueError('请先在当前起始帧添加前景点')
    if any(not (0 <= pt[0] < w and 0 <= pt[1] < h and pt[2] in [0, 1]) for pt in hints):
        raise ValueError('提示点越界')
    event(type='progress',message='加载 SAM3，分割当前初始化帧…')
    import torch
    from integrations.sam3_tracker import OnlineTracker
    tracker = OnlineTracker(history=32)
    if STOP.is_set():event(type='canceled',message='初始化分割已取消');return
    tracker.state = tracker.model.init_state(video_height=h,video_width=w,num_frames=1,offload_state_to_cpu=True)
    tracker.index = 0
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        tracker._features(image,0)
        tracker.model.add_new_points_or_box(tracker.state,frame_idx=0,obj_id=1,points=np.array([v[:2] for v in hints],dtype=np.float32)/[w,h],labels=[v[2] for v in hints],rel_coordinates=True)
        tracker.model.propagate_in_video_preflight(tracker.state)
        mask,score = tracker._propagate(0)
    if STOP.is_set():event(type='canceled',message='初始化分割已取消');return
    folder = root/'metadata/pose_initializations'/stamp()
    atomic_image(folder/'mask.png', mask.astype(np.uint8)*255)
    overlay = image.copy();overlay[mask]=(overlay[mask]*.55+np.array([60,220,60])*.45).astype(np.uint8)
    atomic_image(folder/'preview.jpg', overlay)
    atomic_json(folder/'init.json',dict(dataset=str(root),frame_index=frame_index,frame_name=p.name,
                rgb_sha256=digest(p),points=hints,mask='mask.png',mask_sha256=digest(folder/'mask.png'),
                confirmed=False,pixels=int(mask.sum()),score=float(score)))
    event(type='done',init_metadata=str(folder/'init.json'),image=str(folder/'preview.jpg'),
          message=f'第 {frame_index+1} 帧分割完成，前景 {mask.sum()} 像素；请检查后点击确认')
def segment(root,preview=False,selected=None,force=False):
    import torch
    from integrations.sam3_tracker import OnlineTracker
    root=Path(root);files=rgb_files(root)
    prompts=json.loads((root/'metadata/prompts.json').read_text())
    plan=read_plan(root);segmented=plan.get('enabled',False)
    blocks=ranges(files,plan) if segmented else [(0,len(files))]
    if selected is not None and not 0<=selected<len(blocks):raise ValueError('所选段不存在')
    wanted=[selected] if selected is not None else list(range(len(blocks)))
    for j in wanted:
        start,end=blocks[j]
        if not any(v[2]==1 for v in prompts.get(files[start].name,[])):
            raise ValueError(f'第 {j+1} 段起始帧 {start+1} 尚未添加前景点')
    run=root/'metadata/segmentation_runs'/stamp();run.mkdir(parents=True)
    atomic_json(run/'prompts.json',prompts)
    tracker=None;completed=0
    for j in wanted:
        start,end=blocks[j]
        if STOP.is_set():event(type='canceled',message='已取消；已完成段已缓存，下次可继续');return
        if segmented and not preview and not force and cached_result(root,files,prompts,start,end):
            completed+=end-start
            event(type='progress',index=completed,total=sum(blocks[k][1]-blocks[k][0] for k in wanted),message=f'第 {j+1} 段已完成，复用缓存')
            continue
        if tracker is None:
            event(type='progress',message='加载 SAM3 权重',index=0,total=len(files))
            tracker=OnlineTracker(history=32)
        folder=run/f'segment_{j+1:04d}';(folder/'masks').mkdir(parents=True)
        rows=[]
        for i in range(start,end):
            if STOP.is_set():event(type='canceled',message='已取消；当前未完成段需重跑，已完成段可继续复用');return
            p=files[i];image=cv2.imread(str(p))
            if image is None:raise ValueError(f'图像无法读取：{p}')
            h,w=image.shape[:2];hints=prompts.get(p.name)
            if hints:
                if not any(pt[2]==1 for pt in hints):raise ValueError(f'{p.name} 的校正点至少需一个前景点')
                if any(not(0<=pt[0]<w and 0<=pt[1]<h and pt[2] in [0,1]) for pt in hints):raise ValueError('提示点越界')
                # Reset memory at each segment start and at additional correction frames.
                tracker.state=None
                tracker.state=tracker.model.init_state(video_height=h,video_width=w,num_frames=1,offload_state_to_cpu=True);tracker.index=0
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    tracker._features(image,0)
                    tracker.model.add_new_points_or_box(tracker.state,frame_idx=0,obj_id=1,points=np.array([v[:2] for v in hints],dtype=np.float32)/[w,h],labels=[v[2] for v in hints],rel_coordinates=True)
                    tracker.model.propagate_in_video_preflight(tracker.state)
                    mask,score=tracker._propagate(0)
            else:mask,score=tracker.step(image)
            atomic_image(folder/'masks'/p.name,mask.astype(np.uint8)*255)
            overlay=image.copy();overlay[mask]=(overlay[mask]*.55+np.array([60,220,60])*.45).astype(np.uint8)
            atomic_image(run/'preview.jpg',overlay)
            rows.append(dict(file=p.name,pixels=int(mask.sum()),score=score))
            event(type='progress',index=completed+i-start+1,total=sum(blocks[k][1]-blocks[k][0] for k in wanted),image=str(run/'preview.jpg'),message=f'第 {j+1}/{len(blocks)} 段 · 帧 {i+1}/{len(files)} · 目标像素 {mask.sum()}')
            if preview:event(type='done',message=f'第 {j+1} 段首帧预览完成（不发布掩码）',preview=True);return
        atomic_json(folder/'report.json',dict(frames=rows,state=tracker.state_size(),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30))
        if segmented:
            atomic_json(cache_file(root,files,start),dict(signature=signature(files,prompts,start,end),masks=str((folder/'masks').relative_to(root)),start=start,end=end))
        completed+=end-start
        event(type='segment_done',segment=j,message=f'第 {j+1} 段完成并保存缓存')
    if STOP.is_set():event(type='canceled',message='已取消，已完成段已保存');return
    if segmented:
        results=[cached_result(root,files,prompts,s,e) for s,e in blocks]
        if not all(results):
            event(type='done',message=f'所选段已完成；还有 {sum(r is None for r in results)} 段待处理，尚未发布完整 masks',partial=True);return
        final=run/'merged_masks';final.mkdir()
        for (start,end),result in zip(blocks,results):
            for p in files[start:end]:copy_file(root/result['masks']/p.name,final/p.name)
    else:
        final=folder/'masks'
        copy_file(folder/'report.json',run/'report.json')
    if (root/'masks').exists():
        backup=root/'metadata/backups'/stamp();backup.parent.mkdir(parents=True,exist_ok=True);os.replace(root/'masks',backup)
    os.replace(final,root/'masks')
    status=dict(prompts_sha256=prompt_digest(prompts),frames=len(files),run=str(run))
    if segmented:status['plan_sha256']=plan_signature(files,prompts,plan)
    atomic_json(root/'metadata/segmentation.json',status)
    event(type='done',message=f'全部 {len(files)} 帧分割完成，已合并发布完整 masks',dataset=str(root))
def capture(root,count=0,preview_only=False):
    import queue
    from integrations.orbbec_frames import color_frame_to_bgr,depth_frame_to_mm
    from pyorbbecsdk import Pipeline,Config,OBSensorType,OBStreamType,OBFrameAggregateOutputMode,AlignFilter
    root=Path(root)
    if root.exists() and not preview_only:raise ValueError('录制目标已存在，请使用新名称')
    for sub in (['metadata'] if preview_only else ['rgb','depth','metadata']):(root/sub).mkdir(parents=True,exist_ok=True)
    pipe=Pipeline();cfg=Config();profiles=[]
    for sensor in [OBSensorType.COLOR_SENSOR,OBSensorType.DEPTH_SENSOR]:
        p=pipe.get_stream_profile_list(sensor).get_default_video_stream_profile();cfg.enable_stream(p);profiles.append(p)
    cfg.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    pipe.enable_frame_sync();align=AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    q=queue.Queue(maxsize=24);errors=[];saved=[0];dropped=0;rejected=0;last=0;accepted=0;acquisition_ok=False;started=time.monotonic();meta={};deltas=[]
    def writer():
        try:
            with (root/'metadata/frames.jsonl').open('w') as log:
                while True:
                    item=q.get()
                    try:
                        if item is None:return
                        rgb,depth,row=item;name=row['file']
                        atomic_image(root/'rgb'/name,rgb)
                        try:atomic_image(root/'depth'/name,depth)
                        except Exception:
                            (root/'rgb'/name).unlink(missing_ok=True);raise
                        log.write(json.dumps(row)+'\n');log.flush();saved[0]+=1
                    finally:q.task_done()
        except Exception as exc:errors.append(str(exc));STOP.set()
    thread=threading.Thread(target=writer,daemon=True);
    if not preview_only:thread.start()
    running=False;next_preview=0;last_rate=PREVIEW_FPS;frame_times=deque()
    try:
        pipe.start(cfg);running=True
        misses=0
        while not STOP.is_set():
            frames=pipe.wait_for_frames(1000)
            if not frames:
                misses+=1
                if misses>15:raise RuntimeError('15 秒未收到完整 RGB-D 帧')
                continue
            misses=0;aligned=align.process(frames)
            if not aligned:continue
            aligned=aligned.as_frame_set();c=aligned.get_color_frame();d=aligned.get_depth_frame()
            if not c or not d:continue
            ct,dt=c.get_timestamp_us(),d.get_timestamp_us();delta=abs(ct-dt)
            if ct<=last or delta>500000/profiles[0].get_fps():rejected+=1;continue
            last=ct;rgb=color_frame_to_bgr(c);depth=depth_frame_to_mm(d)
            if rgb is None:continue
            if rgb.shape[:2]!=depth.shape:raise RuntimeError('D2C 后 RGB/Depth 尺寸不同')
            h,w=rgb.shape[:2]
            if not meta:
                intr=c.get_stream_profile().as_video_stream_profile().get_intrinsic()
                dist=c.get_stream_profile().as_video_stream_profile().get_distortion()
                K=np.array([[intr.fx,0,intr.cx],[0,intr.fy,intr.cy],[0,0,1.]])
                coeff=np.array([dist.k1,dist.k2,dist.p1,dist.p2,dist.k3,dist.k4,dist.k5,dist.k6])
                # The color stream profile describes the delivered color pixels; rectify once.
                if 'FISHEYE' in str(dist.model).upper():raise RuntimeError('当前彩色流为鱼眼畸变，需单独标定处理')
                mapx,mapy=cv2.initUndistortRectifyMap(K,coeff,None,K,(w,h),cv2.CV_32FC1)
                cw=min(w,int(h*4/3));ch=min(h,int(w*3/4));x=(w-cw)//2;y=(h-ch)//2
                A=np.array([[640/cw,0,-x*640/cw],[0,480/ch,-y*480/ch],[0,0,1.]])
                if not preview_only:np.savetxt(root/'cam_K.txt',A@K,fmt='%.9f')
                meta=dict(depth_unit='millimeter',alignment='software D2C to color; shared undistortion; center crop; resize',
                    input_size=[w,h],output_size=[640,480],K_input=K.tolist(),distortion=coeff.tolist(),distortion_model=str(dist.model),crop=[x,y,cw,ch],K_output=(A@K).tolist(),
                    profiles=[dict(width=p.get_width(),height=p.get_height(),fps=p.get_fps(),format=str(p.get_format())) for p in profiles],max_sync_delta_us=500000/profiles[0].get_fps(),frame_sync_enabled=True)
                if not preview_only:atomic_json(root/'metadata/capture.json',meta)
            rgb=cv2.remap(rgb,mapx,mapy,cv2.INTER_LINEAR)[y:y+ch,x:x+cw]
            depth=cv2.remap(depth,mapx,mapy,cv2.INTER_NEAREST)[y:y+ch,x:x+cw]
            rgb=cv2.resize(rgb,(640,480),interpolation=cv2.INTER_AREA)
            depth=np.clip(np.rint(cv2.resize(depth,(640,480),interpolation=cv2.INTER_NEAREST)),0,65535).astype(np.uint16)
            now=time.monotonic();frame_times.append(now)
            while len(frame_times)>2 and now-frame_times[0]>1.:frame_times.popleft()
            capture_fps=(len(frame_times)-1)/(frame_times[-1]-frame_times[0]) if len(frame_times)>1 else 0.
            rate=PREVIEW_FPS
            if rate!=last_rate:next_preview=now;last_rate=rate
            show_preview=now>=next_preview
            if show_preview:
                next_preview=next_preview+1/rate if next_preview else now+1/rate
                if next_preview<now-1/rate:next_preview=now+1/rate
            if preview_only:
                if show_preview:
                    atomic_image(root/'metadata/live.jpg',rgb)
                    event(type='live',capture_fps=capture_fps,image=str(root/'metadata/live.jpg'),message='相机实时预览 · 尚未录制 · 可调整物体后点击开始录制')
                continue
            row=dict(file=f'{ct*1000:019d}.png',color_timestamp_us=ct,depth_timestamp_us=dt,delta_us=delta,host_time_ns=time.time_ns(),depth_scale=d.get_depth_scale())
            if count and accepted>=count:break
            try:q.put_nowait((rgb,depth,row));deltas.append(delta);accepted+=1
            except queue.Full:dropped+=1
            if show_preview:
                color=cv2.applyColorMap(np.uint8(np.clip(depth/1500,0,1)*255),cv2.COLORMAP_TURBO);color[depth==0]=0
                atomic_image(root/'metadata/live.jpg',np.concatenate([rgb,color],axis=1))
                event(type='progress',capture_fps=capture_fps,index=saved[0],total=0,image=str(root/'metadata/live.jpg'),message=f'录制 {saved[0]} 帧；队列丢弃 {dropped}，不同步拒绝 {rejected}')
            if time.monotonic()-started>15 and not saved[0]:raise RuntimeError('没有满足同步条件的帧，请检查相机')
        acquisition_ok=True
    finally:
        if running:pipe.stop()
        if not preview_only:
            # Drain accepted pairs before declaring recording complete.
            while thread.is_alive():
                try:q.put(None,timeout=.2);break
                except queue.Full:continue
            thread.join(timeout=30)
            meta.update(saved_frames=saved[0],queue_dropped=dropped,sync_rejected=rejected,delta_us_percentiles=np.percentile(deltas,[0,50,95,100]).tolist() if deltas else [],seconds=time.monotonic()-started,complete=acquisition_ok and not errors and not thread.is_alive())
            atomic_json(root/'metadata/capture.json',meta)
    if preview_only:
        event(type='done',message='相机预览已停止',live_preview=True);return
    if errors:raise RuntimeError(errors[0])
    if thread.is_alive() or not saved[0]:raise RuntimeError('录制未完成或没有保存帧')
    event(type='done',dataset=str(root),message=f'已保存 {saved[0]} 对 RGB-D，内参已保存')
def main():
    global PREVIEW_FPS
    ap=argparse.ArgumentParser();ap.add_argument('job',choices=['capture','live','segment','pose_init','import','validate']);ap.add_argument('dataset');ap.add_argument('--source');ap.add_argument('--export');ap.add_argument('--max-depth',type=float,default=1.);ap.add_argument('--frames',type=int,default=0);ap.add_argument('--frame-index',type=int,default=0);ap.add_argument('--prompts-file');ap.add_argument('--preview',action='store_true');ap.add_argument('--segment-index',type=int);ap.add_argument('--force',action='store_true');ap.add_argument('--preview-fps',type=int,choices=[10,20,30],default=10);args=ap.parse_args()
    PREVIEW_FPS=args.preview_fps
    threading.Thread(target=controls,daemon=True).start()
    from contextlib import ExitStack
    from gpu_lock import GpuLease
    from data_lock import DataAccess
    try:
        with ExitStack() as leases:
            leases.enter_context(DataAccess())
            if args.job in ['segment','pose_init']:leases.enter_context(GpuLease('SAM3 '+args.job))
            return dispatch(args)
    except Exception as exc:
        traceback.print_exc();event(type='error',message=str(exc));return 1

def dispatch(args):
    try:
        if args.job=='capture':capture(args.dataset,args.frames)
        elif args.job=='live':capture(args.dataset,preview_only=True)
        elif args.job=='segment':segment(args.dataset,args.preview,args.segment_index,args.force)
        elif args.job=='pose_init':pose_init(args.dataset,args.frame_index,args.prompts_file)
        elif args.job=='import':event(type='done',**import_rgb(args.source,args.dataset,args.frames or None),message='RGB 工作副本已导入')
        else:
            result=validate(args.dataset,args.max_depth,args.export,event)
            event(type='done',**result,message=f'数据验收：{result["usable"]}/{result["frames"]} 帧有效，导出 {len(result["exported"])} 个连续片段')
    except Exception as exc:
        traceback.print_exc();event(type='error',message=str(exc));return 1
    return 0
if __name__=='__main__':sys.exit(main())

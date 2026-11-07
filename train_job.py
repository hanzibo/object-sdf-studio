"""Supervise the unmodified BundleSDF CLI with validation, logs and process-group cleanup."""
import argparse
from collections import deque
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import traceback

from config import ROOT, RUNTIME, ENVS, BUNDLE_PROJECT, BUNDLE_PYTHON
from data_ops import atomic_json, rgb_files, stamp, validate

PROJECT = BUNDLE_PROJECT
STOP = threading.Event()


def event(**value):
    try:
        print('EVENT ' + json.dumps(value, ensure_ascii=False), flush=True)
    except BrokenPipeError:
        STOP.set()


def controls():
    for line in sys.stdin:
        if line.strip() == 'stop':
            STOP.set()


def safe_path(path):
    result = Path(path).expanduser().resolve()
    # Upstream run_custom uses unquoted shell commands for its output directory.
    if not re.fullmatch(r'[A-Za-z0-9_./-]+', str(result)):
        raise ValueError('原项目的路径处理有限制，请使用仅含英文字母、数字、下划线、短横线的无空格路径')
    return result


def existing_training():
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
            if any(Path(a.decode(errors='replace')).name == 'run_custom.py' for a in args if a):
                return int(proc.name)
        except (OSError, ValueError):
            continue
    return None


def command(dataset, output):
    return [str(BUNDLE_PYTHON), '-u', str(PROJECT / 'run_custom.py'),
            '--mode', 'run_video', '--video_dir', str(dataset), '--out_folder', str(output),
            '--use_segmenter', '0', '--use_gui', '0', '--stride', '1', '--debug_level', '1']


def run(dataset, output_root):
    dataset, output_root = safe_path(dataset), safe_path(output_root)
    if 'rgb' in str(dataset):
        raise ValueError('数据集父路径包含小写 rgb，会影响原项目的路径替换；请将父目录改名后再运行')
    files = rgb_files(dataset)
    if any(p.suffix != '.png' or not re.fullmatch(r'[A-Za-z0-9_-]+', p.stem) for p in files):
        raise ValueError('训练需要同名的 PNG RGB/Depth/Mask，帧名请仅使用数字、字母、下划线或短横线')
    if len(files) < 5:
        raise ValueError('请选择至少 5 帧的连续数据片段')
    if not BUNDLE_PYTHON.exists():
        raise ValueError('找不到 bundlesdf 环境的 Python')
    RUNTIME.mkdir(parents=True, exist_ok=True)
    lock = (RUNTIME / 'training.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ValueError('已有启动器训练任务运行，请等待结束，避免重复占用显存')
    child = None
    reader = None
    job_dir = None
    report = {}
    started = time.monotonic()
    try:
        other = existing_training()
        if other:
            raise ValueError(f'已有 BundleSDF 训练进程 PID {other}，请先等待该任务结束')
        output_root.mkdir(parents=True, exist_ok=True)
        job_dir = output_root / ('run_' + stamp())
        job_dir.mkdir(exist_ok=False)
        output = job_dir / 'reconstruction'  # Upstream may delete this new child only.
        report = dict(dataset=str(dataset), job_dir=str(job_dir), output=str(output),
                      frames=len(files), status='validating', started_at=time.time())
        atomic_json(job_dir / 'job.json', report)
        event(type='job', **report)
        event(type='status', phase='validating', message='检查所有 RGB、深度、掩码与内参…')

        def validation_event(**value):
            if STOP.is_set():
                raise InterruptedError('验收已取消')
            value['phase'] = 'validating'
            event(**value)

        checked = validate(dataset, max_depth_m=1.0, event=validation_event)
        validation = json.loads(Path(checked['report']).read_text())
        atomic_json(job_dir / 'validation.json', validation)
        if checked['usable'] != len(files):
            raise ValueError(f'只有 {checked["usable"]}/{len(files)} 帧有效；请先在 Studio 修正并重新导出')
        if checked['segments'] != 1:
            raise ValueError('该目录含多个不连续片段，请在 Studio 重新导出并选择单个 segment 目录')
        if STOP.is_set():
            raise InterruptedError('验收已取消')
        argv = command(dataset, output)
        env = os.environ.copy()
        for name in ['PYTHONPATH', 'QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_PLUGIN_PATH']:
            env.pop(name, None)
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        atomic_json(job_dir / 'command.json', dict(argv=argv, cwd=str(PROJECT),
                    environment={'PYTORCH_CUDA_ALLOC_CONF': env['PYTORCH_CUDA_ALLOC_CONF']}))
        log_path = job_dir / 'train.log'
        tails = deque(maxlen=30)
        tail_lock = threading.Lock()
        stage = {'phase': 'tracking', 'iteration': 0, 'iterations': 0}
        log_error = []
        with log_path.open('w', encoding='utf-8') as log:
            child = subprocess.Popen(argv, cwd=PROJECT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     start_new_session=True, text=True, errors='replace', bufsize=1)
            group = child.pid
            report.update(status='tracking', pid=child.pid, log=str(log_path))
            atomic_json(job_dir / 'job.json', report)
            event(type='started', pid=child.pid, log=str(log_path), command=argv)

            def read_log():
                try:
                    for line in child.stdout:
                        log.write(line)
                        log.flush()
                        with tail_lock:
                            tails.append(line.rstrip())
                            if re.search(r'project train_images\s+\d+/\d+', line):
                                stage['phase'] = 'texturing'
                                match = re.search(r'(\d+)/(\d+)', line)
                                stage['iteration'], stage['iterations'] = map(int, match.groups())
                            elif (output / 'final').exists():
                                if stage['phase'] != 'texturing':
                                    if stage['phase'] != 'global':
                                        stage.update(phase='global', iteration=0, iterations=0)
                                    match = re.search(r'train progress\s+(\d+)/(\d+)', line)
                                    if match:
                                        stage['iteration'], stage['iterations'] = map(int, match.groups())
                except Exception as exc:
                    log_error.append(str(exc))
                    STOP.set()

            reader = threading.Thread(target=read_log, daemon=True)
            reader.start()
            history = deque()
            stopped_at = None
            term_sent = False
            last_emit = 0
            train_start = time.monotonic()
            while True:
                now = time.monotonic()
                if STOP.is_set() and stopped_at is None:
                    stopped_at = now
                    event(type='status', phase='stopping', message='正在停止训练及其子进程，已保存的结果将保留…')
                    kill_group(group, signal.SIGINT)
                if stopped_at is not None:
                    if now - stopped_at > 12 and not term_sent:
                        kill_group(group, signal.SIGTERM)
                        term_sent = True
                    if now - stopped_at > 17:
                        kill_group(group, signal.SIGKILL)
                code = child.poll()
                if code is not None:
                    break
                if now - last_emit >= 1:
                    count = len(list((output / 'ob_in_cam').glob('*.txt')))
                    history.append((now, count))
                    while len(history) > 2 and now - history[0][0] > 120:
                        history.popleft()
                    elapsed = now - train_start
                    rate = None
                    if elapsed >= 15 and count:
                        seconds, delta = now - history[0][0], count - history[0][1]
                        rate = seconds / delta if seconds >= 20 and delta > 0 else elapsed / count
                    with tail_lock:
                        state = dict(stage)
                        batch = list(tails)
                        tails.clear()
                    if (output / 'final').exists() and state['phase'] == 'tracking':
                        state['phase'] = 'global'
                    eta = (len(files) - count) * rate if rate and count < len(files) else None
                    event(type='progress', phase='stopping' if stopped_at else state['phase'],
                          index=count, total=len(files), elapsed=now-started,
                          seconds_per_frame=rate, eta_input_seconds=eta,
                          iteration=state['iteration'], iterations=state['iterations'], lines=batch)
                    last_emit = now
                time.sleep(.1)
            # A failed or interrupted parent must not leave CUDA NeRF/manager children behind.
            if code != 0 or STOP.is_set():
                kill_group(group, signal.SIGTERM)
                time.sleep(.3)
                kill_group(group, signal.SIGKILL)
            reader.join(timeout=5)
            with tail_lock:
                remaining = list(tails)
            if remaining:
                event(type='logs', lines=remaining)
        if STOP.is_set():
            if log_error:
                raise RuntimeError('日志写入失败：' + '; '.join(log_error))
            raise InterruptedError('训练已停止；结果和日志已保留，重新开始会使用新目录')
        if code != 0:
            raise RuntimeError(f'BundleSDF 退出码 {code}，请查看 train.log 最后的报错')
        expected = ['textured_mesh.obj', 'material.mtl', 'material_0.png']
        if not all((output / name).is_file() and (output / name).stat().st_size for name in expected):
            raise RuntimeError('进程已退出，但缺少完整模型或纹理文件，请检查日志')
        count = len(list((output / 'ob_in_cam').glob('*.txt')))
        report.update(status='complete', saved_poses=count, elapsed_seconds=time.monotonic()-started,
                      finished_at=time.time(), returncode=code)
        atomic_json(job_dir / 'job.json', report)
        event(type='done', **report, message='完整流程已结束，模型与贴图已生成')
        return 0
    except InterruptedError as exc:
        report.update(status='canceled', message=str(exc), elapsed_seconds=time.monotonic()-started)
        if job_dir:
            atomic_json(job_dir / 'job.json', report)
        event(type='canceled', **report)
        return 2
    except Exception as exc:
        report.update(status='failed', message=str(exc), elapsed_seconds=time.monotonic()-started)
        if job_dir:
            atomic_json(job_dir / 'job.json', report)
        event(type='error', **report)
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        if child:
            if child.poll() is None:
                kill_group(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            if child.stdout:
                child.stdout.close()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def kill_group(group, sig):
    try:
        os.killpg(group, sig)
    except ProcessLookupError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output-root', type=Path, default=ROOT/'training_runs')
    args = parser.parse_args()
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda *_: STOP.set())
    threading.Thread(target=controls, daemon=True).start()
    try:
        from gpu_lock import GpuLease
        from data_lock import DataAccess
        with DataAccess(), GpuLease('BundleSDF training'):
            return run(args.dataset, args.output_root)
    except Exception as exc:
        event(type='error', message=str(exc))
        return 1


if __name__ == '__main__':
    sys.exit(main())

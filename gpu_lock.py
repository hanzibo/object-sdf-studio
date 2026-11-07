"""Process-scoped GPU lease shared by Studio workers and its training launcher."""
import fcntl
import json
import os
from config import RUNTIME


class GpuLease:
    def __init__(self, task):
        self.task = task
        self.handle = None

    def __enter__(self):
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.handle = (RUNTIME / 'gpu.lock').open('a+')
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.seek(0)
            owner = self.handle.read().strip()
            self.handle.close()
            self.handle = None
            raise RuntimeError('GPU 正被其他 Studio 任务使用，请先停止或等待结束：' + owner)
        self.handle.seek(0)
        self.handle.truncate()
        json.dump(dict(pid=os.getpid(), task=self.task), self.handle, ensure_ascii=False)
        self.handle.flush()
        return self

    def __exit__(self, *_):
        if self.handle:
            # Keep the inode: unlinking a locked file can create competing locks.
            self.handle.seek(0)
            self.handle.truncate()
            self.handle.flush()
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None

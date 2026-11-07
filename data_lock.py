"""Readers are capture/inference jobs; history deletion needs exclusive access."""
import fcntl
from config import RUNTIME


class DataAccess:
    def __init__(self, exclusive=False, runtime=None):
        self.runtime = runtime or RUNTIME
        self.exclusive = exclusive
        self.handle = None

    def __enter__(self):
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.handle = (self.runtime/'history_access.lock').open('a+')
        try:
            mode = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH
            fcntl.flock(self.handle, mode | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close();self.handle=None
            raise RuntimeError('当前正在采集、推理或清理历史数据，请先等待任务结束')
        return self

    def __exit__(self, *_):
        if self.handle:
            fcntl.flock(self.handle,fcntl.LOCK_UN);self.handle.close();self.handle=None

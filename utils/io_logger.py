import asyncio
from multiprocessing import Queue
from queue import Empty
from enum import Enum
from dataclasses import dataclass


class IOEvent(Enum):
    NET_READ = 1
    NET_WRITE = 2
    DISK_READ = 3
    DISK_WRITE = 4


@dataclass
class IOLog:
    event: IOEvent
    value: int
    timestamp: float


class IOLogger:
    def __init__(self):
        self._message_queue = Queue()
        self._log_queue = Queue()
        self._active = True
        self._active_counter = 0

    def log(self, event, value, timestamp, wait=False, force=False):
        if not self._active:
            return
        try:
            try:
                while True:
                    self._active_counter += self._message_queue.get_nowait()
            except Empty:
                pass

            if self._active_counter >= 0 or force:
                log_item = IOLog(event, value, timestamp)
                self._log_queue.put(log_item, block=wait)
        except ValueError:  # Queues are closed
            self._active = False

    def close(self, wait=False):
        if self._active:
            self._active = False
            self._message_queue.close()
            self._log_queue.close()
            if wait:
                self._message_queue.join_thread()
                self._log_queue.join_thread()

    def begin_watching(self):
        if self._active:
            try:
                self._message_queue.put_nowait(1)
            except ValueError:  # Queues are closed
                self._active = False

    def end_watching(self):
        if self._active:
            try:
                self._message_queue.put_nowait(-1)
            except ValueError:  # Queues are closed
                self._active = False

    def read_all_available(self, wait=False, timeout=None) -> list[IOLog]:
        if not self._active:
            return

        logs = []
        try:
            while True:
                log = self._log_queue.get(block=wait, timeout=timeout)
                if isinstance(log, IOLog):
                    logs.append(log)
        except Empty:
            return logs
        except ValueError:  # Queues are closed
            self._active = False

    async def watch_logs_async(self):
        self.begin_watching()

        while self._active:
            for log in self.read_all_available():
                yield log
            if self._active:
                await asyncio.sleep(0.05)

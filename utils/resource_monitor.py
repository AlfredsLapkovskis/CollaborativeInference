from threading import Thread, Lock
import psutil
import time
from dataclasses import dataclass

from models.events import *
from utils.io_logger import *


@dataclass
class ResourceUsage:
    cpu: float = 0.0
    memory: int = 0
    net_read: float = 0
    net_write: float = 0
    disk_read: float = 0
    disk_write: float = 0


class ResourceMonitor:

    _LOCK = Lock()
    _THREAD = None
    _monitor_counter = 0
    _TASKS = set()

    def __init__(self, pid=None, io_logger: IOLogger = None):
        self._pid = pid
        self._closed = False
        self._active = False
        self._process = psutil.Process(pid)
        self._io_logger = io_logger
        self._exception = None

        self._reset_state()

        with ResourceMonitor._LOCK:
            if ResourceMonitor._monitor_counter <= 0:
                ResourceMonitor._monitor_counter = 1
                ResourceMonitor._THREAD = Thread(
                    target=self._track_resources,
                    name="ResourceMonitorThread",
                )
                ResourceMonitor._THREAD.start()
            else:
                ResourceMonitor._monitor_counter += 1

        print(f"UsageMonitor pid={pid}")

    def start(self):
        with ResourceMonitor._LOCK:
            if self._active:
                raise RuntimeError("UsageMonitor is already active.")
            self._active = True
            if self._io_logger:
                self._io_logger.begin_watching()
            ResourceMonitor._TASKS.add(self._task)

    def stop(self):
        with ResourceMonitor._LOCK:
            if not self._active:
                raise RuntimeError("UsageMonitor is inactive.")
            self._active = False
            ResourceMonitor._TASKS.remove(self._task)
            if self._io_logger:
                self._io_logger.end_watching()

            usage = self._usage
            self._reset_state()

            if self._exception:
                exc = self._exception
                self._exception = None
                raise exc

            return usage

    def close(self, wait=False):
        with ResourceMonitor._LOCK:
            if self._closed:
                return
            self._closed = True

            ResourceMonitor._monitor_counter -= 1
            wait = wait and ResourceMonitor._monitor_counter <= 0

        if wait:
            ResourceMonitor._THREAD.join()

    def _task(self):
        try:
            net_read_bytes = 0
            net_write_bytes = 0
            disk_read_bytes = 0
            disk_write_bytes = 0

            net_read_duration = 0
            net_write_duration = 0
            disk_read_duration = 0
            disk_write_duration = 0

            if self._io_logger:
                last_logs = [self._last_net_read, self._last_net_write,
                             self._last_disk_read, self._last_disk_write]
                last_logs = [l for l in last_logs if l is not None]

                all_logs = [*last_logs, *self._io_logger.read_all_available()]

                def get_info(logs: list[IOLog], event: IOEvent):
                    event_logs = [l for l in logs if l.event == event]
                    if not event_logs:
                        return 0, 0, None
                    if len(event_logs) < 2:
                        return 0, 0, event_logs[-1]
                    total_bytes = sum(log.value for log in event_logs[1:])
                    duration = event_logs[-1].timestamp - \
                        event_logs[0].timestamp
                    return total_bytes, duration, event_logs[-1]

                net_read_bytes, net_read_duration, self._last_net_read = get_info(
                    all_logs, IOEvent.NET_READ)
                net_write_bytes, net_write_duration, self._last_net_write = get_info(
                    all_logs, IOEvent.NET_WRITE)
                disk_read_bytes, disk_read_duration, self._last_disk_read = get_info(
                    all_logs, IOEvent.DISK_READ)
                disk_write_bytes, disk_write_duration, self._last_disk_write = get_info(
                    all_logs, IOEvent.DISK_WRITE)

            self._cpu_counter += 1
            self._net_read_duration += net_read_duration
            self._net_write_duration += net_write_duration
            self._disk_read_duration += disk_read_duration
            self._disk_write_duration += disk_write_duration

            def save_div(a, b):
                return a / b if b != 0 else 0.

            self._usage = ResourceUsage(
                # Computing average CPU time
                cpu=self._usage.cpu +
                (self._process.cpu_percent() - self._usage.cpu) / self._cpu_counter,
                memory=max(self._process.memory_info().rss,
                           self._usage.memory),
                # Computing weighted average I/O throughput
                net_read=self._usage.net_read + save_div(net_read_duration, self._net_read_duration) * (
                    save_div(net_read_bytes, net_read_duration) - self._usage.net_read),
                net_write=self._usage.net_write + save_div(net_write_duration, self._net_write_duration) * (
                    save_div(net_write_bytes, net_write_duration) - self._usage.net_write),
                disk_read=self._usage.disk_read + save_div(disk_read_duration, self._disk_read_duration) * (
                    save_div(disk_read_bytes, disk_read_duration) - self._usage.disk_read),
                disk_write=self._usage.disk_write + save_div(disk_write_duration, self._disk_write_duration) * (
                    save_div(disk_write_bytes, disk_write_duration) - self._usage.disk_write),
            )
        except BaseException as exc:
            self._exception = exc

    def _reset_state(self):
        self._usage = ResourceUsage()
        self._cpu_counter = 0
        self._last_net_read = None
        self._last_net_write = None
        self._last_disk_read = None
        self._last_disk_write = None
        self._net_read_duration = 0
        self._net_write_duration = 0
        self._disk_read_duration = 0
        self._disk_write_duration = 0

    @staticmethod
    def _track_resources():
        while True:
            with ResourceMonitor._LOCK:
                if ResourceMonitor._monitor_counter <= 0:
                    return
                for task in ResourceMonitor._TASKS:
                    task()

            time.sleep(0.05)

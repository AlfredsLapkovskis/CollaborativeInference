from abc import ABC


class IOEvent(ABC):
    def __init__(self, pid, bytes: int, timestamp: float):
        self.pid = pid
        self.bytes = bytes
        self.timestamp = timestamp


class NetSendEvent(IOEvent):
    pass


class NetReceiveEvent(IOEvent):
    pass


class DiskReadEvent(IOEvent):
    pass


class DiskWriteEvent(IOEvent):
    pass

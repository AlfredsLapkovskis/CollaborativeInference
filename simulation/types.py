from unittest import case


class XState:
    EMPTY = 0
    DOWNLOADING = 1
    PAUSED = 2
    READY = 3


class YState:
    IDLE = 0
    DOWNLOADING = 1
    EXECUTING = 2
    UPLOADING = 3
    DONE = 4


class ServerXAction:
    NONE = 0
    DOWNLOAD = 1
    CANCEL = 2


class ServerYAction:
    NONE = 0
    EXECUTE = 1
    FINISH = 2
    CANCEL = 3


class ClientXAction:
    NONE = 0
    ABORT = 1
    PAUSE = 2
    RESUME = 3


class ClientYAction:
    NONE = 0
    ABORT = 1


class Resource:
    BANDWIDTH_IN = 0
    BANDWIDTH_OUT = 1
    CPU = 2
    MEMORY = 3
    STORAGE = 4

    @staticmethod
    def available_resources():
        return [Resource.BANDWIDTH_IN, Resource.BANDWIDTH_OUT, Resource.CPU, Resource.MEMORY, Resource.STORAGE]

    @staticmethod
    def X_resources():
        return [Resource.BANDWIDTH_IN, Resource.CPU, Resource.MEMORY, Resource.STORAGE]

    @staticmethod
    def Y_resources(state=None):
        if state == YState.DOWNLOADING:
            return [Resource.BANDWIDTH_IN, Resource.CPU, Resource.MEMORY]
        elif state == YState.EXECUTING:
            return [Resource.CPU, Resource.MEMORY]
        elif state == YState.UPLOADING:
            return [Resource.BANDWIDTH_OUT, Resource.CPU, Resource.MEMORY]
        elif state == YState.DONE or state == YState.IDLE:
            return []

        return [Resource.BANDWIDTH_IN, Resource.BANDWIDTH_OUT, Resource.CPU, Resource.MEMORY]

    @staticmethod
    def available_resource_idx(resource):
        return Resource.available_resources().index(resource)

    @staticmethod
    def X_resource_idx(resource):
        return Resource.X_resources().index(resource)

    @staticmethod
    def Y_resource_idx(resource, state=None):
        return Resource.Y_resources(state).index(resource)

    @staticmethod
    def get_name(resource):
        match resource:
            case Resource.BANDWIDTH_IN:
                return "bandwidth_in"
            case Resource.BANDWIDTH_OUT:
                return "bandwidth_out"
            case Resource.CPU:
                return "cpu"
            case Resource.MEMORY:
                return "memory"
            case Resource.STORAGE:
                return "storage"
            case _:
                raise ValueError(f"Unknown resource: {resource}")

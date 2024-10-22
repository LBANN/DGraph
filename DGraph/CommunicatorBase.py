class CommunicatorBase:
    _is_initialized = False

    def __init__(self):
        pass

    def init_process_group(self, backend: str, **kwargs):
        raise NotImplementedError

    def get_rank(self) -> int:
        raise NotImplementedError

    def get_world_size(self) -> int:
        raise NotImplementedError

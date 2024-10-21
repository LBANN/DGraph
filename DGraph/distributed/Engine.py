import torch


class BackendEngine(object):
    def __init__(self):
        pass

    def init_process_group(self, *args, **kwargs):
        raise NotImplementedError

    def get_rank(self) -> int:
        raise NotImplementedError

    def get_world_size(self) -> int:
        raise NotImplementedError

    def scatter(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def gather(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def finalize(self) -> None:
        raise NotImplementedError

    def barrier(self) -> None:
        raise NotImplementedError

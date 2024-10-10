import torch
from DGraph.distributed.nccl import NCCLBackendEngine

SUPPORTED_BACKENDS = ["nccl", "mpi", "nvshmem"]


class Communicator:
    """Wrapper class for initializing and managing the distributed communication backend.
    All the communication between the processes should be done through this class.
    """

    def __init__(self, backend: str, **kwargs) -> None:
        assert (
            backend in SUPPORTED_BACKENDS
        ), f"Backend {backend} not supported. Supported backends: {SUPPORTED_BACKENDS}"

        self.backend = backend
        self.kwargs = kwargs

        # TODO: Initialize the process group based on the backend
        # self.__backend_engine
        if backend == "nccl":
            self.__backend_engine = NCCLBackendEngine()
        else:
            raise NotImplementedError(f"Backend {backend} not implemented")
        self._is_initialized = True

    @staticmethod
    def init_process_group(backend: str, **kwargs) -> "Communicator":
        return Communicator(backend, **kwargs)

    def get_rank(self) -> int:
        """Returns the rank of the current process."""
        self.__check_init()
        return 0

    def get_world_size(self) -> int:
        self.__check_init()
        return 1

    def scatter(self, *args, **kwargs):
        pass

    def gather(self, *args, **kwargs):
        pass

    def destroy(self) -> None:
        """Destroys the process group and releases resources."""
        self.__check_init()
        self._is_initialized = False

    def __check_init(self) -> None:
        assert self._is_initialized, "Communicator not initialized"

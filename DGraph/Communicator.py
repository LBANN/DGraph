import torch
from DGraph.distributed.nccl import NCCLBackendEngine
from DGraph.CommunicatorBase import CommunicatorBase

SUPPORTED_BACKENDS = ["nccl", "mpi", "nvshmem"]


class Communicator(CommunicatorBase):
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
        Communicator._is_initialized = True

    @staticmethod
    def init_process_group(backend: str, **kwargs) -> "Communicator":
        """Initializes the process group with the specified backend."""
        if Communicator._is_initialized:
            raise RuntimeError("Communicator already initialized")
        return Communicator(backend, **kwargs)

    def get_rank(self) -> int:
        """Returns the rank of the current process."""
        self.__check_init()
        return self.__backend_engine.get_rank()

    def get_world_size(self) -> int:
        self.__check_init()
        return self.__backend_engine.get_rank()

    def scatter(self, *args, **kwargs) -> torch.Tensor:
        return self.__backend_engine.scatter(*args, **kwargs)

    def gather(self, *args, **kwargs) -> torch.Tensor:
        return self.__backend_engine.gather(*args, **kwargs)

    def destroy(self) -> None:
        """Destroys the process group and releases resources."""
        self.__check_init()
        Communicator._is_initialized = False

    def __check_init(self) -> None:
        """Check if the communicator is initialized."""
        assert Communicator._is_initialized, "Communicator not initialized"

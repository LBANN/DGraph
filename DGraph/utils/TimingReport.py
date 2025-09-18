import torch
import torch.distributed as dist
from DGraph.Communicator import Communicator


class TimingReport:
    """A utility class to report the timing of operations in a distributed setting."""

    _timers = {}
    _communicator = None
    _is_initialized = False

    def __init__(
        self,
    ):
        pass

    @staticmethod
    def init(communicator: Communicator):
        """Initialize the TimingReport with a communicator."""
        if TimingReport._is_initialized:
            raise RuntimeError("TimingReport is already initialized.")
        TimingReport._communicator = communicator
        TimingReport._is_initialized = True
        TimingReport._timers = {}

    @staticmethod
    def start(name: str):
        """Start a timer for the given operation name."""
        if not TimingReport._is_initialized:
            raise RuntimeError("TimingReport is not initialized. Call init first.")
        assert TimingReport._communicator is not None, "Communicator is not set."
        TimingReport._communicator.barrier()  # Ensure all processes are synchronized before starting the timer
        if name not in TimingReport._timers:
            TimingReport._timers[name] = []
        start_event = torch.cuda.Event(enable_timing=True)
        TimingReport._timers[name].append(start_event)
        TimingReport._timers[name][-1].record(torch.cuda.current_stream())

    @staticmethod
    def stop(name: str):
        """Stop the timer for the given operation name."""
        if name not in TimingReport._timers or not TimingReport._timers[name]:
            raise ValueError(f"No timer started for {name}")
        if not TimingReport._is_initialized:
            raise RuntimeError("TimingReport is not initialized. Call init first.")
        assert TimingReport._communicator is not None, "Communicator is not set."
        start = TimingReport._timers[name][-1]
        end = torch.cuda.Event(enable_timing=True)

        end.record(torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
        TimingReport._timers[name][
            -1
        ] = elapsed_time  # Replace the event with the elapsed
        TimingReport._communicator.barrier()

        return elapsed_time

    @staticmethod
    def add_time(name: str, elapsed_time: float):
        """Manually add a timing entry for the given operation name."""
        if not TimingReport._is_initialized:
            raise RuntimeError("TimingReport is not initialized. Call init first.")
        if name not in TimingReport._timers:
            TimingReport._timers[name] = []
        TimingReport._timers[name].append(elapsed_time)

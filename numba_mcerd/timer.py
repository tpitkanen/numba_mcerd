import timeit
from typing import Optional


class Timer:
    """Timer for benchmarking"""

    def __init__(self) -> None:
        self.start_time = None
        self.stop_time = None

    @property
    def elapsed(self) -> Optional[float]:
        """Return elapsed time if timer has started and stopped"""
        if self.stop_time is not None and self.start_time is not None:
            return self.stop_time - self.start_time
        return None

    def __str__(self) -> str:
        """Return string representation"""
        # return f"{type(self).__name__} (start={self.start_time} stop={self.stop_time})"
        return f"{type(self).__name__} (start={self.start_time} stop={self.stop_time} elapsed={self.elapsed})"

    def start(self) -> None:
        """Start timer"""
        if self.start_time is not None:
            raise TimerException("Timer already running")
        self.start_time = timeit.default_timer()

    def stop(self) -> float:
        """Stop timer

        Returns:
            Elapsed time
        """
        self.stop_time = timeit.default_timer()
        if self.start_time is None:
            raise TimerException("Timer not started")
        return self.elapsed


def init_and_start() -> "Timer":
    """Initialize and start a timer

    Returns:
        Timer
    """
    timer = Timer()
    timer.start()
    return timer


class TimerException(BaseException):
    """Exception for timer class"""
    pass

import timeit


class TimerException(BaseException):
    """Exception for timer class"""
    pass


class Timer:
    """Timer for benchmarking. Timer cannot be restarted."""

    def __init__(self) -> None:
        self.start_time = None
        self.stop_time = None

    @property
    def running(self) -> bool:
        """Return whether timer is running"""
        return self.start_time is not None and self.stop_time is None

    @property
    def elapsed(self) -> float:
        """Return elapsed time"""
        if not self.running:
            return self.stop_time - self.start_time
        if self.start_time is not None:
            return timeit.default_timer() - self.start_time
        return 0.0

    def __str__(self) -> str:
        """Return string representation"""
        return f"{type(self).__name__} (start={self.start_time} stop={self.stop_time} elapsed={self.elapsed})"

    def start(self) -> None:
        """Start timer"""
        if self.start_time is not None:
            raise TimerException("Timer already running")
        if self.running:
            raise TimerException("Timer already stopped")
        self.start_time = timeit.default_timer()

    def stop(self) -> float:
        """Stop timer

        Returns:
            Elapsed time
        """
        if self.start_time is None:
            raise TimerException("Timer not started")
        if not self.running:
            raise TimerException("Timer already stopped")
        self.stop_time = timeit.default_timer()
        return self.elapsed

    @classmethod
    def init_and_start(cls) -> "Timer":
        """Initialize and start a timer

        Returns:
            Timer
        """
        timer = cls()
        timer.start()
        return timer


class SplitTimer(Timer):
    """Timer with splits"""
    def __init__(self) -> None:
        super().__init__()
        self.splits = []

    @property
    def previous_split(self) -> float:
        """Return previous split"""
        if self.splits:
            return self.splits[-1]
        if self.start_time is not None:
            return self.start_time
        return 0.0  # TODO: TimerException instead?

    @property
    def elapsed_previous_split(self) -> float:
        """Return elapsed time since last split or start"""
        if self.running:
            return timeit.default_timer() - self.previous_split
        if self.splits:
            return self.stop_time - self.splits[-1]
        return self.stop_time - self.start_time

    def split(self) -> float:
        """Add a split to timer

        Returns:
            Elapsed time since last split
        """
        self.splits.append(timeit.default_timer())
        return self.elapsed_previous_split

    @property
    def split_times(self) -> [float]:
        """Return splits (normalized to start from 0)"""
        return [0.0] + [(split - self.start_time) for split in self.splits] + [self.elapsed]

    @property
    def lap_times(self) -> [float]:
        """Return splits as laps"""
        times = []
        previous = self.start_time
        for split in self.splits:
            times.append(split - previous)
            previous = split
        times.append(self.stop_time - previous)
        return times

    def __str__(self):
        """Return string representation"""
        return f"{type(self).__name__} (start={self.start_time} stop={self.stop_time} elapsed={self.elapsed} splits={self.splits})"

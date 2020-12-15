import timeit


class TimerException(BaseException):
    """Exception for timer class"""
    pass


class Timer:
    """Timer for benchmarking. Timer cannot be restarted."""

    MSG_TIMER_RUNNING = "Timer already running"
    MSG_TIMER_FINISHED = "Timer already finished"
    MSG_TIMER_NOT_STARTED = "Timer not started"

    def __init__(self) -> None:
        self.start_time = None
        self.stop_time = None

    @property
    def running(self) -> bool:
        """Return whether timer is running"""
        return self.start_time is not None and self.stop_time is None

    @property
    def finished(self) -> bool:
        """Return whether timer is finished"""
        return self.start_time is not None and self.stop_time is not None

    @property
    def elapsed(self) -> float:
        """Return elapsed time"""
        if self.finished:
            return self.stop_time - self.start_time
        if self.start_time is not None:
            return timeit.default_timer() - self.start_time
        return 0.0

    def __str__(self) -> str:
        """Return string representation"""
        return f"{type(self).__name__} (start={self.start_time} stop={self.stop_time} elapsed={self.elapsed})"

    def start(self) -> None:
        """Start timer"""
        if self.finished:
            raise TimerException(self.MSG_TIMER_FINISHED)
        if self.running:
            raise TimerException(self.MSG_TIMER_RUNNING)
        self.start_time = timeit.default_timer()

    def stop(self) -> float:
        """Stop timer

        Returns:
            Elapsed time
        """
        if self.start_time is None:
            raise TimerException(self.MSG_TIMER_NOT_STARTED)
        if self.finished:
            raise TimerException(self.MSG_TIMER_FINISHED)
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

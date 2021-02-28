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
    def is_running(self) -> bool:
        """Return whether timer is running"""
        return self.start_time is not None and self.stop_time is None

    @property
    def is_finished(self) -> bool:
        """Return whether timer is finished"""
        return self.start_time is not None and self.stop_time is not None

    @property
    def elapsed(self) -> float:
        """Return elapsed time"""
        if self.is_finished:
            return self.stop_time - self.start_time
        if self.start_time is not None:
            return timeit.default_timer() - self.start_time
        return 0.0

    def __str__(self) -> str:
        """Return string representation"""
        return f"{type(self).__name__} (start={self.start_time} stop={self.stop_time} elapsed={self.elapsed})"

    def start(self) -> None:
        """Start timer"""
        if self.is_finished:
            raise TimerException(self.MSG_TIMER_FINISHED)
        if self.is_running:
            raise TimerException(self.MSG_TIMER_RUNNING)
        self.start_time = timeit.default_timer()

    def stop(self) -> float:
        """Stop timer

        Returns:
            Elapsed time
        """
        if self.start_time is None:
            raise TimerException(self.MSG_TIMER_NOT_STARTED)
        if self.is_finished:
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

    def __str__(self):
        """Return string representation"""
        return f"{type(self).__name__} (start={self.start_time} stop={self.stop_time} elapsed={self.elapsed} splits={self.splits})"

    @property
    def elapsed_splits(self) -> [float]:
        """Return splits as elapsed time"""
        return [(split - self.start_time) for split in self.splits] + [self.elapsed]

    @property
    def elapsed_laps(self) -> [float]:
        """Return laps as elapsed time"""
        times = []
        previous = self.start_time
        for split in self.splits:
            times.append(split - previous)
            previous = split
        if self.is_finished:
            times.append(self.stop_time - previous)
        else:
            times.append(timeit.default_timer() - previous)
        return times

    def split(self) -> float:
        """Add a split to timer

        Returns:
            Elapsed time
        """
        self.splits.append(timeit.default_timer())
        return self.elapsed

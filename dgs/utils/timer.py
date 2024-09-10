"""
Models, functions and helpers for timing operations.
"""

import time
from collections import UserDict, UserList
from datetime import timedelta


class DifferenceTimer(UserList):
    """A simple timer based on time differences, with a few helper functions."""

    data: list[float]
    """A list containing time differences in seconds."""

    def __init__(self) -> None:
        super().__init__()

    def add(self, prev_time: float, now: float = None) -> float:
        """Append the difference between a previous time and the current time to this timer.

        Args:
            prev_time: The previous time in seconds. The value is used to compute the time difference in seconds to now.
            now: The current time in seconds.
                If not provided, the current time is used.
                Can be used to make sure,
                that the time difference is computed relative to a specific time when computing multiple values.

        Returns:
             diff: The difference between now and the previous value in seconds.
        """
        if now is None:
            now = time.time()
        diff = now - prev_time
        self.data.append(diff)
        return diff

    def average(self) -> float:
        """Return the average time in seconds."""
        if len(self.data) == 0:
            return 0.0
        return self.sum() / len(self.data)

    def sum(self) -> float:
        """Return the absolute sum of all the individual timings in seconds."""
        if len(self.data) == 0:
            return 0.0
        return sum(self.data)

    def avg_hms(self) -> str:
        """Get the total average time and return it as str with `HH:MM:SS`."""
        return str(timedelta(seconds=round(self.average())))

    def sum_hms(self) -> str:
        """Get the summed-up time and return it as str with `HH:MM:SS`."""
        return str(timedelta(seconds=round(self.sum())))

    def print(self, name: str, prepend: str, hms: bool = False) -> str:  # pragma: no cover
        """Generate string for printing, containing average and total time."""
        if hms:
            return (
                f"{str(prepend)}: "
                f"{str(name)} time average: {self.avg_hms()} [H:MM:SS], "
                f"{str(name)} time total: {self.sum_hms()} [H:MM:SS]"
            )
        return (
            f"{str(prepend)}: "
            f"{str(name)} time average: {self.average():.1f} [s], "
            f"{str(name)} time total: {self.sum():.1f} [s]"
        )


class DifferenceTimers(UserDict):
    """Object to store the information of multiple :class:`DifferenceTimer` objects."""

    data: dict[str, DifferenceTimer]

    def __init__(self, names: list[str] = None) -> None:
        super().__init__({name: DifferenceTimer() for name in names or []})

    def __getitem__(self, item) -> DifferenceTimer:
        return self.data[item]

    def add(self, name: str, prev_time: float, now: float = None) -> float:
        """Add a new time difference to the timer with the given name.
        Creates a new timer if it does not exist yet.

        Args:
            name: The name of the timer.
            prev_time: The previous time in seconds.
                This value is used to compute the time difference in seconds relative to now.
            now: The current time in seconds.

        Returns:
            The difference between now and the previous value in seconds.
        """
        if now is None:
            now = time.time()
        if name not in self.data:
            self.data[name] = DifferenceTimer()
        return self.data[name].add(prev_time=prev_time, now=now)

    def add_multiple(self, prev_times: dict[str, float]) -> dict[str, float]:
        """Add a bunch of new time differences to the respective timers.
        Creates new timers if they do not exist yet.

        Args:
            prev_times: A dict mapping the name of the timer to the previous time in seconds.

        Returns:
            A dict containing the time differences in seconds for each of the named timers.
        """
        diffs = {}
        now = time.time()
        for name, prev_time in prev_times.items():
            if name not in self.data:
                self.data[name] = DifferenceTimer()
            diffs[name] = self.data[name].add(prev_time=prev_time, now=now)
        return diffs

    def print(self, prepend: str, hms: bool = False) -> str:  # pragma: no cover
        """Generate a string for printing, containing average and total time for all timers."""
        s = prepend + ":\n"
        for name, timer in self.data.items():
            s += timer.print(name, prepend="", hms=hms) + "\n"
        return s

    def get_sums(self) -> dict[str, float]:
        """Return the summed-up times for all timers."""
        return {name: timer.sum() for name, timer in self.data.items()}

    def get_avgs(self) -> dict[str, float]:
        """Return the average times for all timers."""
        return {name: timer.average() for name, timer in self.data.items()}

    def get_last(self) -> dict[str, float]:
        """Return the last time difference for all timers."""
        return {name: timer.data[-1] for name, timer in self.data.items()}

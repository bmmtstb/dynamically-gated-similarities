"""
Models, functions and helpers for timing operations.
"""

import time
from collections import UserList
from datetime import timedelta


class DifferenceTimer(UserList):
    """A simple timer based on time differences, with a few helper functions."""

    data: list[float]
    """A list containing time differences in seconds."""

    def __init__(self) -> None:
        super().__init__()

    def add(self, prev_time: float) -> float:
        """Append the difference between a previous time and the current time to this timer.

        Args:
            prev_time: The previous time in seconds. The value is used to compute the time difference in seconds to now.

        Returns:
             diff: The difference between now and the previous value in seconds.
        """
        diff = time.time() - prev_time
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
        """Generate string for printin, containing average and total time."""
        if hms:
            return (
                f"{str(prepend)}: "
                f"{str(name)} time average: {self.avg_hms()} [H:MM:SS], "
                f"{str(name)} time total: {self.sum_hms()} [H:MM:SS]"
            )
        return (
            f"{str(prepend)}: "
            f"{str(name)} time average: {self.average():.1} [s], "
            f"{str(name)} time total: {self.sum():.1} [s]"
        )

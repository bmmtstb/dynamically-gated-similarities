"""
Example of what a tracker might look like
"""

from dgs.tracker_api import DGSTracker


if __name__ == "__main__":
    tracker = DGSTracker(tracker_cfg="./configs/simplest_tracker.yaml")
    tracker.run()

"""
Classes and helpers for Track, Tracks, and other tracking related objects.
"""

from collections import deque, UserDict
from copy import deepcopy
from enum import Enum

import torch

from dgs.utils.config import DEF_CONF
from dgs.utils.state import collate_states, State

TrackID = int
"""The ID of any given track is a positive integer."""


class TrackStatus(Enum):
    """Enumerate for handling the status of a :class:`.Track`.

    A track can be deleted and re-activated.
    If a track was 'Inactive' and becomes 'Active' again its status is simply 'Active'.
    """

    New = 0
    Active = 1
    Inactive = 2
    Reactivated = 3
    Removed = 4


class TrackStatistics:
    """Data object to save and analyze the statistics of some tracks."""

    # active
    new: list[TrackID]
    reactivated: list[TrackID]
    found: list[TrackID]
    still_active: list[TrackID]

    # inactive
    lost: list[TrackID]
    still_inactive: list[TrackID]

    # removed
    removed: list[TrackID]

    def __init__(self) -> None:
        self.clear()

    # ######### #
    # Functions #
    # ######### #

    def print(self, logger, frame_idx: int) -> None:  # pragma: no cover
        """Print the current Track statistics. Debug only."""
        logger.debug(f"===========Frame{frame_idx}==========")
        logger.debug(
            f"Active: {self.active} of which {self.new} are new, "
            f"{self.found} are re-found, and {self.reactivated} are reactivated."
        )
        logger.debug(f"Inactive: {self.inactive} of which {self.lost} are lost.")
        logger.debug(f"Removed: {self.removed}")

    def clear(self) -> None:
        """Clear the current Track statistics. Mostly used for tests."""
        # active
        self.new = []
        self.reactivated = []
        self.found = []
        self.still_active = []

        # inactive
        self.lost = []
        self.still_inactive = []

        # removed
        self.removed = []

    # ########## #
    # Properties #
    # ########## #

    @property
    def active(self) -> set[TrackID]:
        return set(self.new + self.reactivated + self.found + self.still_active)

    @property
    def inactive(self) -> set[TrackID]:
        return set(self.lost + self.still_inactive)

    # ######### #
    # Number of #
    # ######### #

    @property
    def nof_active(self) -> int:
        return len(self.active)

    @property
    def nof_found(self) -> int:
        return len(self.found)

    @property
    def nof_inactive(self) -> int:
        return len(self.inactive)

    @property
    def nof_lost(self) -> int:
        return len(self.lost)

    @property
    def nof_new(self) -> int:
        return len(self.new)

    @property
    def nof_reactivated(self) -> int:
        return len(self.reactivated)

    @property
    def nof_removed(self) -> int:
        return len(self.removed)

    @property
    def nof_still_active(self) -> int:
        return len(self.still_active)

    @property
    def nof_still_inactive(self) -> int:
        return len(self.still_inactive)


class Track:
    """A Track is a single (de-)queue containing multiple :class:`.State` s, keeping the last N states.

    Args:
        N: The max length of this track.
        states: A list of :class:`.State` objects, describing the initial values of this track.
            Default None.
        tid: The Track ID of this object.
            Default -1.
    """

    _N: int
    """Maximum number of states in this Track."""
    _states: deque
    """The deque of the current states with a max length of _N."""
    _id: TrackID
    """The Track-ID of this Track."""

    _status: TrackStatus
    """The status of this Track."""
    _start_frame: int
    """The number describing the first frame this Track was visible."""
    _nof_active: int = 0
    """The number of frames this track has been active."""

    def __init__(self, N: int, curr_frame: int, states: list[State] = None, tid: int = -1) -> None:
        # max nof states
        if N <= 0:
            raise ValueError(f"N must be greater than 0 but got '{N}'.")
        self._N = N

        # track-id
        self.id = tid

        # status and frame management
        self._status = TrackStatus.New
        self._start_frame = curr_frame
        self._nof_active = len(states) if states is not None else 0

        # already existing states
        if states is not None and len(states) and any(state.B != 1 for state in states):
            raise ValueError(f"The batch size of all the States '{[state.B for state in states]}' must be 1.")
        self._states = deque(iterable=states if states else [], maxlen=N)

    def __repr__(self) -> str:
        return f"Track-{self.id}-{len(self)}-{self._start_frame}"

    def __getitem__(self, index: TrackID) -> State:
        return self._states[index]

    def __len__(self) -> int:
        return len(self._states)

    def __eq__(self, other: "Track") -> bool:
        """Return whether another Track is equal to self."""
        if not isinstance(other, Track):
            return False
        variable_equality: bool = (
            self.N == other.N
            and self.id == other.id
            and self.status == other.status
            and self.nof_active == other.nof_active
            and self._start_frame == other._start_frame
        )
        if len(self) == 0 and len(other) == 0:
            return variable_equality
        return (
            variable_equality
            and len(self._states) == len(other._states)
            and all(s == other[i] for i, s in enumerate(self._states))
        )

    # ########## #
    # Properties #
    # ########## #

    @property
    def status(self) -> TrackStatus:
        return self._status

    @property
    def nof_active(self) -> int:
        return self._nof_active

    @nof_active.setter
    def nof_active(self, value: int) -> None:
        self._nof_active = value

    @property
    def id(self) -> TrackID:
        return self._id

    @id.setter
    def id(self, value: TrackID):
        if isinstance(value, torch.Tensor) and (value.ndim == 0 or (value.ndim == 1 and len(value) == 1)):
            self._id = int(value.item())
        elif isinstance(value, int):
            self._id = value
        else:
            raise NotImplementedError(f"unknown type for ID, expected int but got '{type(value)}' - '{value}'")

    @property
    def N(self) -> int:
        """Get the max length of this Track."""
        return self._N

    @property
    def device(self) -> torch.device:
        """Get the device of every tensor in this Track."""
        if len(self) == 0:
            raise ValueError("Can not get the device of an empty Track.")
        device = self._states[-1].device
        assert all(state.device == device for state in self._states), "Not all tensors are on the same device"
        return device

    # ############## #
    # State handling #
    # ############## #

    def append(self, state: State) -> None:
        """Append a new state to the Track."""
        if state.B != 1:
            raise ValueError(f"A Track should only get a State with the a batch size of 1, but got {state.B}.")
        self._states.append(state)
        self.set_active()
        self._nof_active += 1

    def get_all(self) -> State:
        """Get all the states from the Track and stack them into a single :class:`State`."""
        if len(self) == 0:
            raise ValueError("Can not stack the items of an empty Track.")
        return collate_states(list(self._states))

    # ############### #
    # Status handling #
    # ############### #
    def set_active(self) -> None:
        self._status = TrackStatus.Active

    def set_inactive(self) -> None:
        self._status = TrackStatus.Inactive
        self._nof_active = 0

    def set_removed(self) -> None:
        self._status = TrackStatus.Removed
        self._nof_active = 0
        self._id = -1  # unset tID

    def set_reactivated(self, tid: TrackID) -> None:
        self._status = TrackStatus.Reactivated
        self._nof_active = 0
        self._id = tid

    def set_status(self, status: TrackStatus, tid: TrackID = 0) -> None:
        """Set the status of this Track."""
        if status == TrackStatus.Active:
            self.set_active()
        elif status == TrackStatus.Inactive:
            self.set_inactive()
        elif status == TrackStatus.Removed:
            self.set_removed()
        elif status == TrackStatus.Reactivated:
            self.set_reactivated(tid)
        elif status == TrackStatus.New:
            self._status = TrackStatus.New
        else:
            raise ValueError(f"Unknown TrackStatus {status}")  # pragma: no cover

    def age(self, curr_frame: int) -> int:
        """Get the age of this track (in frames).

        The age does not account for frames where the track has been deleted.
        """
        return curr_frame - self._start_frame

    # ####### #
    # Utility #
    # ####### #

    def to(self, *args, **kwargs) -> "Track":
        """Call ``.to()`` like you do with any other ``torch.Tensor``."""
        for i, state in enumerate(self._states):
            self._states[i] = state.to(*args, **kwargs)
        return self

    def clear(self) -> None:
        """Clear all the states from the Track."""
        self._states.clear()
        self._nof_active = 0

    def copy(self) -> "Track":
        """Return a (deep) copy of self."""
        t = Track(
            N=self.N,
            curr_frame=self._start_frame,
            states=[s.copy() for s in self._states],
            tid=self.id,
        )
        t.nof_active = self._nof_active
        t.set_status(status=self._status, tid=self.id)
        return t


class Tracks(UserDict):
    """Multiple Track objects stored as a dictionary,
    where the Track is the value and the key is this tracks' unique ID.
    """

    # pylint: disable=too-many-public-methods

    _N: int
    """The maximum number of frames in each track."""

    data: dict[TrackID, Track]
    """All the Tracks that are currently tracked, including inactive Tracks as mapping 'Track-ID -> Track'."""

    inactive: dict[TrackID, int]
    """All the inactive Tracks as 'Track-ID -> number of inactive frames / steps'."""

    inactivity_threshold: int
    """The number of steps a Track can be inactive before deleting it."""

    removed: dict[TrackID, Track]
    """All the Tracks that have been removed, to be able to reactivate them."""

    _curr_frame: int
    """The number of the current frame."""

    def __init__(self, N: int, thresh: int = None, start_frame: int = 0) -> None:
        super().__init__()

        # set N - the maximum length of every track
        if N <= 0:
            raise ValueError(f"N must be greater than 0 but got '{N}'")
        self._N = N

        # set the inactivity threshold
        if thresh is None:
            self.inactivity_threshold = DEF_CONF.tracks.inactivity_threshold
        elif not isinstance(thresh, int):
            raise TypeError(f"Threshold is expected to be int or None, but got {thresh}")
        elif thresh < 0:
            raise ValueError(f"Threshold must be positive, got {thresh}.")
        else:
            self.inactivity_threshold = thresh

        self.reset()

        # make sure to set the initial current frame after resetting
        self._curr_frame = start_frame

    def __len__(self) -> int:
        """Get the length of data.
        If you want the number of active or inactive Tracks,
        use :meth:`.nof_active` and :meth:`.nof_inactive` respectively.
        If you want the age, use :meth:`.age`.
        """
        return len(self.data)

    def __eq__(self, other: "Tracks") -> bool:
        """Check the equality of two Tracks.

        This method does not validate whether the removed Tracks are equal.
        """
        if not isinstance(other, Tracks):
            return False
        return (
            self.inactive == other.inactive
            and self.inactivity_threshold == other.inactivity_threshold
            and self._curr_frame == other._curr_frame
            and set(self.data.keys()) == set(other.data.keys())
            and all(t == other.data[k] for k, t in self.data.items())
        )

    def __getitem__(self, key: TrackID) -> Track:
        """Given the Track-ID return the Track."""
        return self.data[key]

    def __repr__(self) -> str:
        return f"Tracks-{self.age}-{self.ids_active}"

    # ########## #
    # Properties #
    # ########## #

    @property
    def N(self) -> int:
        return self._N

    @property
    def age(self) -> int:
        return self._curr_frame

    @property
    def ages(self) -> dict[int, int]:
        """Get the age of all the tracks (in frames)."""
        return {i: t.age(self._curr_frame) for i, t in self.data.items()}

    @property
    def ids(self) -> set[TrackID]:
        """Get all the track-IDs in this object."""
        return set(int(k) for k in self.data.keys())

    @property
    def ids_active(self) -> set[TrackID]:
        """Get all the track-IDs currently active."""
        return self.ids - self.ids_inactive

    @property
    def ids_inactive(self) -> set[TrackID]:
        """Get all the track-IDs currently inactive."""
        return set(int(k) for k in self.inactive.keys())

    @property
    def ids_removed(self) -> set[TrackID]:
        """Get all the track-IDs that have been deleted."""
        return set(int(k) for k in self.removed.keys())

    @property
    def nof_active(self) -> int:
        """Get the number of active Tracks."""
        return len(self.data) - len(self.inactive)

    @property
    def nof_inactive(self) -> int:
        """Get the number of inactive Tracks."""
        return len(self.inactive)

    @property
    def nof_removed(self) -> int:
        """Get the number of Tracks that have been removed."""
        return len(self.removed)

    # ######################## #
    # State and Track Handling #
    # ######################## #

    def remove_tid(self, tid: TrackID) -> None:
        """Given a Track-ID, remove the track associated with it from this object."""
        if tid not in self.data:
            raise KeyError(f"Track-ID {tid} can not be deleted, because it is not present in Tracks.")

        self.data[tid].set_removed()
        self.removed[tid] = self.data.pop(tid)

        self.inactive.pop(tid, None)

    def remove_tids(self, tids: list[TrackID]) -> None:
        for tid in tids:
            self.remove_tid(tid)

    def is_active(self, tid: TrackID) -> bool:
        """Return whether the given Track-ID is active."""
        return tid in self.data and tid not in self.inactive

    def is_inactive(self, tid: TrackID) -> bool:
        """Return whether the given Track-ID is inactive."""
        return tid in self.data and tid in self.inactive

    def is_removed(self, tid: TrackID) -> bool:
        """Return whether the given Track-ID has been removed."""
        return tid not in self.data and tid in self.removed

    def add(self, tracks: dict[TrackID, State], new: list[State]) -> tuple[list[TrackID], TrackStatistics]:
        """Given tracks with existing Track-IDs update those and create new Tracks for States without Track-IDs.
        Additionally,
        mark Track-IDs that are not in either of the inputs as unseen and therefore as inactive for one more step.

        Returns:
            The Track-IDs of the new_tracks in the same order as provided.
        """
        stats = TrackStatistics()

        inactive_ids = self.ids - set(int(k) for k in tracks.keys())

        # get the next free ID and create track(s)
        new_tids = self.add_empty_tracks(len(new))

        # add the new state to the new tracks
        for tid, new_state in zip(new_tids, new):
            self._update_track(tid=tid, add_state=new_state, stats=stats)
            stats.new.append(tid)

        # add state to Track and remove track from inactive if present
        for tid, new_state in tracks.items():
            self._update_track(tid=tid, add_state=new_state, stats=stats)

        self._handle_inactive(tids=inactive_ids, stats=stats)

        # step to the next frame
        self._next_frame()
        return new_tids, stats

    def _next_frame(self) -> None:
        self._curr_frame += 1

    def get_states(self) -> list[State]:
        """Get the last state of **every** track in this object as a :class:`State`."""
        states: list[State] = []
        tids: list[TrackID] = []

        for tid, track in self.data.items():
            states.append(track[-1])
            tids.append(tid)

        return states

    def get_active_states(self) -> list[State]:
        """Get the last state of every **active** track in this object as a :class:`State`."""
        states: list[State] = []

        for tid, track in self.data.items():
            if tid in self.inactive:
                continue
            states.append(track[-1])

        return states

    def add_empty_tracks(self, n: int = 1) -> list[TrackID]:
        """Given a Track, compute the next track-ID, and save this track in data using this ID.

        Args:
            n: The number of new Tracks to add.

        Returns:
             tids: The track-IDs of the added tracks.

        """
        tids = []
        for _ in range(n):
            tid = self._get_next_id()
            self.data[tid] = Track(N=self._N, curr_frame=self._curr_frame, tid=tid)
            tids.append(tid)
        return tids

    def reactivate_track(self, tid: TrackID) -> None:
        """Given the track-ID of a previously removed track, reactivate it."""
        if tid not in self.removed:
            raise KeyError(f"Track-ID {tid} not present in removed Tracks.")
        self.data[tid] = self.removed.pop(tid)
        self.data[tid].set_reactivated(tid)

        # todo should the states of the track be removed / cleared ?

    def _update_track(self, tid: TrackID, add_state: State, stats: TrackStatistics) -> None:
        """Use the track-ID to update a track given an additional :class:`State` for the :class:`Track`.
        Will additionally remove the tid from the inactive Tracks.

        Returns:
            Whether this track has been reactivated with this update.
        """
        if tid not in self.data.keys():
            if tid not in self.removed.keys():
                raise KeyError(f"Track-ID {tid} neither present in the current or previously removed Tracks.")
            # reactivate previously removed track
            self.reactivate_track(tid)
            stats.reactivated.append(tid)
        elif tid in self.inactive:
            # update inactive
            self.inactive.pop(tid)
            stats.found.append(tid)
        else:
            stats.still_active.append(tid)

        # append state to track
        self.data[tid].append(state=add_state)
        # add track id to state
        self.data[tid][-1]["pred_tid"] = torch.tensor(tid, dtype=torch.long, device=add_state.device).flatten()

    def _handle_inactive(self, tids: set[TrackID], stats: TrackStatistics) -> None:
        """Given the Track-IDs of the Tracks that haven't been seen this step, update the inactivity tracker.
        Create the counter for inactive Track-IDs and update existing counters.
        Additionally, remove tracks that have been inactive for too long.
        """
        for tid in tids:
            if tid in self.inactive.keys():
                self.inactive[tid] += 1

                if self.inactive[tid] >= self.inactivity_threshold:
                    self.remove_tid(tid)
                    stats.removed.append(tid)
                else:
                    stats.still_inactive.append(tid)
            else:
                self.inactive[tid] = 1
                self.data[tid].set_inactive()
                stats.lost.append(tid)

    def _get_next_id(self) -> TrackID:
        """Get the next free track-ID."""
        if len(self.data) == 0:
            return 0
        return max(self.data.keys()) + 1

    # ####### #
    # Utility #
    # ####### #

    def reset(self) -> None:
        """Reset this object."""
        self.data = {}
        self.inactive = {}
        self.removed = {}
        self._curr_frame = 0

    def reset_deleted(self) -> None:
        """Fully remove the deleted Tracks."""
        self.removed = {}

    def copy(self) -> "Tracks":
        """Return a (deep) copy of this object."""
        new_t = Tracks(N=self.N, thresh=self.inactivity_threshold)
        new_t.data = {i: t.copy() for i, t in self.data.items()}
        new_t.inactive = deepcopy(self.inactive)
        new_t.removed = deepcopy(self.removed)
        return new_t

    def to(self, *args, **kwargs) -> "Tracks":
        """Create function similar to :func:`torch.Tensor.to` ."""
        self.data = {i: t.to(*args, **kwargs) for i, t in self.data.items()}
        self.removed = {i: t.to(*args, **kwargs) for i, t in self.removed.items()}
        return self

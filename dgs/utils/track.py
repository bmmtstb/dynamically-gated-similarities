"""
Classes and helpers for Track, Tracks, and other tracking related objects.
"""

from collections import deque, UserDict
from copy import deepcopy
from enum import Enum

import torch
from torchvision import tv_tensors

from dgs.utils.config import DEF_CONF
from dgs.utils.state import collate_states, State


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


class Track:
    """A Track is a single (de-)queue containing multiple :class:`.State` s, keeping the last N states.

    Args:
        N: The max length of this track.
        states: A list of :class:`.State` objects, describing the initial values of this track.
            Default None.
        B: The batch size, every :class:`.State` object should have.
            Default 1.
        tid: The Track ID of this object.
            Default -1.
    """

    _N: int
    """Maximum number of states in this Track."""
    _states: deque
    """The deque of the current states with a max length of _N."""
    _id: int
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

    def __getitem__(self, index: int) -> State:
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
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, value: int):
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

    def set_reactivated(self, tid: int) -> None:
        self._status = TrackStatus.Reactivated
        self._nof_active = 0
        self._id = tid

    def set_status(self, status: TrackStatus, tid: int = 0) -> None:
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
            raise ValueError(f"Unknown TrackStatus {status}")

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

    _N: int
    """The maximum number of frames in each track."""

    data: dict[int, Track]
    """All the Tracks that are currently tracked, including inactive Tracks as mapping 'Track-ID -> Track'."""

    inactive: dict[int, int]
    """All the inactive Tracks as 'Track-ID -> number of inactive frames / steps'."""

    inactivity_threshold: int
    """The number of steps a Track can be inactive before deleting it."""

    removed: dict[int, Track]
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

    def __getitem__(self, key: int) -> Track:
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
    def ids(self) -> set[int]:
        """Get all the track-IDs in this object."""
        return set(int(k) for k in self.data.keys())

    @property
    def ids_active(self) -> set[int]:
        """Get all the track-IDs currently active."""
        return self.ids - self.ids_inactive

    @property
    def ids_inactive(self) -> set[int]:
        """Get all the track-IDs currently inactive."""
        return set(int(k) for k in self.inactive.keys())

    @property
    def ids_removed(self) -> set[int]:
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

    # ######################## #
    # State and Track Handling #
    # ######################## #

    def remove_tid(self, tid: int) -> None:
        """Given a Track-ID, remove the track associated with it from this object."""
        if tid not in self.data:
            raise KeyError(f"Track-ID {tid} can not be deleted, because it is not present in Tracks.")

        self.data[tid].set_removed()
        self.removed[tid] = self.data.pop(tid)

        self.inactive.pop(tid, None)

    def remove_tids(self, tids: list[int]) -> None:
        for tid in tids:
            self.remove_tid(tid)

    def add(self, tracks: dict[int, State], new_tracks: list[State]) -> list[int]:
        """Given tracks with existing Track-IDs update those and create new Tracks for States without Track-IDs.
        Additionally,
        mark Track-IDs that are not in either of the inputs as unseen and therefore as inactive for one more step.

        Returns:
            The Track-IDs of the new_tracks in the same order as provided.
        """
        newly_inactive_ids = self.ids - set(int(k) for k in tracks.keys())

        # get the next free ID and create track(s)
        new_tids = self.add_empty_tracks(len(new_tracks))
        # add the new state to the new tracks
        for tid, new_state in zip(new_tids, new_tracks):
            self._update_track(tid=tid, add_state=new_state)

        # add state to Track and remove track from inactive if present
        for tid, new_state in tracks.items():
            self._update_track(tid=tid, add_state=new_state)

        self._handle_inactive(tids=newly_inactive_ids)

        self._next_frame()

        return new_tids

    def _next_frame(self) -> None:
        self._curr_frame += 1

    def get_states(self) -> State:
        """Get the last state of every track in this object as a :class:`State`."""
        states: list[State] = []
        tids: list[int] = []

        for tid, track in self.data.items():
            states.append(track[-1])
            tids.append(tid)

        if len(states) == 0:
            return State(
                bbox=tv_tensors.BoundingBoxes(
                    torch.zeros((0, 4)), canvas_size=(0, 0), format="XYWH", dtype=torch.float32, requires_grad=False
                ),
                validate=False,
            )
        state = collate_states(states)
        return state

    def add_empty_tracks(self, n: int = 1) -> list[int]:
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

    def reactivate_track(self, tid: int) -> None:
        """Given the track-ID of a previously removed track, reactivate it."""
        if tid not in self.removed:
            raise KeyError(f"Track-ID {tid} not present in removed Tracks.")
        self.data[tid] = self.removed.pop(tid)
        self.data[tid].set_reactivated(tid)

        # todo should the states of the track be removed / cleared ?

    def _update_track(self, tid: int, add_state: State) -> None:
        """Use the track-ID to update a track given an additional :class:`State` for the :class:`Track`.
        Will additionally remove the tid from the inactive Tracks.
        """
        if tid not in self.data.keys():
            if tid not in self.removed.keys():
                raise KeyError(f"Track-ID {tid} neither present in the current or previously removed Tracks.")
            self.reactivate_track(tid)

        # append state to track
        self.data[tid].append(state=add_state)

        # update inactive
        self.inactive.pop(tid, None)

    def _handle_inactive(self, tids: set[int]) -> None:
        """Given the Track-IDs of the Tracks that haven't been seen this step, update the inactivity tracker.
        Create the counter for inactive Track-IDs and update existing counters.
        Additionally, remove tracks that have been inactive for too long.
        """
        for tid in tids:
            if tid in self.inactive.keys():
                self.inactive[tid] += 1
            else:
                self.inactive[tid] = 1
                self.data[tid].set_inactive()
            if self.inactive[tid] >= self.inactivity_threshold:
                self.remove_tid(tid)

    def _get_next_id(self) -> int:
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

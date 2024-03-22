"""
Classes and helpers for Track, Tracks, and other tracking related objects.
"""

from collections import deque, UserDict
from copy import deepcopy

import torch
from torchvision import tv_tensors

from dgs.utils.config import DEF_CONF
from dgs.utils.state import collate_states, State


class Track:
    """A Track is a single (de-)queue containing multiple :class:`State` s, keeping the last N states."""

    _B: int
    """The Batch size of every State object in this Track."""
    _N: int
    """Maximum number of states in this Track."""
    _states: deque
    """The deque of the current states with a max length of _N."""

    def __init__(self, N: int, states: list[State] = None, B: int = 1) -> None:

        # max nof states
        if N <= 0:
            raise ValueError(f"N must be greater than 0 but got {N}")
        self._N = N

        # batch size
        if B <= 0:
            raise ValueError(f"B must be greater than 0 but got {B}")
        self._B = B

        # already existing states
        if states is not None and len(states) and any(state.B != B for state in states):
            raise ValueError(
                f"The batch size of all the States '{[state.B for state in states]}' "
                f"must have the same shape as the given shape '{B}'."
            )
        self._states = deque(iterable=states if states else [], maxlen=N)

    def __getitem__(self, index: int) -> State:
        return self._states[index]

    def __len__(self) -> int:
        return len(self._states)

    def __eq__(self, other: "Track") -> bool:
        """Return whether another Track is equal to self."""
        if not isinstance(other, Track):
            return False
        if len(self) == 0 and len(other) == 0:
            return self._N == other._N and self._B == other._B
        return (
            self._N == other._N
            and self._B == other._B
            and len(self._states) == len(other._states)
            and all(s == other[i] for i, s in enumerate(self._states))
        )

    @property
    def N(self) -> int:
        """Get the max length of this Track."""
        return self._N

    @property
    def B(self) -> int:
        """Get the batch size of every State object in this Track. Should be 1 in most cases."""
        return self._B

    @property
    def device(self) -> torch.device:
        """Get the device of every tensor in this Track."""
        if len(self) == 0:
            raise ValueError("Can not get the device of an empty Track.")
        device = self._states[-1].device
        assert all(state.device == device for state in self._states), "Not all tensors are on the same device"
        return device

    def append(self, state: State) -> None:
        """Append a new state to the Track."""
        if state.B != self.B:
            raise ValueError(
                f"A Track should only get a State with the same batch size of B ({self.B}), but got {state.B}."
            )
        self._states.append(state)

    def to(self, *args, **kwargs) -> "Track":
        """Call ``.to()`` like you do with any other ``torch.Tensor``."""
        for i, state in enumerate(self._states):
            self._states[i] = state.to(*args, **kwargs)
        return self

    def get_all(self) -> State:
        """Get all the states from the Track and stack them into a single :class:`State`."""
        if len(self) == 0:
            raise ValueError("Can not stack the items of an empty Track.")
        return collate_states(list(self._states))

    def clear(self) -> None:
        """Clear all the states from the Track."""
        self._states.clear()

    def copy(self) -> "Track":
        """Return a (deep) copy of self."""
        return Track(N=self._N, states=[s.copy() for s in self._states], B=self._B)


class Tracks(UserDict):
    """Multiple Track objects stored as a dictionary,
    where the Track is the value and the key is this tracks' unique ID.
    """

    data: dict[int, Track]
    """All the Tracks that are currently tracked, including inactive Tracks as mapping 'Track-ID -> Track'."""

    inactive: dict[int, int]
    """All the inactive Tracks as 'Track-ID -> number of inactive frames / steps'."""

    inactivity_threshold: int
    """The number of steps a Track can be inactive before deleting it."""

    def __init__(self, thresh: int = None) -> None:
        super().__init__()

        # set the inactivity threshold
        if thresh is None:
            self.inactivity_threshold = DEF_CONF.tracks.inactivity_threshold
        elif thresh < 0:
            raise ValueError(f"Threshold must be positive, got {thresh}.")
        else:
            self.inactivity_threshold = thresh

        self.reset()

    def __len__(self) -> int:
        """Get the length of data.
        If you want the number of active or inactive Tracks,
        use :meth:`.nof_active` and :meth:`.nof_inactive` respectively.
        """
        return len(self.data)

    def __eq__(self, other: "Tracks") -> bool:
        """Check the equality of two Tracks."""
        if not isinstance(other, Tracks):
            return False
        return (
            self.inactive == other.inactive
            and self.inactivity_threshold == other.inactivity_threshold
            and set(self.data.keys()) == set(other.data.keys())
            and all(t == other.data[k] for k, t in self.data.items())
        )

    def __getitem__(self, key: int) -> Track:
        """Given the Track-ID return the Track."""
        return self.data[key]

    def reset(self) -> None:
        """Reset this object."""
        self.data = {}
        self.inactive = {}

    def copy(self) -> "Tracks":
        """Return a (deep) copy of this object."""
        new_t = Tracks()
        new_t.data = {i: t.copy() for i, t in self.data.items()}
        new_t.inactive = deepcopy(self.inactive)
        new_t.inactivity_threshold = self.inactivity_threshold
        return new_t

    def remove_tid(self, tid: int) -> None:
        """Given a Track-ID, remove the track associated with it from this object."""
        self.data.pop(tid, None)
        self.inactive.pop(tid, None)

    def to(self, *args, **kwargs) -> "Tracks":
        """Create function similar to :func:`torch.Tensor.to` ."""
        self.data = {i: t.to(*args, **kwargs) for i, t in self.data.items()}
        return self

    def add(self, tracks: dict[int, State], new_tracks: list[Track]) -> list[int]:
        """Given tracks with existing Track-IDs update those and create new Tracks for States without Track-IDs.
        Additionally,
        mark Track-IDs that are not in either of the inputs as unseen and therefore as inactive for one more step.

        Returns:
            The Track-IDs of the new_tracks in the same order as provided.
        """
        newly_inactive_ids = self.ids() - set(int(k) for k in tracks.keys())
        added_ids = []

        # add state to Track and remove track from inactive if present
        for tid, new_state in tracks.items():
            self._update_track(tid=tid, add_state=new_state)

        # get the next free ID and create track
        for new_track in new_tracks:
            new_id = self._add_track(new_track)

            # add track ID to State
            for s in self.data[new_id]:
                s.track_id = torch.tensor([new_id])
            added_ids.append(new_id)

        self._handle_inactive(tids=newly_inactive_ids)

        return added_ids

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
        state.track_id = torch.tensor(tids, device=state.device, dtype=torch.long)
        return state

    def _add_track(self, t: Track) -> int:
        """Given a Track, compute the next track-ID, and save this track in data using this ID.

        Args:
            t: The :class:`Track` to add.

        Returns:
             tid: The track-ID of the added track.

        """
        tid = self._get_next_id()
        self.data[tid] = t
        return tid

    def _update_track(self, tid: int, add_state: State) -> None:
        """Use the track-ID to update a track given an additional :class:`State` for the :class:`Track`.
        Will additionally remove the tid from the inactive Tracks.
        """
        if tid not in self.data.keys():
            raise KeyError(f"Track-ID {tid} not present in Tracks.")

        # set track ID in state and append state to track
        add_state.track_id = torch.tensor([tid])
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
            if self.inactive[tid] >= self.inactivity_threshold:
                self.remove_tid(tid)

    def _get_next_id(self) -> int:
        """Get the next free track-ID."""
        if len(self.data) == 0:
            return 0
        return max(self.data.keys()) + 1

    def ids(self) -> set[int]:
        """Get all the track-IDs in this object."""
        return set(int(k) for k in self.data.keys())

    def ids_active(self) -> set[int]:
        """Get all the track-IDs currently active."""
        return self.ids() - self.ids_inactive()

    def ids_inactive(self) -> set[int]:
        """Get all the track-IDs currently inactive."""
        return set(int(k) for k in self.inactive.keys())

    def nof_active(self) -> int:
        """Get the number of active Tracks."""
        return len(self.ids()) - len(self.inactive)

    def nof_inactive(self) -> int:
        """Get the number of inactive Tracks."""
        return len(self.inactive.keys())

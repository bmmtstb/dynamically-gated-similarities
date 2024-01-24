"""
Implementation if kalman filter for basic pose warping
"""

import torch
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter

from dgs.models.pose_warping.pose_warping import PoseWarpingModule
from dgs.models.states import PoseState, PoseStates
from dgs.utils.types import Config, FilePath, NodePath, Validations

KFWM_validations: Validations = {
    "dim_x": ["dict", ("longer", 0)],
    "dim_z": ["dict", ("longer", 0)],
    "measures": ["optional", ("instance", "list")],
}


class KalmanFilterWarpingModule(PoseWarpingModule):
    r"""Kalman Filter for pose and box warping using `torch_kalman <https://github.com/strongio/torch-kalman>`_ package.

    Module Name
    -----------

    KalmanFilterWarping, KalmanFilterWarpingModule, or KFWM

    Description
    -----------

    A basic Kalman filter using the `filterpy <https://filterpy.readthedocs.io/en/latest/index.html>`_ package.
    Given the current state, predict the next one.
    Will indirectly compute velocities and variances.

    Params
    ------

    dim_x: (dict[str, int])
        For every measure, the number of state variables for the Kalman filter.
        For example, if you are tracking the (x-y)-position of a person with 17 key-points, dim_x would be
        :math:`2 \cdot 17 = 34`.
        This is used to set the default size of P, Q, and u
    dim_z: (dict[str, int])
        Number of measurement inputs.
        For example, if the measurement provides you with bbox-position as (x,y,w,h), dim_z would be 4.
    measures: (list[str], default=["pose", "box"])
        A list of measurement names to compute the Kalman Filter prediction from.
        The variables will be extracted from a given DataSample object using `__getitem__(name)`.
    """

    model: dict[str, KalmanFilter]
    measures: list[str]

    def __init__(self, config: Config, path: NodePath) -> None:
        """"""
        super().__init__(config, path)
        self.validate_params(validations=KFWM_validations)
        self.measures = self.params.get("measures", ["pose", "box"])
        # create a basic KF for every measurement
        for m in self.measures:
            self.model[m] = KalmanFilter(
                dim_x=self.params["dim_x"], dim_z=self.params["dim_z"], dim_u=self.params.get("dim_u", 0)
            )

    def forward(self, ps: PoseStates) -> PoseStates:
        """Given the current pose state, use the kalman filter to predict the next state."""
        curr_state: PoseState = ps.get_states()
        for m in self.measures:
            prediction = self.model[m].predict(curr_state[str(m)])
        return ...

    def forward_pred(self) -> ...:
        """Get `torch_kalman` internal prediction"""

    def load_weights(self, path: FilePath) -> None:
        """..."""
        self.model.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)
        raise NotImplementedError

    def train(self, inp: torch.Tensor, epochs: int = 8) -> None:
        """Train kalman filter prediction.

        References:
            See

        Args:
            inp:
            epochs:

        Returns:

        """

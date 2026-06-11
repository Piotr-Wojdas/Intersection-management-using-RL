"""Observation functions for traffic signals."""

from abc import abstractmethod

import numpy as np
from gymnasium import spaces

from .traffic_signal import TrafficSignal


class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        self.ts = ts

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [
            1 if self.ts.green_phase == i else 0
            for i in range(self.ts.num_green_phases)
        ]  # one-hot encoding
        # Continuous phase progress [0, 1]: 0 = just switched, 1 = at max_green.
        # Richer than a binary min_green flag — the agent can anticipate a forced
        # switch before enforce_max_green fires.
        phase_progress = [
            min(1.0, self.ts.time_since_last_phase_change / max(1, self.ts.max_green))
        ]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        # Outgoing lane densities expose downstream congestion, which is
        # essential for coordination between neighbouring intersections.
        out_density = self.ts.get_out_lanes_density()
        # Normalised accumulated waiting time per lane [0, 1].
        # The reward is diff-waiting-time, so exposing the waiting time directly
        # lets the agent observe what it is rewarded for.
        wait_norm = self.ts.get_normalized_waiting_time_per_lane()
        observation = np.array(
            phase_id + phase_progress + density + queue + out_density + wait_norm,
            dtype=np.float32,
        )
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        n = (
            self.ts.num_green_phases
            + 1  # phase_progress
            + 2 * len(self.ts.lanes)  # density + queue
            + len(self.ts.out_lanes)  # out_density
            + len(self.ts.lanes)  # wait_norm
        )
        return spaces.Box(
            low=np.zeros(n, dtype=np.float32),
            high=np.ones(n, dtype=np.float32),
        )

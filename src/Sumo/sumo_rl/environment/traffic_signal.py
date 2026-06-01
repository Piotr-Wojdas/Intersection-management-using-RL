"""This module contains the TrafficSignal class, which represents a traffic signal in the simulation."""

from typing import Callable, List, Union

# Central SUMO setup (ensure SUMO_HOME/tools on sys.path)
import numpy as np
from gymnasium import spaces


class TrafficSignal:
    """This class represents a Traffic Signal controlling an intersection.

    It is responsible for retrieving information and changing the traffic phase using the Traci API.

    IMPORTANT: It assumes that the traffic phases defined in the .net file are green phases.
    The environment switches directly between those green phases.

    # Observation Space
    The default observation for each traffic signal agent is a vector:

    obs = [phase_one_hot, min_green, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue]

    - ```phase_one_hot``` is a one-hot encoded vector indicating the current active green phase
    - ```min_green``` is a binary variable indicating whether min_green seconds have already passed in the current phase
    - ```lane_i_density``` is the number of vehicles in incoming lane i dividided by the total capacity of the lane
    - ```lane_i_queue``` is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane

    You can change the observation space by implementing a custom observation class. See :py:class:`sumo_rl.environment.observations.ObservationFunction`.

    # Action Space
    Action space is discrete, corresponding to which green phase is going to be open for the next delta_time seconds.

    # Reward Function
    The default reward function is 'diff-waiting-time'. You can change the reward function by implementing a custom reward function and passing to the constructor of :py:class:`sumo_rl.environment.env.SumoEnvironment`.
    """

    # Default min gap of SUMO (see https://sumo.dlr.de/docs/Simulation/Safety.html). Should this be parameterized?
    MIN_GAP = 2.5

    def __init__(
        self,
        env,
        ts_id: str,
        delta_time: int,
        yellow_time: int,
        min_green: int,
        max_green: int,
        enforce_max_green: bool,
        begin_time: int,
        reward_fn: Union[str, Callable, List],
        reward_weights: List[float],
        sumo,
    ):
        """Initializes a TrafficSignal object.

        Args:
            env (SumoEnvironment): The environment this traffic signal belongs to.
            ts_id (str): The id of the traffic signal.
            delta_time (int): The time in seconds between actions.
            yellow_time (int): The time in seconds of the yellow phase.
            min_green (int): The minimum time in seconds of the green phase.
            max_green (int): The maximum time in seconds of the green phase.
            enforce_max_green (bool): If True, the traffic signal will always change phase after max green seconds.
            begin_time (int): The time in seconds when the traffic signal starts operating.
            reward_fn (Union[str, Callable]): The reward function. Can be a string with the name of the reward function or a callable function.
            reward_weights (List[float]): The weights of the reward function.
            sumo (Sumo): The Sumo instance.
        """
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.enforce_max_green = enforce_max_green
        self.green_phase = 0
        self.time_since_last_phase_change = 0
        self.next_action_time = begin_time
        self.last_ts_waiting_time = 0.0
        self.last_reward = None
        self.reward_fn = reward_fn
        self.reward_weights = reward_weights
        self.sumo = sumo

        if isinstance(self.reward_fn, list):
            self.reward_dim = len(self.reward_fn)
            self.reward_list = [
                self._get_reward_fn_from_string(reward_fn)
                for reward_fn in self.reward_fn
            ]
        else:
            self.reward_dim = 1
            self.reward_list = [self._get_reward_fn_from_string(self.reward_fn)]

        if self.reward_weights is not None:
            self.reward_dim = 1  # Since it will be scalarized

        self.reward_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.reward_dim,), dtype=np.float32
        )

        self.observation_fn = self.env.observation_class(self)

        self._build_phases()

        self.lanes = list(
            dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.id))
        )  # Remove duplicates and keep order
        self.out_lanes = [
            link[0][1]
            for link in self.sumo.trafficlight.getControlledLinks(self.id)
            if link
        ]
        self.out_lanes = list(set(self.out_lanes))
        self.lanes_length = {
            lane: self.sumo.lane.getLength(lane) for lane in self.lanes + self.out_lanes
        }

        self.observation_space = self.observation_fn.observation_space()
        self.action_space = spaces.Discrete(self.num_green_phases)

        # Read standing-vehicle penalty params (fallback to defaults if params missing)
        try:
            from src.params import (
                PENDING_VEHICLE_PENALTY_WEIGHT,
                QUEUE_PENALTY_WEIGHT,
                STANDING_PENALTY_WEIGHT,
                STANDING_WAIT_THRESHOLD,
                EXPONENTIAL_WAIT_PENALTY,
                EXP_WAIT_PENALTY_SCALE,
            )
        except Exception:
            STANDING_WAIT_THRESHOLD = 20
            STANDING_PENALTY_WEIGHT = 0.5
            PENDING_VEHICLE_PENALTY_WEIGHT = 0.1
            QUEUE_PENALTY_WEIGHT = 0.05
            EXPONENTIAL_WAIT_PENALTY = False
            EXP_WAIT_PENALTY_SCALE = 0.2

        self._standing_wait_threshold = STANDING_WAIT_THRESHOLD
        self._standing_penalty_weight = STANDING_PENALTY_WEIGHT
        self._pending_vehicle_penalty_weight = PENDING_VEHICLE_PENALTY_WEIGHT
        self._queue_penalty_weight = QUEUE_PENALTY_WEIGHT
        self._exp_wait_penalty_enabled = EXPONENTIAL_WAIT_PENALTY
        self._exp_wait_penalty_scale = EXP_WAIT_PENALTY_SCALE

    def _get_reward_fn_from_string(self, reward_fn):
        if isinstance(reward_fn, str):
            if reward_fn in TrafficSignal.reward_fns.keys():
                return TrafficSignal.reward_fns[reward_fn]
            else:
                raise NotImplementedError(
                    f"Reward function {reward_fn} not implemented"
                )
        return reward_fn

    def _build_phases(self):
        phases = self.sumo.trafficlight.getAllProgramLogics(self.id)[0].phases
        self.green_phases = []
        for phase in phases:
            state = phase.state
            if "y" not in state and (state.count("r") + state.count("s") != len(state)):
                self.green_phases.append(
                    self.sumo.trafficlight.Phase(self.min_green, state)
                )
        self.num_green_phases = len(self.green_phases)
        self.all_phases = self.green_phases.copy()

        if self.env.fixed_ts:
            return

        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.all_phases
        self.sumo.trafficlight.setProgramLogic(self.id, logic)
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)

    @property
    def time_to_act(self):
        """Returns True if the traffic signal should act in the current step."""
        return self.next_action_time == self.env.sim_step

    def update(self):
        """Updates the traffic signal state.

        If the traffic signal should act, it will set the next green phase and update the next action time.
        """
        self.time_since_last_phase_change += 1

    def set_next_phase(self, new_phase: int):
        """Sets what will be the next green phase.

        Args:
            new_phase (int): Number between [0 ... num_green_phases]
        """
        new_phase = int(new_phase)

        # Ensure max green time is enforced if needed
        if (
            self.enforce_max_green
            and new_phase == self.green_phase
            and self.time_since_last_phase_change >= self.max_green
        ):
            new_phase = (
                self.green_phase + 1
            ) % self.num_green_phases  # Next phase is activated

        if (
            self.green_phase == new_phase
            or self.time_since_last_phase_change < self.min_green
        ):
            self.sumo.trafficlight.setRedYellowGreenState(
                self.id, self.all_phases[self.green_phase].state
            )
            self.next_action_time = self.env.sim_step + self.delta_time
        else:
            self.sumo.trafficlight.setRedYellowGreenState(
                self.id,
                self.all_phases[new_phase].state,
            )
            self.green_phase = new_phase
            self.next_action_time = self.env.sim_step + self.delta_time
            self.time_since_last_phase_change = 0

    def compute_observation(self):
        """Computes the observation of the traffic signal."""
        return self.observation_fn()

    def compute_reward(self) -> Union[float, np.ndarray]:
        """Computes the reward of the traffic signal. If it is a list of rewards, it returns a numpy array."""
        # Compute base reward (scalar or vector)
        if self.reward_dim == 1:
            self.last_reward = float(self.reward_list[0](self))
        else:
            base = np.array(
                [reward_fn(self) for reward_fn in self.reward_list], dtype=np.float32
            )
            if self.reward_weights is not None:
                base = float(np.dot(base, self.reward_weights))
            self.last_reward = base

        return self.last_reward

    def _pressure_reward(self):
        return self.get_pressure()

    def _average_speed_reward(self):
        return self.get_average_speed()

    def _queue_reward(self):
        return -self.get_total_queued()

    def _co2_reward(self):
        return -self.get_total_co2()

    def _diff_waiting_time_reward(self):
        ts_wait = sum(self.get_accumulated_waiting_time_per_lane()) / 100.0
        reward = self.last_ts_waiting_time - ts_wait
        self.last_ts_waiting_time = ts_wait
        return reward

    def _pending_vehicle_penalty(self):
        pending = len(self.sumo.simulation.getPendingVehicles())
        return -self._pending_vehicle_penalty_weight * pending

    def _congestion_aware_reward(self):
        # Local waiting-time improvement plus penalties for local queue and global backlog.
        reward = self._diff_waiting_time_reward()
        reward -= self._queue_penalty_weight * self.get_total_queued()
        reward += self._standing_penalty()
        reward += self._starvation_penalty()
        reward += self._pending_vehicle_penalty()
        return reward

    def _standing_penalty(self):
        """Penalty proportional to the number of vehicles whose waiting time exceeds threshold."""
        import math

        if self._exp_wait_penalty_enabled:
            penalty = 0.0
            for lane in self.lanes:
                for veh in self.sumo.lane.getLastStepVehicleIDs(lane):
                    try:
                        wt = float(self.sumo.vehicle.getWaitingTime(veh))
                    except Exception:
                        wt = 0.0
                    over = max(0.0, wt - self._standing_wait_threshold)
                    if over > 0:
                        penalty += (
                            math.exp(self._exp_wait_penalty_scale * over) - 1.0
                        ) / 100.0
            return -self._standing_penalty_weight * penalty

        # Original count-based penalty
        count = 0
        for lane in self.lanes:
            for veh in self.sumo.lane.getLastStepVehicleIDs(lane):
                try:
                    wt = float(self.sumo.vehicle.getWaitingTime(veh))
                except Exception:
                    wt = 0.0
                if wt >= self._standing_wait_threshold:
                    count += 1
        return -self._standing_penalty_weight * count

    def _starvation_penalty(self):
        """Penalty for letting one incoming lane wait much longer than the others."""
        lane_waits = self.get_accumulated_waiting_time_per_lane()
        if not lane_waits:
            return 0.0

        worst_wait = max(lane_waits)
        if worst_wait < self._standing_wait_threshold:
            return 0.0

        return (
            -self._standing_penalty_weight
            * (worst_wait - self._standing_wait_threshold)
            / 100.0
        )

    def _observation_fn_default(self):
        phase_id = [
            1 if self.green_phase == i else 0 for i in range(self.num_green_phases)
        ]  # one-hot encoding
        min_green = [0 if self.time_since_last_phase_change < self.min_green else 1]
        density = self.get_lanes_density()
        queue = self.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def get_accumulated_waiting_time_per_lane(self) -> List[float]:
        """Returns the accumulated waiting time per lane.

        Returns:
            List[float]: List of accumulated waiting time of each intersection lane.
        """
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [
                            self.env.vehicles[veh][lane]
                            for lane in self.env.vehicles[veh].keys()
                            if lane != veh_lane
                        ]
                    )
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_average_speed(self) -> float:
        """Returns the average speed normalized by the maximum allowed speed of the vehicles in the intersection.

        Obs: If there are no vehicles in the intersection, it returns 1.0.
        """
        avg_speed = 0.0
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return 1.0
        for v in vehs:
            avg_speed += self.sumo.vehicle.getSpeed(
                v
            ) / self.sumo.vehicle.getAllowedSpeed(v)
        return avg_speed / len(vehs)

    def get_pressure(self):
        """Returns the pressure (#veh leaving - #veh approaching) of the intersection."""
        return sum(
            self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes
        ) - sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes)

    def get_out_lanes_density(self) -> List[float]:
        """Returns the density of the vehicles in the outgoing lanes of the intersection."""
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (
                self.lanes_length[lane]
                / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane))
            )
            for lane in self.out_lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_density(self) -> List[float]:
        """Returns the density [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        """
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (
                self.lanes_length[lane]
                / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane))
            )
            for lane in self.lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_queue(self) -> List[float]:
        """Returns the queue [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        """
        lanes_queue = [
            self.sumo.lane.getLastStepHaltingNumber(lane)
            / (
                self.lanes_length[lane]
                / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane))
            )
            for lane in self.lanes
        ]
        return [min(1, queue) for queue in lanes_queue]

    def get_total_queued(self) -> int:
        """Returns the total number of vehicles halting in the intersection."""
        return sum(self.sumo.lane.getLastStepHaltingNumber(lane) for lane in self.lanes)

    def get_total_co2(self) -> float:
        """Returns the total CO2 emissions (mg/s) of the vehicles in the incoming lanes of the intersection."""
        return sum(self.sumo.lane.getCO2Emission(lane) for lane in self.lanes)

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += self.sumo.lane.getLastStepVehicleIDs(lane)
        return veh_list

    @classmethod
    def register_reward_fn(cls, fn: Callable):
        """Registers a reward function.

        Args:
            fn (Callable): The reward function to register.
        """
        if fn.__name__ in cls.reward_fns.keys():
            raise KeyError(f"Reward function {fn.__name__} already exists")

        cls.reward_fns[fn.__name__] = fn

    reward_fns = {
        "diff-waiting-time": _diff_waiting_time_reward,
        "average-speed": _average_speed_reward,
        "queue": _queue_reward,
        "pressure": _pressure_reward,
        "co2": _co2_reward,
        "congestion-aware": _congestion_aware_reward,
    }

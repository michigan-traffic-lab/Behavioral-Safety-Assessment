from terasim.agent.agent_decision_model import AgentDecisionModel

import numpy as np
import math

from settings import *


def cal_dis(point1, point2):
    dis = 0
    for x, y in zip(point1, point2):
        dis += (x - y) ** 2
    return math.sqrt(dis)


class VRUDecisionModel(AgentDecisionModel):
    """dummy decision model:

    This decision model will constantly move the vehicle to the given x, y coordinates

    """

    def __init__(self):
        super().__init__()

    def init(self, id, init_conditions, settings):
        # relative_dis, speed, dis, start_point, angle, scenario
        self.scenario_list = settings.VRU_scenarios[id - 1]
        self.init_conditions = init_conditions
        self.start_points = {
            scenario: self._cal_start_point(
                settings.AV_route[
                    settings.scenario_info[scenario]["challenger"]["start AV ind"]
                ],
                self.init_conditions[scenario]["dis to edge"],
                settings.lane_width,
            )
            for scenario in self.scenario_list
        }
        self.angles = {
            scenario: settings.scenario_info[scenario]["challenger"]["angle"]
            for scenario in self.scenario_list
        }
        self.time_step = 0.1

        self.start_flags = {scenario: False for scenario in self.scenario_list}
        self.triggered_flags = {scenario: False for scenario in self.scenario_list}

        self.dis_err = {scenario: 0 for scenario in self.scenario_list}
        self.sp_err = {scenario: 0 for scenario in self.scenario_list}
        self.init_AV_sp = {scenario: 0 for scenario in self.scenario_list}
        self.init_timestamp = {scenario: 0 for scenario in self.scenario_list}
        self.AV_hist = {scenario: [] for scenario in self.scenario_list}

    def _cal_start_point(self, AV_point, dis2edge, lane_width):
        dis = lane_width / 2 + dis2edge
        AV_x, AV_y, AV_heading, _ = AV_point
        return [
            AV_x + dis * math.cos(AV_heading / 180 * math.pi),
            AV_y - dis * math.sin(AV_heading / 180 * math.pi),
            AV_heading,
        ]

    def update(self, AV_state, VRU_state):
        self.AV_state = AV_state
        self.VRU_state = VRU_state

    def derive_control_command_from_observation(self, obs_dict):
        """derive control command from observation

        Args:
            obs_dict (dict): vehicle observation dictionary

        Returns:
            dict: command
        """

        self._manage_state(self.AV_state, self.VRU_state)

        for scenario in self.scenario_list:
            command = {
                "type": "SetSumoTransform",
                "position": self.start_points[scenario][:2],
                "velocity": 0,
                "angle": self.start_points[scenario][2],  # degree
            }
            if self.AV_state.scenario == scenario:
                relative_dis = cal_dis(
                    [self.AV_state.x, self.AV_state.y],
                    [self.VRU_state.x, self.VRU_state.y],
                )
                s = cal_dis(
                    self.start_points[scenario][:2],
                    [self.VRU_state.x, self.VRU_state.y],
                )
                desired_position = self.start_points[scenario][:2]
                desired_speed = 0
                angle = self.start_points[scenario][2]
                if self.start_flags[scenario] and not self.triggered_flags[scenario]:
                    desired_speed = self.init_conditions[scenario]["alongside sp"]
                    angle = self.angles[scenario] + 90
                    desired_position[0] = (
                        self.VRU_state.x
                        + desired_speed
                        * math.cos(math.pi / 2 - angle / 180 * math.pi)
                        * self.time_step
                    )
                    desired_position[1] = (
                        self.VRU_state.y
                        + desired_speed
                        * math.sin(math.pi / 2 - angle / 180 * math.pi)
                        * self.time_step
                    )
                elif self.start_flags[scenario] and self.triggered_flags[scenario]:
                    desired_speed = self.init_conditions[scenario]["crossing sp"]
                    angle = self.angles[scenario]
                    desired_position[0] = (
                        self.VRU_state.x
                        + desired_speed
                        * math.cos(math.pi / 2 - angle / 180 * math.pi)
                        * self.time_step
                    )
                    desired_position[1] = (
                        self.VRU_state.y
                        + desired_speed
                        * math.sin(math.pi / 2 - angle / 180 * math.pi)
                        * self.time_step
                    )
                command = {
                    "type": "SetSumoTransform",
                    "position": desired_position,
                    "velocity": desired_speed,
                    "angle": angle,  # degree
                }
        command["keepRoute"] = 2
        command["speedmode"] = 0
        return command, None

    def _manage_state(self, AV_state, VRU_state):
        for scenario in self.scenario_list:
            if AV_state.scenario == scenario and not self.start_flags[scenario]:
                relative_dis = cal_dis(
                    [AV_state.x, AV_state.y], [VRU_state.x, VRU_state.y]
                )
                if relative_dis < self.init_conditions[scenario]["triggered dis"] + 20:
                    self.start_flags[scenario] = True
            if (
                AV_state.scenario == scenario
                and self.start_flags[scenario]
                and not self.triggered_flags[scenario]
            ):
                relative_dis = cal_dis(
                    [AV_state.x, AV_state.y], [VRU_state.x, VRU_state.y]
                )
                if len(self.AV_hist[scenario]) > 10:
                    self.AV_hist[scenario].pop(0)
                self.AV_hist[scenario].append(
                    [AV_state.timestamp, AV_state.speed, relative_dis]
                )
                k, _ = np.polyfit(
                    [line[0] for line in self.AV_hist[scenario]],
                    [line[1] for line in self.AV_hist[scenario]],
                    1,
                )
                if relative_dis < self.init_conditions[scenario]["triggered dis"] or (
                    abs(AV_state.speed - VRU_state.speed) < 0.5 and abs(k) < 0.2
                ):
                    self.triggered_flags[scenario] = True
                    self.dis_err[scenario] = (
                        relative_dis - self.init_conditions[scenario]["triggered dis"]
                    )
                    self.sp_err[scenario] = (
                        VRU_state.speed - self.init_conditions[scenario]["crossing sp"]
                    )
                    self.init_AV_sp[scenario] = AV_state.speed
                    self.init_timestamp[scenario] = AV_state.timestamp

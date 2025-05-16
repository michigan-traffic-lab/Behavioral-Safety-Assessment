from terasim.agent.agent_decision_model import AgentDecisionModel

import copy
import math

from vehicle.decision_models.BVPlanner.FlexibleCPPlanner import *
from vehicle.decision_models.BVPlanner.NoCPPlanner import *
from vehicle.decision_models.BVPlanner.FixedCPPlanner import *
from vehicle.decision_models.BVPlanner.SingleAgentPlanner import *
from settings import *
from utils import *


class BVDecisionModel(AgentDecisionModel):
    """dummy decision model:

    This decision model will constantly move the vehicle to the given x, y coordinates

    """

    def __init__(self):
        super().__init__()

    def init(self, id, init_conditions, settings):
        self.time_step = 0.1
        self.scenario_list = settings.BV_scenarios[id - 1]
        self.route = copy.deepcopy(settings.BV_routes[id - 1])
        self.start_point = list(self.route[0, :2])
        self.angle = self.route[0, 2]
        self.state = None
        self.init_conditions = init_conditions
        self.scenario_category = settings.scenario_category

        self.planner = {}
        for scenario in list(init_conditions.keys()):
            if scenario in self.scenario_category["flexible conflict point"]:
                self.planner[scenario] = FlexibleConflictPointPlanner(
                    settings.AV_route,
                    self.route,
                    settings.scenario_info[scenario],
                    init_conditions[scenario],
                    settings.lane_width,
                )
            elif scenario in self.scenario_category["no conflict point"]:
                self.planner[scenario] = NoConflictPointPlanner(
                    self.route,
                    settings.scenario_info[scenario],
                    init_conditions[scenario],
                )
            elif scenario in self.scenario_category["fixed conflict point"]:
                self.planner[scenario] = FixedConflictPointPlanner(
                    settings.AV_route,
                    self.route,
                    settings.scenario_info[scenario],
                    init_conditions[scenario],
                )
            elif scenario in self.scenario_category["single agent"]:
                for scenario_list, BV_lenth in zip(
                    settings.BV_scenarios, settings.BV_length
                ):
                    if scenario in scenario_list:
                        self.planner[scenario] = SingleAgentPlanner(
                            settings.AV_route,
                            settings.scenario_info[scenario],
                            init_conditions[scenario],
                            BV_lenth,
                        )

    def update(self, AV_state, BV_state):
        self.AV_state = AV_state
        self.BV_state = BV_state

    def derive_control_command_from_observation(self, obs_dict):
        """derive control command from observation

        Args:
            obs_dict (dict): vehicle observation dictionary

        Returns:
            dict: command
        """
        if self.BV_state.scenario is None:
            command = self._plain_planning()
        elif self.BV_state.scenario == self.AV_state.scenario:
            if self.BV_state.scenario in self.scenario_category['fixed conflict point']:
                command = self.planner[self.BV_state.scenario].planning(self.AV_state, self.BV_state)
            elif self.BV_state.scenario in self.scenario_category['flexible conflict point']:
                command = self.planner[self.BV_state.scenario].planning(self.AV_state, self.BV_state)
            elif self.BV_state.scenario in self.scenario_category['no conflict point']:
                command = self.planner[self.BV_state.scenario].planning(self.AV_state, self.BV_state)
            elif self.BV_state.scenario in self.scenario_category['single agent']:
                command = self.planner[self.BV_state.scenario].planning(self.AV_state, self.BV_state)
        else:
            command = {
                "type": "SetSumoTransform",
                "position": (self.BV_state.x, self.BV_state.y),
                "velocity": 0,
                "angle": self.BV_state.heading,
            }
        command["keepRoute"] = 2
        command["speedmode"] = 0

        return command, None

    def _plain_planning(self):
        sp = 5
        NPN = self.BV_state.NPN
        if NPN < len(self.route) - 1:
            distanceToLaneCenter = distancPoint2Line(
                self.route[NPN][0],
                self.route[NPN][1],
                self.route[NPN + 1][0],
                self.route[NPN + 1][1],
                self.BV_state.x,
                self.BV_state.y,
            )
            heading = self.route[NPN][2] / 180 * math.pi
            x = (
                self.BV_state.x
                + sp * self.time_step * math.sin(heading)
                + distanceToLaneCenter * math.cos(heading)
            )
            y = (
                self.BV_state.y
                + sp * self.time_step * math.cos(heading)
                - distanceToLaneCenter * math.sin(heading)
            )
        else:
            x = self.BV_state.x
            y = self.BV_state.y
            sp = 0
        command = {
            "type": "SetSumoTransform",
            "position": (x, y),
            "velocity": sp,
            "angle": self.route[NPN][2],
        }
        return command

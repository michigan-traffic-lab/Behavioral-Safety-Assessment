import math

from utils import *


class PlainPlanner():
    def __init__(self, scenario_list, scenario_info, AV_route, BV_route):
        self.scenario_list = scenario_list
        self.scenario_info = scenario_info
        self.init_conditions = []
        self.AV_route = AV_route
        self.BV_route = BV_route
        self.BV_status = None
        self.NPN = 0

        self.dis_err = 0
        self.sp_err = 0
        self.init_AV_sp = 0
        self.init_timestamp = 0

    def init(self, init_conditions):
        self.init_conditions = init_conditions

    def planning(self, AV_state, BV_state, route, step_time):
        NPN = BV_state.NPN
        distanceToLaneCenter = distancPoint2Line(route[NPN][0], route[NPN][1], route[NPN+1][0], route[NPN+1][1], BV_state.x, BV_state.y)

        sp = 10
        heading = route[NPN][2]
        x = BV_state.x + sp * step_time * math.sin(heading) + distanceToLaneCenter * math.cos(heading)
        y = BV_state.y + sp * step_time * math.cos(heading) - distanceToLaneCenter * math.sin(heading)
        command = {
            "type": "SetSumoTransform",
            "position": (x, y),
            "velocity": sp,
            "angle": heading
        }
        return command

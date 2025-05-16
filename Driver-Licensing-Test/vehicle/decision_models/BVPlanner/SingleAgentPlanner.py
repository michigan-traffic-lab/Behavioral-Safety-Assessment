import numpy as np
import math

from utils import *
from settings import *


class SingleAgentPlanner():
    def __init__(self, AV_route, scenario_info, init_condition, BV_length):
        self.dis = init_condition['dis']
        self.angle = init_condition['angle']
        self.time_step = 0.1
        self.AV_hist = []
        self.BV_length = BV_length
        self.init_timestamp = 0
        self.sp_err = 0
        self.dis_err = 0
        self.init_AV_sp = 0

        self.start_flag = False
        self.complete_flag = False

    def planning(self, AV_state, BV_state):
        self._manage_state(AV_state, BV_state)
        self._set_desired_speed(AV_state, BV_state)
        self._set_desired_position(AV_state, BV_state)
        command = {
            "type": "SetSumoTransform",
            "position": self.desired_position,
            "velocity": self.desired_speed,
            "angle": self.angle
        }
        return command

    def _manage_state(self, AV_state, BV_state):
        dis = cal_dis([AV_state.x, AV_state.y], [BV_state.x, BV_state.y])
        if len(self.AV_hist) > 10:
            self.AV_hist.pop(0)
        self.AV_hist.append([AV_state.timestamp, AV_state.speed, dis])
        if len(self.AV_hist) == 11:
            k2, _ = np.polyfit([line[0] for line in self.AV_hist], [line[2] for line in self.AV_hist], 1)
            if abs(k2) < 0.1:
                self.complete_flag = True

    def _set_desired_speed(self, AV_state, BV_state):
        self.desired_speed = 0

    def _set_desired_position(self, AV_state, BV_state):
        if not self.start_flag:
            expected_x = BV_state.x + self.dis * math.cos(BV_state.heading / 180 * math.pi) + self.BV_length / 2 * math.sin(self.angle / 180 * math.pi)
            expected_y = BV_state.y - self.dis * math.sin(BV_state.heading / 180 * math.pi) - self.BV_length / 2 * math.cos(self.angle / 180 * math.pi)
            self.desired_position = [expected_x, expected_y]
            self.start_flag = True
            self.init_timestamp = BV_state.timestamp
            self.init_AV_sp = AV_state.speed

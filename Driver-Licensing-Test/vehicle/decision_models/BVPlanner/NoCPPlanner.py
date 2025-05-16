import numpy as np
import math

from utils import *
from settings import *


class NoConflictPointPlanner():
    def __init__(self, BV_route, scenario_info, init_condition):
        self.BV_route = BV_route
        self.BV_end_ind = scenario_info['challenger']['end ind']
        self.acc = init_condition['acc']
        self.dec = init_condition['dec']
        self.init_sp = init_condition['sp']
        self.time_step = 0.1
        self.AV_hist = []

        self.start_acc_flag = False
        self.start_dec_flag = False
        self.complete_flag = False

        self.dis_err = 0
        self.sp_err = 0
        self.init_AV_sp = 0
        self.init_timestamp = 0

    def planning(self, AV_state, BV_state):
        self._manage_state(AV_state, BV_state)
        self._set_desired_speed(AV_state, BV_state)
        self._set_desired_position(BV_state)
        command = {
            "type": "SetSumoTransform",
            "position": self.desired_position,
            "velocity": self.desired_speed,
            "angle": self.BV_route[BV_state.NPN][2]
        }
        return command

    def _manage_state(self, AV_state, BV_state):
        dis = cal_dis([AV_state.x, AV_state.y], [BV_state.x, BV_state.y])
        if len(self.AV_hist) > 10:
            self.AV_hist.pop(0)
        self.AV_hist.append([AV_state.timestamp, AV_state.speed, dis])
        if not self.start_acc_flag and not self.start_dec_flag and not self.complete_flag:
            if dis < 15 and AV_state.speed < 1:
                self.start_acc_flag = True
        elif self.start_acc_flag and not self.start_dec_flag and not self.complete_flag:
            if abs(self.init_sp - BV_state.speed) < 0.2 and AV_state.speed > self.init_sp / 2:
                k1, _ = np.polyfit([line[0] for line in self.AV_hist], [line[1] for line in self.AV_hist], 1)
                k2, _ = np.polyfit([line[0] for line in self.AV_hist], [line[2] for line in self.AV_hist], 1)
                if k1 < 0 and k2 > 0:
                    self.start_dec_flag = True
                if BV_state.NPN > self.BV_end_ind - 100:
                    self.start_dec_flag = True
        elif self.start_acc_flag and self.start_dec_flag and not self.complete_flag:
            if AV_state.speed < .1 and BV_state.speed < .1:
                self.complete_flag = True
    
    def _set_desired_speed(self, AV_state, BV_state):
        if not self.start_acc_flag and not self.start_dec_flag and not self.complete_flag:
            self.desired_speed = 0
        elif self.start_acc_flag and not self.start_dec_flag and not self.complete_flag:
            self.desired_speed = min(self.init_sp, BV_state.speed + self.time_step * self.acc)
        elif self.start_acc_flag and self.start_dec_flag and not self.complete_flag:
            self.desired_speed = max(BV_state.speed + self.time_step * self.dec, 0)
        elif self.start_acc_flag and self.start_dec_flag and self.complete_flag:
            self.desired_speed = 10
    
    def _set_desired_position(self, BV_state):
        NPN = BV_state.NPN
        if NPN < len(self.BV_route) - 1:
            distanceToLaneCenter = distancPoint2Line(self.BV_route[NPN][0], self.BV_route[NPN][1], self.BV_route[NPN+1][0], self.BV_route[NPN+1][1], BV_state.x, BV_state.y)
        else:
            distanceToLaneCenter = distancPoint2Line(self.BV_route[NPN][0], self.BV_route[NPN][1], self.BV_route[NPN-1][0], self.BV_route[NPN-1][1], BV_state.x, BV_state.y)
        heading = self.BV_route[NPN][2] / 180 * math.pi
        x = BV_state.x + self.desired_speed * self.time_step * math.sin(heading) + distanceToLaneCenter * math.cos(heading)
        y = BV_state.y + self.desired_speed * self.time_step * math.cos(heading) - distanceToLaneCenter * math.sin(heading)
        self.desired_position = [x, y]

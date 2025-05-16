from math import sqrt
from numpy import zeros, savez_compressed
from rich.progress import track
from termcolor import colored

from utils import *


class AnalyzeFeasibility():
    def __init__(self, scenario, name, settings):
        self.scenario = scenario
        self.name = name
        self.AV_length = settings.AV_length
        self.AV_width = settings.AV_width
        for scenarios, length, width in zip(settings.BV_scenarios, settings.BV_length, settings.BV_width):
            if self.scenario in scenarios:
                self.BV_length = length
                self.BV_width = width
        self.D_BV = settings.scenario_info[self.scenario]['stop line to conflict point distance']
        self.T_REACT_acc = settings.scenario_info[self.scenario]['acceleration reaction time']
        self.T_REACT_dec = settings.scenario_info[self.scenario]['deceleration reaction time']

        self.d_AV_min = settings.scenario_info[self.scenario]['min initial distance']
        self.d_AV_max = settings.scenario_info[self.scenario]['max initial distance']
        self.v_AV_min = settings.scenario_info[self.scenario]['min AV speed']
        self.v_AV_max = settings.scenario_info[self.scenario]['max AV speed']
        self.v_BV_min = settings.scenario_info[self.scenario]['min BV speed']
        self.v_BV_max = settings.scenario_info[self.scenario]['max BV speed']
        self.amin = settings.scenario_info[self.scenario]['min acceleration']
        self.amax = settings.scenario_info[self.scenario]['max acceleration']

        self.dis_step = settings.scenario_info[self.scenario]['distance step']
        self.AV_sp_step = settings.scenario_info[self.scenario]['AV speed step']
        self.BV_sp_step = settings.scenario_info[self.scenario]['BV speed step']
        self.acc_step = settings.scenario_info[self.scenario]['acceleration step']

        self.dis_num = int((self.d_AV_max - self.d_AV_min) / self.dis_step) + 1
        self.AV_sp_num = int((self.v_AV_max - self.v_AV_min) / self.AV_sp_step) + 1
        self.BV_sp_num = int((self.v_BV_max - self.v_BV_min) / self.BV_sp_step) + 1
        self.acc_num = int((self.amax - self.amin) / self.acc_step) + 1

        self.record_dec = zeros([self.dis_num, self.AV_sp_num, self.BV_sp_num])
        self.record_acc = zeros([self.dis_num, self.AV_sp_num, self.BV_sp_num])


    def collision_judgement_dec(self, d_AV, v_AV, v_BV, a):
        d_AV2cols = d_AV - self.BV_length  # distance from AV start point to the conflict point at where AV RF corner reaches
        t_BV2cols = (self.D_BV - self.AV_length) / v_BV  # time for BV reach at the conflict area
        flag = False

        if v_AV**2 - 2*a*(d_AV2cols - v_AV*self.T_REACT_dec) < 0:
            # if AV does not reach the conflict point when it stops
            flag = False  # collision can be avoided
        else:
            if d_AV2cols > v_AV*self.T_REACT_dec:
                t_AV2cols = 2*(d_AV2cols - v_AV*self.T_REACT_dec) / (v_AV + sqrt(max(v_AV**2 - 2*a*(d_AV2cols - v_AV*self.T_REACT_dec), 0))) + self.T_REACT_dec
            else:
                t_AV2cols = d_AV2cols / v_AV
            if t_AV2cols < t_BV2cols:
                # if AV reaches the conflict point earlier than BV
                flag = True  # collision cannot be avoided
            else:
                if v_AV - a * (t_BV2cols - self.T_REACT_dec) < v_BV:
                    # if AV speed is lower than BV speed when BV reaches at conflict area
                    flag = False
                else:
                    if a == 0:
                        flag = True
                    else:
                        if v_AV * self.T_REACT_dec + v_AV * (v_AV - v_BV) / a - (v_AV - v_BV)**2 / 2 / a - d_AV2cols > v_BV * (self.T_REACT_dec + (v_AV - v_BV) / a) - self.D_BV - self.BV_length:
                            # if AV front bumper is ahead of BV rear bumper when AV speed is the same as BV
                            flag = True
                        else:
                            flag = False
        return flag

    def collision_judgement_acc(self, d_AV, v_AV, v_BV, a):
        d_AV2cols = d_AV - self.BV_length  # distance from AV start point to the conflict point at where AV RF corner reaches
        t_BV2cols = (self.D_BV - self.AV_length) / v_BV  # time for BV reach at the conflict area
        flag = False

        v_AV, s_AV = self.cal_AV_sp_dis(v_AV, a, t_BV2cols)
        if s_AV <= d_AV2cols + self.AV_length:
            # if BV rear bumper reaches at the conflict area first
            flag = True  # collision cannot be avoided
        else:
            flag = False
        return flag

    def cal_AV_sp_dis(self, v, a, t):
        if t < self.T_REACT_acc or a == 0:
            return v, v*t
        else:
            if v + a * (t - self.T_REACT_acc) < self.v_AV_max:
                return v + a * (t - self.T_REACT_acc), v * t + a / 2 * (t - self.T_REACT_acc) ** 2
            else:
                t_ = self.T_REACT_acc + (self.v_AV_max - v) / a
                return self.v_AV_max, v * t_ + a / 2 * t_ ** 2 + self.v_AV_max * (t - t_)

    def analyze_feasibility(self, v_AV_max=None, v_AV_min=None):
        if v_AV_max is not None:
            self.v_AV_max = v_AV_max
        if v_AV_min is not None:
            self.v_AV_min = v_AV_min
        ori_record_dec = zeros([self.dis_num, self.AV_sp_num, self.BV_sp_num, self.acc_num])
        ori_record_acc = zeros([self.dis_num, self.AV_sp_num, self.BV_sp_num, self.acc_num])
        for ind_d in track(range(self.dis_num), description='Analyzing acceleration requirement for each case...'):
            for ind_v1 in range(self.AV_sp_num):
                for ind_v2 in range(self.BV_sp_num):
                    dec = []
                    acc = []
                    for ind_a in range(self.acc_num):
                        flag = self.collision_judgement_dec(
                            self.d_AV_min + (self.d_AV_max - self.d_AV_min) / (self.dis_num - 1) * ind_d,
                            self.v_AV_min + (self.v_AV_max - self.v_AV_min) / (self.AV_sp_num - 1) * ind_v1,
                            self.v_BV_min + (self.v_BV_max - self.v_BV_min) / (self.BV_sp_num - 1) * ind_v2,
                            self.amin + (self.amax - self.amin) / (self.acc_num - 1) * ind_a)
                        ori_record_dec[ind_d, ind_v1, ind_v2, ind_a] = flag
                        if not flag:
                            dec.append(self.amin + (self.amax - self.amin) / (self.acc_num - 1) * ind_a)
                        flag = self.collision_judgement_acc(
                            self.d_AV_min + (self.d_AV_max - self.d_AV_min) / (self.dis_num - 1) * ind_d,
                            self.v_AV_min + (self.v_AV_max - self.v_AV_min) / (self.AV_sp_num - 1) * ind_v1,
                            self.v_BV_min + (self.v_BV_max - self.v_BV_min) / (self.BV_sp_num - 1) * ind_v2,
                            self.amin + (self.amax - self.amin) / (self.acc_num - 1) * ind_a)
                        ori_record_acc[ind_d, ind_v1, ind_v2, ind_a] = flag
                        if not flag:
                            acc.append(self.amin + (self.amax - self.amin) / (self.acc_num - 1) * ind_a)
                    if len(dec) == 0:
                        self.record_dec[ind_d, ind_v1, ind_v2] = 7.1
                    else:
                        self.record_dec[ind_d, ind_v1, ind_v2] = min(dec)
                    if len(acc) == 0:
                        self.record_acc[ind_d, ind_v1, ind_v2] = 7.1
                    else:
                        self.record_acc[ind_d, ind_v1, ind_v2] = min(acc)

    def save(self, feasibility_path):
        path = feasibility_path + '/' + self.name + '_dec.npz'
        savez_compressed(path, self.record_dec)
        path = feasibility_path + '/' + self.name + '_acc.npz'
        savez_compressed(path, self.record_acc)
        print(colored(f'[ {self.scenario} ] Analyzing results of right turn have been saved to ' + feasibility_path + '/' + self.name, 'green'))

from math import sqrt, acos, pi
from numpy import dot, arange, zeros, savez_compressed
from numpy.linalg import norm
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
            if scenario in scenarios:
                self.BV_length = length
                self.BV_width = width
        self.turning_radius = settings.scenario_info[scenario]['turning radius']
        self.lane_width = settings.lane_width

        self.turning_radius = settings.scenario_info[scenario]['turning radius']
        self.acc_react_time = settings.scenario_info[scenario]['acceleration reaction time']
        self.dec_react_time = settings.scenario_info[scenario]['deceleration reaction time']
        self.min_dis = settings.scenario_info[scenario]['min initial distance']
        self.max_dis = settings.scenario_info[scenario]['max initial distance']
        self.min_AV_sp = settings.scenario_info[scenario]['min AV speed']
        self.max_AV_sp = settings.scenario_info[scenario]['max AV speed']
        self.min_BV_sp = settings.scenario_info[scenario]['min BV speed']
        self.max_BV_sp = settings.scenario_info[scenario]['max BV speed']
        self.min_a = settings.scenario_info[scenario]['min acceleration']
        self.max_a = settings.scenario_info[scenario]['max acceleration']
        self.acc_bound = settings.scenario_info[scenario]['acceleration bound']
        self.dec_bound = settings.scenario_info[scenario]['deceleration bound']

        self.dis_step = settings.scenario_info[scenario]['distance step']
        self.AV_sp_step = settings.scenario_info[scenario]['AV speed step']
        self.BV_sp_step = settings.scenario_info[scenario]['BV speed step']
        self.acc_step = settings.scenario_info[scenario]['acceleration step']

        self.dis_num = int((self.max_dis - self.min_dis) / self.dis_step) + 1
        self.AV_sp_num = int((self.max_AV_sp - self.min_AV_sp) / self.AV_sp_step) + 1
        self.BV_sp_num = int((self.max_BV_sp - self.min_BV_sp) / self.BV_sp_step) + 1
        self.acc_num = int((self.max_a - self.min_a) / self.acc_step) + 1

        self.d_res = settings.scenario_info[scenario]['distance resolution']
        self.v_res = settings.scenario_info[scenario]['speed resolution']

        self.bounds = {
            'acc': settings.scenario_info[scenario]['acceleration bound'],
            'dec': settings.scenario_info[scenario]['deceleration bound']
        }

        self.v_AV = settings.AV_speed_list[settings.scenario_list.index(scenario)]

        self.D2 = self.BV_length / 2 + self.turning_radius
        self.R2 = sqrt((self.turning_radius + self.BV_width / 2) ** 2 + self.BV_length ** 2 / 4)

        self.x2_lr1 = self.turning_radius + self.BV_length / 2
        self.y2_lr1 = -self.BV_width / 2
        self.x2_lr2 = self.AV_width / 2
        self.y2_lr2 = self.turning_radius - sqrt((self.AV_width + self.BV_width) *self.turning_radius + (self.BV_width ** 2 + self.BV_length ** 2 - self.AV_width ** 2) / 4)

        self.vector1 = [self.x2_lr1 -self.turning_radius, self.y2_lr1 -self.turning_radius]
        self.vector2 = [self.x2_lr2 -self.turning_radius, self.y2_lr2 -self.turning_radius]

        self.angle = acos(dot(self.vector1, self.vector2) / (norm(self.vector1)*norm(self.vector2)))
        self.d2_cols = self.angle *self.turning_radius + self.D2 -self.turning_radius  # distance from BV start point to the conflict point at where BV LF corner reaches
        self.record_dec = zeros([self.dis_num, self.AV_sp_num, self.BV_sp_num])
        self.record_acc = zeros([self.dis_num, self.AV_sp_num, self.BV_sp_num])


    def collision_judgement_dec(self, d1, v1, v2, a):
        d1_cols = self.y2_lr2 + d1 - self.AV_length / 2  # distance from AV start point to the conflict point at where AV RF corner reaches
        t1 = (self.D2 - self.turning_radius) / v2  # time for BV to start turning
        t2 = t1 + pi * self.turning_radius / 2 / v2  # time for BV to stop turning and enter the straight line
        t2_cols = self.d2_cols / v2  # time for BV LF corner to reach at the conflict point
        flag = False

        if v1**2 - 2*a*(d1_cols - v1*self.dec_react_time) < 0:
            # if AV right-front corner does not reach the conflict point when it stops
            flag = False  # collision can be avoided
        else:
            if d1_cols > v1*self.dec_react_time:
                t1_cols = 2*(d1_cols - v1*self.dec_react_time) / (v1 + sqrt(v1**2 - 2*a*(d1_cols - v1*self.dec_react_time))) + self.dec_react_time
            else:
                t1_cols = d1_cols / v1
            if t1_cols < t2_cols:
                # if AV reaches the conflict point earlier than BV
                flag = True  # collision cannot be avoided
            else:
                for t in arange(max(t1_cols, 0), t2, 0.01):
                    if t <= self.dec_react_time:
                        y1 = v1*t - d1 + self.AV_length/2
                    else:
                        y1 = v1*t - a*(t-self.dec_react_time)**2/2 - d1 + self.AV_length/2
                    if t <= t1:
                        x2 = self.D2 + self.BV_length / 2 - v2 * t
                        y2 = -self.BV_width / 2
                    else:
                        x2, y2 = rotation(self.turning_radius+self.BV_length/2, -self.BV_width/2, self.turning_radius, self.turning_radius, v2/self.turning_radius*(t-t1))
                    if y1 > y2:
                        # if AV RF corner is ahead of BV LR corner
                        flag = True
                        break
                    if v1 < a * (t-self.dec_react_time):
                        # if AV speed is 0
                        break
                if not flag and v1 >= a * (t2-self.dec_react_time):
                    # if no collision in conflict area and AV does not stop when BV passes the conflict area
                    if v1 - a * (t2-self.dec_react_time) <= v2:
                        # if AV speed is lower then BV when BV in the straight lane
                        flag = False
                    else:
                        if (-d1 + self.AV_length/2 + v1*self.dec_react_time)*a + v1*(v1-v2) - (v1-v2)**2/2 >= (self.turning_radius - self.BV_length/2)*a + (v1-v2-t2*a)*v2:
                            # if AV RF corner is ahead of BV LR corner when AV speed is the same as BV
                            flag = True
                        else:
                            flag = False
        return flag

    def collision_judgement_acc(self, d1, v1, v2, a):
        d1_cols = self.y2_lr2 + d1 + self.AV_length / 2  # distance from AV start point to the conflict point at where AV RR corner reaches
        t1 = (self.D2 - self.turning_radius) / v2  # time for BV to start turning
        t2 = t1 + pi * self.turning_radius / 2 / v2  # time for BV to stop turning and enter the straight line
        t2_cols = self.d2_cols / v2  # time for BV LF corner to reach at the conflict point
        flag = False

        v_AV, s_AV = self.cal_AV_sp_dis(v1, a, t2_cols)
        if s_AV <= d1_cols:
            # if BV FL corner reaches at the conflict point first
            flag = True  # collision cannot be avoided
        else:
            flag = False
        return flag

    def cal_AV_sp_dis(self, v, a, t):
        if t < self.acc_react_time or a == 0:
            return v, v*t
        else:
            if v + a * (t - self.acc_react_time) < self.max_AV_sp:
                return v + a * (t - self.acc_react_time), v * t + a / 2 * (t - self.acc_react_time) ** 2
            else:
                t_ = self.acc_react_time + (self.max_AV_sp - v) / a
                return self.max_AV_sp, v * t_ + a / 2 * t_ ** 2 + self.max_AV_sp * (t - t_)
    
    def cal_BV_pos(self, v, t, t1, t2):
        if t <= t1:
            x2 = self.D2 + self.BV_length / 2 - v * t
            y2 = -self.BV_width / 2
        elif t1 < t < t2:
            x2, y2 = rotation(self.turning_radius+self.BV_length/2, -self.BV_width/2, self.turning_radius, self.turning_radius, v/self.turning_radius*(t-t1))
        else:
            x2 = 0
            y2 = self.turning_radius + v * (t - t2)
        return x2, y2

    def analyze_feasibility(self, v1max=None, v1min=None):
        if v1max is not None:
            self.max_AV_sp = v1max
        if v1min is not None:
            self.min_AV_sp = v1min
        ori_record_dec = zeros([self.dis_num, self.AV_sp_num, self.BV_sp_num, self.acc_num])
        ori_record_acc = zeros([self.dis_num, self.AV_sp_num, self.BV_sp_num, self.acc_num])
        for ind_d in track(range(self.dis_num), description='Analyzing acceleration requirement for each case...'):
            for ind_v1 in range(self.AV_sp_num):
                for ind_v2 in range(self.BV_sp_num):
                    dec = []
                    acc = []
                    for ind_a in range(self.acc_num):
                        flag = self.collision_judgement_dec(
                            self.min_dis + (self.max_dis - self.min_dis) / (self.dis_num - 1) * ind_d,
                            self.min_AV_sp + (self.max_AV_sp - self.min_AV_sp) / (self.AV_sp_num - 1) * ind_v1,
                            self.min_BV_sp + (self.max_BV_sp - self.min_BV_sp) / (self.BV_sp_num - 1) * ind_v2,
                            self.min_a + (self.max_a - self.min_a) / (self.acc_num - 1) * ind_a)
                        ori_record_dec[ind_d, ind_v1, ind_v2, ind_a] = flag
                        if not flag:
                            dec.append(self.min_a + (self.max_a - self.min_a) / (self.acc_num - 1) * ind_a)
                        flag = self.collision_judgement_acc(
                            self.min_dis + (self.max_dis - self.min_dis) / (self.dis_num - 1) * ind_d,
                            self.min_AV_sp + (self.max_AV_sp - self.min_AV_sp) / (self.AV_sp_num - 1) * ind_v1,
                            self.min_BV_sp + (self.max_BV_sp - self.min_BV_sp) / (self.BV_sp_num - 1) * ind_v2,
                            self.min_a + (self.max_a - self.min_a) / (self.acc_num - 1) * ind_a)
                        ori_record_acc[ind_d, ind_v1, ind_v2, ind_a] = flag
                        if not flag:
                            acc.append(self.min_a + (self.max_a - self.min_a) / (self.acc_num - 1) * ind_a)
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
        print(colored(f'[ {self.scenario} ] Analyzing results of right turn have been saved to ' + path, 'green'))

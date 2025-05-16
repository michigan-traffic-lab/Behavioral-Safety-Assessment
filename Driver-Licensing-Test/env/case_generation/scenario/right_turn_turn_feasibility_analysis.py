from math import sqrt, acos, pi, atan
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
        self.min_dis = self.AV_length / 2
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
        
        self.risk_levels = settings.risk_levels
        self.d_res = settings.scenario_info[scenario]['distance resolution']
        self.v_res = settings.scenario_info[scenario]['speed resolution']

        self.bounds = {
            'acc': settings.scenario_info[scenario]['acceleration bound'],
            'dec': settings.scenario_info[scenario]['deceleration bound']
        }

        self.v_AV = settings.AV_speed_list[settings.scenario_list.index(scenario)]

        self.D1 = settings.scenario_info[scenario]['straight distance'] + self.BV_length / 2
        self.D2 = self.turning_radius + self.AV_length / 2
        self.R2 = sqrt((self.turning_radius + self.BV_width / 2) ** 2 + self.BV_length ** 2 / 4)

        self.x2_lr1 = self.turning_radius + self.AV_length / 2
        self.y2_lr1 = -self.AV_width / 2
        self.x2_lr2 = self.BV_width / 2
        self.y2_lr2 = self.turning_radius - sqrt((self.BV_width + self.AV_width) * self.turning_radius + (self.AV_width ** 2 + self.AV_length ** 2 - self.BV_width ** 2) / 4)

        self.vector1 = [self.x2_lr1 - self.turning_radius, self.y2_lr1 - self.turning_radius]
        self.vector2 = [self.x2_lr2 - self.turning_radius, self.y2_lr2 - self.turning_radius]

        self.angle = acos(dot(self.vector1, self.vector2) / (norm(self.vector1)*norm(self.vector2)))
        
        self.d2_cols = self.D1 + self.y2_lr2 - self.BV_length / 2  # distance from BV start point to the conflict point at where BV RF corner reaches
        self.record_dec = zeros([self.dis_num, self.AV_sp_num, self.BV_sp_num])
        self.record_acc = zeros([self.dis_num, self.AV_sp_num, self.BV_sp_num])


    def collision_judgement_dec(self, d2, v1, v2, a):
        d1_cols = self.angle * self.turning_radius + d2 - self.turning_radius  # distance from AV start point to the conflict point at where AV LF corner reaches

        t2_cols = self.d2_cols / v2  # time for BV RF corner to reach at the conflict point
        flag = False

        if v1*self.dec_react_time * a + v1**2/2 < d1_cols * a:
            # if AV left-front corner does not reach the conflict point when it stops
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
                if a > 0:
                    if d2 - self.turning_radius - v1 * self.dec_react_time > 0:
                        t1 = (v1 - sqrt(v1 ** 2 * a * (d2 - self.turning_radius - v1 * self.dec_react_time))) / a + self.dec_react_time  # time for AV to start turning
                    else:
                        t1 = (d2 - self.turning_radius) / v1
                    t2 = (v1 - sqrt(v1 ** 2 * a * (d2 - self.turning_radius + pi * self.turning_radius / 2 - v1 * self.dec_react_time))) / a + self.dec_react_time  # time for AV to stop turning and enter the turn line
                else:
                    t1 = (d2 - self.turning_radius) / v1
                    t2 = (d2 - self.turning_radius + pi * self.turning_radius / 2) / v1
                for t in arange(max(t1_cols, 0), t2, 0.01):
                    y2 = v2 * t - self.D1 - self.BV_length / 2  # y coordinate of right-rear corner of BV
                    if t <= t1:
                        y1 = -self.AV_width / 2  # y coordinate of left-front corner of AV
                    else:
                        x1, y1 = rotation(self.turning_radius - self.BV_length/2, -self.BV_width/2, self.turning_radius, self.turning_radius, (v1 * t - a / 2 * (t - self.dec_react_time) ** 2 - self.D2 + self.turning_radius) / self.turning_radius)
                    if y1 > y2:
                        # if AV LF corner is ahead of BV RR corner
                        flag = True
                        break
                    if v1 < a * (t-self.dec_react_time):
                        # if AV speed is 0
                        break
                if not flag and v1 >= a * (t2-self.dec_react_time):
                    # if no collision in conflict area and AV does not stop when AV passes the conflict area
                    if v1 - a * (t2-self.dec_react_time) <= v2:
                        # if AV speed is lower then BV when BV in the turn lane
                        flag = False
                    else:
                        if a > 0:
                            t_same_sp = (v1 - v2) / a + self.dec_react_time  # time for AV to slow down to BV speed
                            if v1 * t_same_sp - a / 2 * (t_same_sp - self.dec_react_time) ** 2 - (self.D2 - self.turning_radius) - pi * self.turning_radius / 2 + self.turning_radius >= v2 * t_same_sp - self.D1:
                                # if AV RF corner is ahead of BV LR corner when AV speed is the same as BV
                                flag = True
                            else:
                                flag = False
                        else:
                            flag = True
        return flag

    def collision_judgement_acc(self, d2, v1, v2, a):
        d1_cols = self.angle * self.turning_radius + d2 - self.turning_radius + atan(self.AV_length / (2 * self.turning_radius + self.AV_width)) * 2 * self.turning_radius  # distance from AV start point to the conflict point at where AV LR corner reaches
        t2_cols = self.d2_cols / v2  # time for BV RF corner to reach at the conflict point
        flag = False

        BV_react_time = 0  # self.dec_react_time

        v_AV, s_AV = self.cal_AV_sp_dis(v1, a, t2_cols)
        if s_AV <= d1_cols:
            # if BV FL corner reaches at the conflict point first
            flag = True  # collision cannot be avoided
        else:
            dis_BV_1 = v2 * BV_react_time  # distance for BV to react
            dis_AV_1 = v_AV * BV_react_time + a / 2 * BV_react_time ** 2  # distance for AV
            if dis_AV_1 + s_AV - d1_cols < dis_BV_1:
                flag = True
            else:
                v_AV_1 = v_AV + a * BV_react_time
                if v2 < v_AV_1:
                    flag = False
                else:
                    t_BV_slow_down = (v2 - v_AV_1) / (a + self.bounds['dec'][-2])
                    dis_BV_2 = dis_BV_1 + v2 * t_BV_slow_down - self.bounds['dec'][-2] / 2 * t_BV_slow_down ** 2
                    dis_AV_2 = dis_AV_1 + v_AV_1 * t_BV_slow_down + a / 2 * t_BV_slow_down ** 2
                    if dis_AV_2 + s_AV - d1_cols < dis_BV_2:
                        flag = True
                    else:
                        flag = False
        return flag

    def cal_AV_sp_dis(self, v, a, t):
        if t < self.acc_react_time:
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

    def analyze_feasibility(self, max_AV_sp=None, min_AV_sp=None):
        if max_AV_sp is not None:
            self.max_AV_sp = max_AV_sp
        if min_AV_sp is not None:
            self.min_AV_sp = min_AV_sp
        ori_record_dec = zeros([self.dis_num, self.AV_sp_num, self.BV_sp_num, self.acc_num])
        ori_record_acc = zeros([self.dis_num, self.AV_sp_num, self.BV_sp_num, self.acc_num])
        for ind_d in track(range(self.dis_num), description='Analyzing acceleration requirement for each case...'):
            for ind_v1 in range(self.AV_sp_num):
                for ind_v2 in range(self.BV_sp_num):
                    dec = []
                    acc = []
                    for ind_a in range(self.acc_num):
                        flag = self.collision_judgement_dec(
                            self.min_dis + (self.max_dis - self.min_dis) / (self.dis_num - 1) * (ind_d - 1),
                            self.min_AV_sp + (self.max_AV_sp - self.min_AV_sp) / (self.AV_sp_num - 1) * (ind_v1 - 1),
                            self.min_BV_sp + (self.max_BV_sp - self.min_BV_sp) / (self.BV_sp_num - 1) * (ind_v2 - 1),
                            self.min_a + (self.max_a - self.min_a) / (self.acc_num - 1) * (ind_a - 1))
                        ori_record_dec[ind_d, ind_v1, ind_v2, ind_a] = flag
                        if not flag:
                            dec.append(self.min_a + (self.max_a - self.min_a) / (self.acc_num - 1) * (ind_a - 1))
                        flag = self.collision_judgement_acc(
                            self.min_dis + (self.max_dis - self.min_dis) / (self.dis_num - 1) * (ind_d - 1),
                            self.min_AV_sp + (self.max_AV_sp - self.min_AV_sp) / (self.AV_sp_num - 1) * (ind_v1 - 1),
                            self.min_BV_sp + (self.max_BV_sp - self.min_BV_sp) / (self.BV_sp_num - 1) * (ind_v2 - 1),
                            self.min_a + (self.max_a - self.min_a) / (self.acc_num - 1) * (ind_a - 1))
                        ori_record_acc[ind_d, ind_v1, ind_v2, ind_a] = flag
                        if not flag:
                            acc.append(self.min_a + (self.max_a - self.min_a) / (self.acc_num - 1) * (ind_a - 1))
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

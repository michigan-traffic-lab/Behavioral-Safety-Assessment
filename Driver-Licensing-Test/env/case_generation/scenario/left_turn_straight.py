import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
import json
import os
import math

from env.case_generation.covering_suite import *
from utils import *


class LeftTurnStraightGen:
    def __init__(self, settings):
        self.scenario = 'Left Turn (AV goes straight)'
        self.name = get_scenario_folder_name(self.scenario)
        self.feasibility_path = settings.feasibility_path + '/' + self.name
        if not os.path.exists(self.feasibility_path):
            os.mkdir(self.feasibility_path)
        self.risk_level_path = settings.risk_level_path + '/' + self.name
        if not os.path.exists(self.risk_level_path):
            os.mkdir(self.risk_level_path)
        self.case_path = settings.test_case_path + '/' + self.name
        if not os.path.exists(self.case_path):
            os.mkdir(self.case_path)
        self.risk_level_bounds = {
            'dec': [],
            'acc': []
        }
        self.risk_levels = settings.risk_levels
        self.risk_levels.reverse()
        self.cases = {risk_level: [] for risk_level in self.risk_levels[1:4]}
        self.case_num = 0

        self.AV_length = settings.AV_length
        self.AV_width = settings.AV_width
        for scenarios, length, width in zip(settings.BV_scenarios, settings.BV_length, settings.BV_width):
            if self.scenario in scenarios:
                self.BV_length = length
                self.BV_width = width
        self.turning_radius = settings.scenario_info[self.scenario]['turning radius']
        # size of Mcity LT intersection: params can be changed based on the dimension of the target intersection
        self.lane_width = settings.lane_width

        self.amax = settings.scenario_info[self.scenario]['max acceleration']
        self.amin = settings.scenario_info[self.scenario]['min acceleration']
        self.t_reac_acc = settings.scenario_info[self.scenario]['acceleration reaction time']
        self.t_reac_dec = settings.scenario_info[self.scenario]['deceleration reaction time']

        alpha = math.acos((self.turning_radius - self.lane_width + self.AV_width / 2) / (self.turning_radius - self.BV_width / 2))
        beta = math.atan((self.BV_length / 2) / (self.turning_radius - self.BV_width / 2))
        self.arc_len_1 = (alpha - beta) * (self.turning_radius - self.BV_width / 2)  # distance for BV enter conflict area
        self.d1 = math.sin(alpha) * (self.turning_radius - self.BV_width / 2)  # dist between BV starting center pt and the upper bound of conflict zone
        gamma = math.acos((self.turning_radius - self.lane_width - self.AV_width / 2) / (self.turning_radius + self.BV_width / 2))
        theta = math.atan((self.BV_length / 2) / (self.turning_radius + self.BV_width / 2))
        self.arc_len_2 = (gamma + theta) * (self.turning_radius + self.BV_width / 2)  # distance for BV leave conflict area
        self.d2 = math.sin(gamma) * (self.turning_radius + self.BV_width / 2) - self.d1  # dist between lower & upper bound of conflict zone
        self.d1 += settings.scenario_info[self.scenario]['turning offset']
        self.turning_offset = settings.scenario_info[self.scenario]['turning offset']

        self.v_AV_max = settings.scenario_info[self.scenario]['max AV speed']
        self.v_AV_min = settings.scenario_info[self.scenario]['min AV speed']
        self.v_BV_max = settings.scenario_info[self.scenario]['max BV speed']
        self.v_BV_min = settings.scenario_info[self.scenario]['min BV speed']
        self.dx_max = settings.scenario_info[self.scenario]['max initial distance']
        self.dx_min = self.d1 + self.d2
        self.step = 0.1
        self.v_res = settings.scenario_info[self.scenario]['speed resolution']
        self.d_res = settings.scenario_info[self.scenario]['distance resolution']

        self.v_AV = settings.AV_speed_list[settings.scenario_list.index(self.scenario)]

        self.unexecuted_risk_levels = self.risk_levels[1:4]
        self.executed_cases = {risk_level: [] for risk_level in self.risk_levels[1:4]}
        self.unexecuted_cases = {risk_level: [] for risk_level in self.risk_levels[1:4]}
        self.merged_risk_level_bounds = []

        self.colors = settings.colors
        self.alphas = settings.alphas
        self.fontsize = settings.fontsize

        self.covering_suite = CoveringSuite()

    def init(self):
        feasibility_data_file_acc = self.feasibility_path + '/' + self.name + '_acc.npz'
        feasibility_data_file_dec = self.feasibility_path + '/' + self.name + '_dec.npz'
        flag = False
        if not os.path.exists(feasibility_data_file_acc) or not os.path.exists(feasibility_data_file_dec):
            print(colored(f'[ {self.scenario} ] There is no feasibility data!', 'red'))
            self.risk_level_gen()
            flag = True

        cases_data_file = self.case_path + '/' + self.name + '.json'
        if not os.path.exists(cases_data_file):
            print(colored(f'[ {self.scenario} ] There are no cases that have been generated! Expected file path is ' + cases_data_file + ' .', 'red'))
            flag = True
        else:
            openfile = open(cases_data_file, 'r')
            self.cases = json.load(openfile)
            if self.v_AV != self.cases['v_AV']:
                print(colored(f'[ {self.scenario} ] Case speed is wrong! Expected ' + str(self.v_AV) + ' but now is ' + str(self.cases['v_AV']) + ' exist.', 'red'))
                flag = True

        if flag:
            self.generate_risk_level_bounds(self.v_AV)
            self.merge_risk_level_bounds()
            self.generate_merged_cases()
            self.save_cases()
            self.plot_cases()
            self.plot_merged_cases()

    def risk_level_gen(self):
        amin = self.amin
        amax = self.amax
        t_reac_dec = self.t_reac_dec
        t_reac_acc = self.t_reac_acc
        v_AV_pool=np.arange(self.v_AV_min, self.v_AV_max + self.step, self.step)
        v_BV_pool=np.arange(self.v_BV_min, self.v_BV_max + self.step, self.step)
        self.v_AV_pool = v_AV_pool
        self.v_BV_pool = v_BV_pool
        L_min_dec = np.zeros((len(amin),len(v_BV_pool),len(v_AV_pool)))
        L_max_acc = np.zeros((len(amax),len(v_BV_pool),len(v_AV_pool)))

        # Assume the AV notice when BV starts to turn
        # distance is from bumper to bumper
        for i in range(len(amin)):
            for j in range(len(v_BV_pool)):
                t1 = (self.turning_offset + self.arc_len_1 + self.BV_length / 2)/v_BV_pool[j] # moment when BV enters the conflict zone
                t2 = (self.turning_offset + self.arc_len_2 + self.BV_length / 2)/v_BV_pool[j] # moment when BV exits the conflict zone
                for k in range(len(v_AV_pool)):
                    if v_AV_pool[k] < -amin[i]*(t2-t_reac_dec[i]):  # AV speed is lower, AV can stop before entering conflict zone
                        L_min_dec[i,j,k] = v_AV_pool[k]*t_reac_dec[i] - (v_AV_pool[k]**2)*0.5/amin[i] + self.d2 + self.d1
                    else:  # AV speed is higher, cannot stopped when enter the conflict zone
                        L_min_dec[i,j,k] = v_AV_pool[k]*t2 + 0.5*amin[i]*(t2-t_reac_dec[i])**2 + self.d2 + self.d1
                    if v_AV_pool[k] + amax[i]*max(t1-t_reac_acc[i], 0) <= self.v_AV_max:
                        L_max_acc[i,j,k] = v_AV_pool[k]*t1 + 0.5*amax[i]*max(t1-t_reac_acc[i], 0)**2 + self.d1 - self.BV_length
                    else:
                        t = (self.v_AV_max - v_AV_pool[k]) / amax[i]
                        L_max_acc[i,j,k] = v_AV_pool[k]*(t + self.t_reac_acc[i]) + 0.5*amax[i]*t**2 + (t1 - t - t_reac_acc[i]) * self.v_AV_max + self.d1 - self.BV_length

        L_min_dec[L_min_dec>self.dx_max] = self.dx_max
        L_min_dec[L_min_dec<self.dx_min] = self.dx_min
        L_max_acc[L_max_acc>self.dx_max] = self.dx_max
        L_max_acc[L_max_acc<self.dx_min] = self.dx_min
        self.L_min_dec = L_min_dec
        self.L_max_acc = L_max_acc

        path = self.feasibility_path + '/' + self.name + '_dec.npz'
        np.savez_compressed(path, self.L_min_dec)
        path = self.feasibility_path + '/' + self.name + '_acc.npz'
        np.savez_compressed(path, self.L_max_acc)
        print(colored(f'[ {self.scenario} ] Analyzing results have been saved to ' + path, 'green'))

    def generate_risk_level_bounds(self, v_AV):
        try:
            self.L_min_dec = np.load(self.feasibility_path + '/' + self.name + '_dec.npz')
            self.L_min_dec = self.L_min_dec['arr_0']
            self.L_max_acc = np.load(self.feasibility_path + '/' + self.name + '_acc.npz')
            self.L_max_acc = self.L_max_acc['arr_0']
        except Exception as e:
            print(colored(f'[ {self.scenario} ] Fail to load risk level bounds!', 'red'))
            self.risk_level_gen()

        print(f'[ {self.scenario} ] Generating risk level...')
        ind = (v_AV - self.v_AV_min) / self.step
        if abs(round(ind) - ind) < 0.01:
            ind = round(ind)
            self.risk_level_bounds['dec'] = self.L_min_dec[:, :, ind]
            self.risk_level_bounds['acc'] = self.L_max_acc[:, :, ind]
        else:
            ind1 = int(ind)
            ind2 = ind1 + 1
            v1_1 = self.v_AV_min + self.step * ind1
            v1_2 = v1_1 + self.step
            self.risk_level_bounds['dec'] = ratio(v_AV, [v1_1, v1_2], [self.L_min_dec[:, :, ind1], self.L_min_dec[:, :, ind2]])
            self.risk_level_bounds['acc'] = ratio(v_AV, [v1_1, v1_2], [self.L_max_acc[:, :, ind1], self.L_max_acc[:, :, ind2]])
        self.risk_level_bounds['dec'] = np.concatenate(
            (
                np.asarray([[self.dx_min for _ in range(len(self.risk_level_bounds['dec'][0]))]]),
                self.risk_level_bounds['dec'],
                np.asarray([[self.dx_max for _ in range(len(self.risk_level_bounds['dec'][0]))]])
            ),
            axis=0
        )
        self.risk_level_bounds['dec'] = self.risk_level_bounds['dec'][::-1]
        self.risk_level_bounds['acc'] = np.concatenate(
            (
                np.asarray([[self.dx_max for _ in range(len(self.risk_level_bounds['acc'][0]))]]),
                self.risk_level_bounds['acc'],
                np.asarray([[self.dx_min for _ in range(len(self.risk_level_bounds['acc'][0]))]])
            ),
            axis=0
        )
        self.risk_level_bounds['acc'] = self.risk_level_bounds['acc'][::-1]
        print(colored(f'[ {self.scenario} ] Risk levels have been generated!', 'green'))

    def merge_risk_level_bounds(self):
        for dec_ind in range(len(self.risk_level_bounds['dec']) - 1):
            same_risk_level_bounds = self.merge_same_risk_level(dec_ind, dec_ind)
            deleted_risk_level_bounds = self.del_risk_level(same_risk_level_bounds, dec_ind)
            for bound in deleted_risk_level_bounds:
                self.merged_risk_level_bounds.append(bound)
        data = json.dumps(self.merged_risk_level_bounds)
        path = self.risk_level_path + '/' + self.name + '.json'
        f = open(path, 'w')
        f.write(data)
        f.close()
    
    def del_risk_level(self, risk_level_bounds, dec_ind):
        cand_bounds = []
        for bound in risk_level_bounds:
            cand_bound = bound
            for merged_risk_level_bound in self.merged_risk_level_bounds:
                if len(cand_bound['v_BV']) > 1:
                    bound1 = {
                        'risk_level': self.risk_levels[dec_ind],
                        'upper': [],
                        'lower': [],
                        'v_BV': []
                    }
                    bound2 = {
                        'risk_level': self.risk_levels[dec_ind],
                        'upper': [],
                        'lower': [],
                        'v_BV': []
                    }
                    cand_v_min = cand_bound['v_BV'][0]
                    cand_v_max = cand_bound['v_BV'][-1]
                    merged_v_min = merged_risk_level_bound['v_BV'][0]
                    merged_v_max = merged_risk_level_bound['v_BV'][-1]
                    if not (cand_v_min >= merged_v_max or cand_v_max <= merged_v_min):
                        start_v = max(cand_v_min, merged_v_min)
                        end_v = min(cand_v_max, merged_v_max)
                        cand_start_ind = int((start_v - cand_v_min) / (cand_v_max - cand_v_min) * len(cand_bound['v_BV']))
                        cand_end_ind = int((end_v - cand_v_min) / (cand_v_max - cand_v_min) * len(cand_bound['v_BV']))
                        merged_start_ind = int((start_v - merged_v_min) / (merged_v_max - merged_v_min) * len(merged_risk_level_bound['v_BV']))
                        merged_end_ind = int((end_v - merged_v_min) / (merged_v_max - merged_v_min) * len(merged_risk_level_bound['v_BV']))
                        ind = cand_start_ind
                        for upper_bound_dec, lower_bound_dec, upper_bound_acc, lower_bound_acc in \
                            zip(cand_bound['upper'][cand_start_ind:cand_end_ind+1], cand_bound['lower'][cand_start_ind:cand_end_ind+1],
                            merged_risk_level_bound['upper'][merged_start_ind:merged_end_ind+1], merged_risk_level_bound['lower'][merged_start_ind:merged_end_ind+1]):
                            if lower_bound_dec >= upper_bound_acc:
                                bound1['upper'].append(upper_bound_dec)
                                bound1['lower'].append(lower_bound_dec)
                                bound1['v_BV'].append((cand_v_max - cand_v_min) / (len(cand_bound['v_BV']) - 1) * ind + cand_v_min)
                            elif upper_bound_dec > upper_bound_acc and upper_bound_acc > lower_bound_dec >= lower_bound_acc:
                                bound1['upper'].append(upper_bound_dec)
                                bound1['lower'].append(upper_bound_acc)
                                bound1['v_BV'].append((cand_v_max - cand_v_min) / (len(cand_bound['v_BV']) - 1) * ind + cand_v_min)
                            elif upper_bound_dec > upper_bound_acc and lower_bound_acc > lower_bound_dec:
                                bound1['upper'].append(upper_bound_dec)
                                bound1['lower'].append(upper_bound_acc)
                                bound1['v_BV'].append((cand_v_max - cand_v_min) / (len(cand_bound['v_BV']) - 1) * ind + cand_v_min)
                                bound2['upper'].append(lower_bound_acc)
                                bound2['lower'].append(lower_bound_dec)
                                bound2['v_BV'].append((cand_v_max - cand_v_min) / (len(cand_bound['v_BV']) - 1) * ind + cand_v_min)
                            elif upper_bound_acc >= upper_bound_dec >= lower_bound_acc and upper_bound_acc >= lower_bound_dec >= lower_bound_acc:
                                pass
                            elif upper_bound_acc >= upper_bound_dec > lower_bound_acc and lower_bound_acc > lower_bound_dec:
                                bound1['upper'].append(lower_bound_acc)
                                bound1['lower'].append(lower_bound_dec)
                                bound1['v_BV'].append((cand_v_max - cand_v_min) / (len(cand_bound['v_BV']) - 1) * ind + cand_v_min)
                            elif lower_bound_acc >= upper_bound_dec:
                                bound1['upper'].append(upper_bound_dec)
                                bound1['lower'].append(lower_bound_dec)
                                bound1['v_BV'].append((cand_v_max - cand_v_min) / (len(cand_bound['v_BV']) - 1) * ind + cand_v_min)
                            ind = ind + 1
                        cand_bound = bound1
            if len(cand_bound['v_BV']) > 1:
                cand_bounds.append(cand_bound)
            
        return cand_bounds

    def merge_same_risk_level(self, dec_ind, acc_ind):
        same_risk_level_bounds = []
        bound = {
            'risk_level': self.risk_levels[dec_ind],
            'upper': [],
            'lower': [],
            'v_BV': []
        }
        bound2 = {
            'risk_level': self.risk_levels[dec_ind],
            'upper': [],
            'lower': [],
            'v_BV': []
        }
        ind = 0
        flag = 1
        for upper_bound_dec, lower_bound_dec, upper_bound_acc, lower_bound_acc in zip(self.risk_level_bounds['dec'][dec_ind], self.risk_level_bounds['dec'][dec_ind + 1], self.risk_level_bounds['acc'][acc_ind + 1], self.risk_level_bounds['acc'][acc_ind]):
            if lower_bound_dec >= upper_bound_acc or lower_bound_acc > upper_bound_dec:
                if flag == 1 and (len(bound['upper']) > 1 or len(bound2['upper']) > 0):
                    if len(bound['upper']) > 0:
                        bound['upper'].append(max(upper_bound_dec, upper_bound_acc))
                        bound['lower'].append(min(lower_bound_dec, lower_bound_acc))
                        bound['v_BV'].append(self.step * ind + self.v_BV_min)
                        same_risk_level_bounds.append(bound)
                        bound = {
                            'risk_level': self.risk_levels[dec_ind],
                            'upper': [],
                            'lower': [],
                            'v_BV': []
                        }
                bound['upper'].append(max(upper_bound_dec, upper_bound_acc))
                bound['lower'].append(max(lower_bound_dec, lower_bound_acc))
                bound['v_BV'].append(self.step * ind + self.v_BV_min)
                bound2['upper'].append(min(upper_bound_dec, upper_bound_acc))
                bound2['lower'].append(min(lower_bound_dec, lower_bound_acc))
                bound2['v_BV'].append(self.step * ind + self.v_BV_min)
                flag = 2
            else:
                if flag == 2 and (len(bound['upper']) > 1 or len(bound2['upper']) > 0):
                    if len(bound['upper']) > 0:
                        bound['upper'].append(max(upper_bound_dec, upper_bound_acc))
                        bound['lower'].append(max(lower_bound_dec, lower_bound_acc))
                        bound['v_BV'].append(self.step * ind + self.v_BV_min)
                        same_risk_level_bounds.append(bound)
                        bound = {
                            'risk_level': self.risk_levels[dec_ind],
                            'upper': [],
                            'lower': [],
                            'v_BV': []
                        }
                    if len(bound2['upper']) > 0:
                        bound2['upper'].append(min(upper_bound_dec, upper_bound_acc))
                        bound2['lower'].append(min(lower_bound_dec, lower_bound_acc))
                        bound2['v_BV'].append(self.step * ind + self.v_BV_min)
                        same_risk_level_bounds.append(bound2)
                        bound2 = {
                            'risk_level': self.risk_levels[dec_ind],
                            'upper': [],
                            'lower': [],
                            'v_BV': []
                        }
                bound['upper'].append(max(upper_bound_dec, upper_bound_acc))
                bound['lower'].append(min(lower_bound_dec, lower_bound_acc))
                bound['v_BV'].append(self.step * ind + self.v_BV_min)
                flag = 1
            ind = ind + 1
        if len(bound['upper']) > 0:
            same_risk_level_bounds.append(bound)
        if len(bound2['upper']) > 0:
            same_risk_level_bounds.append(bound2)
        return same_risk_level_bounds

    def generate_merged_cases(self):
        print(f'[ {self.scenario} ] Generating cases...')
        generated_cases = self.covering_suite.k_wise_covering_suite(self.risk_levels[1:-1], self.merged_risk_level_bounds, 'v_BV', self.v_res, self.d_res, 1)
        for risk_level, cases in generated_cases.items():
            self.cases[risk_level] = [
                {
                    'dis': dis,
                    'sp': sp
                } for sp, dis in cases
            ]
        self.cases = {risk_level: self.cases[risk_level] for risk_level in ['low', 'mid', 'high'] if risk_level in self.cases.keys()}
        self.cases['v_AV'] = self.v_AV
        print(colored(f'[ {self.scenario} ] Cases for each merged risk level have been generated!', 'green'))

    def save_cases(self):
        data = json.dumps(self.cases)
        path = self.case_path + '/' + self.name + '.json'
        f = open(path, 'w')
        f.write(data)
        f.close()
        print(colored(f'[ {self.scenario} ] Cases have been save to ' + path, 'green'))
    
    def plot_cases(self):
        fig_format_setting(self.fontsize)
        x = list(np.arange(self.v_BV_min, self.v_BV_max, self.step))
        x.append(self.v_BV_max)
        x = np.concatenate((x, x[::-1]))
        for ind in range(len(self.risk_level_bounds['dec']) - 1):
            y = np.concatenate((self.risk_level_bounds['dec'][ind], self.risk_level_bounds['dec'][ind + 1][::-1]))
            plt.fill(x, y, facecolor=self.colors[self.risk_levels[ind]], alpha=self.alphas[self.risk_levels[ind]])
        for risk_level in self.risk_levels[1:4]:
            for one_case in self.cases[risk_level]:
                plt.plot(one_case['sp'], one_case['dis'], 'k.')
        plt.xlabel('BV speed (m/s)')
        plt.ylabel('Longitudinal distance (m)')
        
        path = self.case_path + '/' + self.name + '_dec.svg'
        plt.savefig(path, format='svg', bbox_inches='tight')
        print(colored(f'[ {self.scenario} ] Figure has been save to ' + path, 'green'))
        plt.close()

        fig_format_setting(self.fontsize)
        x = list(np.arange(self.v_BV_min, self.v_BV_max, self.step))
        x.append(self.v_BV_max)
        x = np.concatenate((x, x[::-1]))
        for ind in range(len(self.risk_level_bounds['acc']) - 1):
            y = np.concatenate((self.risk_level_bounds['acc'][ind], self.risk_level_bounds['acc'][ind + 1][::-1]))
            plt.fill(x, y, facecolor=self.colors[self.risk_levels[ind]], alpha=self.alphas[self.risk_levels[ind]])
        for risk_level in self.risk_levels[1:4]:
            for one_case in self.cases[risk_level]:
                plt.plot(one_case['sp'], one_case['dis'], 'k.')
        plt.xlabel('BV speed (m/s)')
        plt.ylabel('Longitudinal distance (m)')
        
        path = self.case_path + '/' + self.name + '_acc.svg'
        plt.savefig(path, format='svg', bbox_inches='tight')
        print(colored(f'[ {self.scenario} ] Figure has been save to ' + path, 'green'))
        plt.close()

    def plot_merged_cases(self):
        fig_format_setting(self.fontsize)
        for bound in self.merged_risk_level_bounds:
            x = np.concatenate((bound['v_BV'], bound['v_BV'][::-1]))
            y = np.concatenate((bound['upper'], bound['lower'][::-1]))
            plt.fill(x, y, facecolor=self.colors[bound['risk_level']], alpha=self.alphas[bound['risk_level']])
        for risk_level in self.risk_levels[1:4]:
            for one_case in self.cases[risk_level]:
                plt.plot(one_case['sp'], one_case['dis'], 'k.')
        plt.xlabel(r'BV speed ($m/s$)')
        plt.ylabel(r'Longitudinal distance ($m$)')
        path = self.case_path + '/' + self.name + '.svg'
        plt.savefig(path, format='svg', bbox_inches='tight')
        print(colored(f'[ {self.scenario} ] Merged case figure has been save to ' + path, 'green'))

    def choose_case(self, round_num=-1):
        for risk_level, cases in self.cases.items():
            if round_num < len(cases):
                selected_case = cases[round_num]
                real_case = {
                    'dis': selected_case['dis'] - math.sqrt(2 * self.turning_radius * self.lane_width - self.lane_width ** 2),  # distance to the conflict point along the track
                    'sp': selected_case['sp']
                }
                return risk_level, real_case, selected_case
            else:
                round_num -= len(cases)
        return "None", {}, {}

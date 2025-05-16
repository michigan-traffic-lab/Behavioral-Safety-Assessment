import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
import json
import os

from env.case_generation.covering_suite import *

from utils import *


class VRUWithoutCrosswalkGen:
    def __init__(self, settings):
        self.scenario = 'VRU Crossing the Street without Crosswalk'
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
        
        self.AV_length = settings.AV_length
        self.AV_width = settings.AV_width

        self.lateral_dis = settings.scenario_info[self.scenario]['lateral distance']
        self.dis2edge = settings.scenario_info[self.scenario]['lateral distance'] - settings.lane_width / 2
        # parameters for the RLS
        self.dec_bound = settings.scenario_info[self.scenario]['deceleration bound']
        self.t_reac = settings.scenario_info[self.scenario]['deceleration reaction time']
        self.dx_max = settings.scenario_info[self.scenario]['max initial distance']
        self.dx_min = settings.scenario_info[self.scenario]['min initial distance']

        self.v_AV_max = settings.scenario_info[self.scenario]['max AV speed']
        self.v_AV_min = settings.scenario_info[self.scenario]['min AV speed']
        self.step_AV = settings.scenario_info[self.scenario]['AV speed step']
        self.v_VRU_max = settings.scenario_info[self.scenario]['max VRU speed']
        self.v_VRU_min = settings.scenario_info[self.scenario]['min VRU speed']
        self.step_VRU = settings.scenario_info[self.scenario]['VRU speed step']

        self.v_res = settings.scenario_info[self.scenario]['speed resolution']
        self.d_res = settings.scenario_info[self.scenario]['distance resolution']

        self.risk_level_bounds = []
        self.risk_levels = settings.risk_levels[1:4]
        self.cases = {risk_level: [] for risk_level in self.risk_levels}
        self.case_num = 0

        self.v_AV = settings.AV_speed_list[settings.scenario_list.index(self.scenario)]
        self.v_AV_pool=np.arange(self.v_AV_min, self.v_AV_max + self.step_AV, self.step_AV)
        self.v_VRU_pool=np.arange(self.v_VRU_min, self.v_VRU_max + self.step_VRU, self.step_VRU)

        self.covering_suite = CoveringSuite()

        self.colors = list(settings.colors.values())
        self.alphas = list(settings.alphas.values())
        self.fontsize = settings.fontsize
    
    def init(self):
        feasibility_file = self.feasibility_path + '/' + self.name + '.npz'
        flag = False
        if not os.path.exists(feasibility_file):
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
            self.generate_risk_level_bounds()
            self.generate_cases()
            self.save_cases()
            self.plot_cases()

    def risk_level_gen(self):
        W0 = self.AV_width
        # L_min is from the AV starting point to the start of VRU Crossing the Street without Crosswalk point of BV
        L_min = np.zeros((len(self.dec_bound),len(self.v_VRU_pool),len(self.v_AV_pool)))

        # Assume the AV notice when BV start the LT
        for i in range(len(self.dec_bound)):
            for j in range(len(self.v_VRU_pool)):
                t1 = (self.lateral_dis - W0 / 2)/self.v_VRU_pool[j]  # moment when VRU enter the conflict zone
                t2 = (self.lateral_dis + W0 / 2)/self.v_VRU_pool[j]  # moment when VRU exit the conflict zone
                for k in range(len(self.v_AV_pool)):
                    if self.v_AV_pool[k] < -self.dec_bound[i]*(t2-self.t_reac[i]):  # AV speed is lower, AV can stop before entering conflict zone
                        L_min[i,j,k] = self.v_AV_pool[k]*self.t_reac[i] - (self.v_AV_pool[k]**2)*0.5/self.dec_bound[i]
                    else:  # AV speed is higher, cannot stop when VRU enters the conflict zone
                        L_min[i,j,k] = self.v_AV_pool[k]*t2 + 0.5*self.dec_bound[i]*max(t2-self.t_reac[i], 0)**2
        L_min[L_min>self.dx_max] = self.dx_max
        L_min[L_min<self.dx_min] = self.dx_min
        self.L_min = L_min

        path = self.feasibility_path + '/' + self.name + '.npz'
        np.savez_compressed(path, self.L_min)
        print(colored(f'[ {self.scenario} ] Analyzing results have been saved to ' + path, 'green'))

    def generate_risk_level_bounds(self):
        risk_level_data = self.feasibility_path + '/' + self.name + '.npz'
        try:
            self.L_min = np.load(risk_level_data)
            self.L_min = self.L_min['arr_0']
        except Exception as e:
            print(colored(f'[ {self.scenario} ] Fail to load risk level bounds!', 'red'))
            self.risk_level_gen()

        print(f'[ {self.scenario} ] Generating risk level...')
        ind = (self.v_AV - self.v_AV_min) / self.step_AV
        if abs(round(ind) - ind) < 0.01:
            ind = round(ind)
            self.risk_level_bounds = self.L_min[:, :, ind]
        else:
            ind1 = int(ind)
            ind2 = ind1 + 1
            v1_1 = self.v_AV_min + self.step_AV * ind1
            v1_2 = v1_1 + self.step_AV
            self.risk_level_bounds = ratio(self.v_AV, [v1_1, v1_2], [self.L_min[:, :, ind1], self.L_min[:, :, ind2]])
        self.risk_level_bounds = np.concatenate(
            (
                np.asarray([[self.dx_min for _ in range(len(self.risk_level_bounds[0]))]]),
                self.risk_level_bounds,
                np.asarray([[self.dx_max for _ in range(len(self.risk_level_bounds[0]))]])
            ),
            axis=0
        )
        print(colored(f'[ {self.scenario} ] Risk levels have been generated!', 'green'))
        np.savez_compressed(self.risk_level_path + '/' + self.name + '.npz', self.risk_level_bounds)
    
    def generate_cases(self):
        print(f'[ {self.scenario} ] Generating VRU Crossing the Street without Crosswalk cases...')
        generated_cases = self.covering_suite.k_wise_covering_suite(self.risk_levels, self.risk_level_bounds, self.v_VRU_pool, self.v_res, self.d_res, 1)
        for risk_level, cases in generated_cases.items():
            self.cases[risk_level] = [
                {
                    'dis to edge': self.dis2edge,
                    'alongside sp': sp,
                    'triggered dis': dis,
                    'crossing sp': sp,
                    'crossing dis': 6
                } for sp, dis in cases
            ]
        self.cases = {risk_level: self.cases[risk_level] for risk_level in ['low', 'mid', 'high'] if risk_level in self.cases.keys()}
        self.cases['v_AV'] = self.v_AV
        print(colored(f'[ {self.scenario} ] Cases for each risk level have been generated!', 'green'))

    def save_cases(self):
        data = json.dumps(self.cases)
        path = self.case_path + '/' + self.name + '.json'
        f = open(path, 'w')
        f.write(data)
        f.close()
        print(colored(f'[ {self.scenario} ] Cases have been save to ' + path, 'green'))

    def plot_cases(self):
        fig_format_setting(self.fontsize)
        x = np.concatenate((self.v_VRU_pool, self.v_VRU_pool[::-1]))
        for ind in range(len(self.risk_level_bounds) - 1):
            y = np.concatenate((self.risk_level_bounds[ind], self.risk_level_bounds[ind + 1][::-1]))
            plt.fill(x, y, facecolor=self.colors[ind], alpha=self.alphas[ind])
        for risk_level in self.risk_levels:
            for one_case in self.cases[risk_level]:
                plt.plot(one_case['crossing sp'], one_case['triggered dis'], 'k.')
        plt.xlabel(r'VRU speed ($m/s$)')
        plt.ylabel(r'Longitudinal distance ($m$)')
        path = self.case_path + '/' + self.name + '.svg'
        plt.savefig(path, format='svg', bbox_inches='tight')
        print(colored(f'[ {self.scenario} ] Figure has been save to ' + path, 'green'))
        plt.close()

    def choose_case(self, round_num=-1):
        for risk_level, cases in self.cases.items():
            if round_num < len(cases):
                selected_case = cases[round_num]
                return risk_level, selected_case, selected_case
            else:
                round_num -= len(cases)
        return "None", {}, {}

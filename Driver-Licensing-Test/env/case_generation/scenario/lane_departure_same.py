import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from termcolor import colored
import json
import random
import os

from env.case_generation.covering_suite import *

from utils import *


class LaneDepartureSameGen:
    def __init__(self, settings):
        self.scenario = 'Lane Departure (same direction)'
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
        for scenarios, length, width in zip(settings.BV_scenarios, settings.BV_length, settings.BV_width):
            if self.scenario in scenarios:
                self.BV_length = length
                self.BV_width = width
        self.amax = settings.scenario_info[self.scenario]['max acceleration']
        self.amin = settings.scenario_info[self.scenario]['min acceleration']
        self.t_reac = settings.scenario_info[self.scenario]['reaction time']
        self.t_lc = settings.scenario_info[self.scenario]['lane change duration']
        self.dx_min = settings.scenario_info[self.scenario]['min initial distance']
        self.dx_max = settings.scenario_info[self.scenario]['max initial distance']
        self.dv_max = settings.scenario_info[self.scenario]['max delta speed']
        self.dv_min = settings.scenario_info[self.scenario]['min delta speed']
        self.v_res = settings.scenario_info[self.scenario]['speed resolution']
        self.d_res = settings.scenario_info[self.scenario]['distance resolution']
        self.lane_width = settings.lane_width
        self.step = 0.1

        self.dv = np.arange(self.dv_min, self.dv_max + self.step, self.step)
        self.dx = []

        self.risk_level_bounds = []
        self.risk_levels = settings.risk_levels[1:4]
        self.cases = {risk_level: [] for risk_level in self.risk_levels}
        self.case_num = 0

        self.covering_suite = CoveringSuite()

        self.colors = list(settings.colors.values())
        self.alphas = list(settings.alphas.values())
        self.fontsize = settings.fontsize

    def init(self):
        risk_level_file = self.risk_level_path + '/' + self.name + '.npz'
        flag = False
        if not os.path.exists(risk_level_file):
            print(colored(f'[ {self.scenario} ] There is no risk level data!', 'red'))
            self.risk_level_gen()
            flag = True
        else:
            self.risk_level_bounds = np.load(risk_level_file)['arr_0']

        cases_data_file = self.case_path + '/' + self.name + '.json'
        if not os.path.exists(cases_data_file):
            print(colored(f'[ {self.scenario} ] There are no cases that have been generated! Expected file path is ' + cases_data_file + ' .', 'red'))
            flag = True
        else:
            openfile = open(cases_data_file, 'r')
            self.cases = json.load(openfile)

        if flag:
            self.generate_cases()
            self.save_cases()
            self.plot_cases()

        self.unexecuted_cases = self.cases
        self.unexecuted_risk_levels = self.risk_levels
        self.executed_cases = self.executed_cases = {risk_level: [] for risk_level in self.risk_levels}

    def risk_level_gen(self):
        dx =   np.zeros((len(self.dv),len(self.amin))) 
        TTC =  np.zeros((len(self.dv),len(self.amin)))
        t1 = self.t_lc / 2
        for i in range(len(self.amin)):
            for j in range(len(self.dv)):
                dx[j,i] = -self.dv[j]**2 /(2*self.amin[i]) + (self.t_reac[i]+t1)*self.dv[j]
                dx[j,i] = max(self.dx_min, dx[j,i])
                if self.dv[j]<=0:
                    TTC[j,i] = 100000
                else:
                    TTC[j,i] = dx[j,i]/self.dv[j]
        dx_max = self.dx_max
        dx[dx>dx_max] = dx_max
        self.dx = dx

        self.risk_level_bounds = np.concatenate(
            (
                np.asarray([[self.dx_min for _ in range(len(dx))]]),
                np.transpose(dx),
                np.asarray([[self.dx_max for _ in range(len(dx))]])
            ),
            axis=0
        )

        file = self.risk_level_path + '/' + self.name + '.npz'
        np.savez_compressed(file, self.risk_level_bounds)
        print(colored(f'[ {self.scenario} ] Analyzing results have been saved to ' + file, 'green'))

    def generate_cases(self):
        print(f'[ {self.scenario} ] Generating cases...')
        generated_cases = self.covering_suite.k_wise_covering_suite(self.risk_levels, self.risk_level_bounds, self.dv, self.v_res, self.d_res, 1)
        for risk_level, cases in generated_cases.items():
            self.cases[risk_level] = [
                {
                    'relative sp': sp,
                    'dis': dis,
                    'ratio': random.random() * 0.4 + 0.3,
                    'direction': 'same'
                } for sp, dis in cases
            ]
        self.cases = {risk_level: self.cases[risk_level] for risk_level in ['low', 'mid', 'high'] if risk_level in self.cases.keys()}
        print(colored(f'[ {self.scenario} ] Cases for each risk level have been generated!', 'green'))

    def save_cases(self):
        data = json.dumps(self.cases)
        file = self.case_path + '/' + self.name + '.json'
        f = open(file, 'w')
        f.write(data)
        f.close()
        print(colored(f'[ {self.scenario} ] Cases have been save to ' + file, 'green'))

    def plot_cases(self):
        fig_format_setting(self.fontsize)
        x = np.concatenate((self.dv, self.dv[::-1]))
        for ind in range(len(self.risk_level_bounds) - 1):
            y = np.concatenate((self.risk_level_bounds[ind], self.risk_level_bounds[ind + 1][::-1]))
            plt.fill(x, y, facecolor=self.colors[ind], alpha=self.alphas[ind])
        for risk_level in self.risk_levels:
            for one_case in self.cases[risk_level]:
                plt.plot(one_case['relative sp'], one_case['dis'], 'k.')
        plt.xlabel(r'Relative speed ($m/s$)')
        plt.ylabel(r'Relative distance ($m$)')
        path = self.case_path + '/' + self.name + '.svg'
        plt.savefig(path, format='svg', bbox_inches='tight')
        plt.close()
        print(colored(f'[ {self.scenario} ] Case figure has been save to ' + path, 'green'))

    def choose_case(self, round_num=-1):
        for risk_level, cases in self.cases.items():
            if round_num < len(cases):
                selected_case = cases[round_num]
                real_case = {
                    'relative sp': selected_case['relative sp'],
                    'dis': selected_case['dis'] + self.BV_length,
                    'ratio': selected_case['ratio'],
                    'direction': selected_case['direction']
                }
                return risk_level, real_case, selected_case
            else:
                round_num -= len(cases)
        return "None", {}, {}

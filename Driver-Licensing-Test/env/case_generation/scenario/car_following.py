import os
import math
import json
import random
from termcolor import colored
import matplotlib.pyplot as plt

from utils import *


class CarFollowingGen:
    def __init__(self, settings):
        self.scenario = 'Car Following'
        self.name = get_scenario_folder_name(self.scenario)
        self.feasibility_path = settings.test_info_path
        if not os.path.exists(self.feasibility_path):
            os.mkdir(self.feasibility_path)
        self.risk_level_path = settings.risk_level_path + '/' + self.name
        if not os.path.exists(self.risk_level_path):
            os.mkdir(self.risk_level_path)
        self.case_path = settings.test_case_path + '/' + self.name
        if not os.path.exists(self.case_path):
            os.mkdir(self.case_path)
        self.risk_levels = settings.risk_levels[1:4]
        self.risk_level_bounds = {}
        self.cases = {risk_level: [] for risk_level in self.risk_levels}
        self.AV_speed = settings.AV_speed_list[settings.scenario_list.index(self.scenario)]

        self.acc_res = settings.scenario_info[self.scenario]['acceleration resolution']
        self.dec_res = settings.scenario_info[self.scenario]['deceleration resolution']
        self.v_res = settings.scenario_info[self.scenario]['speed resolution']
        self.acc_bounds = settings.scenario_info[self.scenario]['acceleration bounds']
        self.dec_bounds = settings.scenario_info[self.scenario]['deceleration bounds']

        self.colors = list(settings.colors.values())
        self.alphas = list(settings.alphas.values())
        self.fontsize = settings.fontsize

    def init(self):
        risk_level_file = self.risk_level_path + '/car_following.json'
        if not os.path.exists(risk_level_file):
            print(colored(f'[ {self.scenario} ] There is no risk level data!', 'red'))
            self.generate_risk_level_bounds()
        else:
            f = open(risk_level_file)
            self.risk_level_bounds = json.load(f)
        
        cases_data_file = self.case_path + '/car_following.json'
        flag = False
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

    def generate_risk_level_bounds(self):
        self.risk_level_bounds = {
            'acc': self.acc_bounds,
            'dec': self.dec_bounds,
        }
        path = self.risk_level_path + '/car_following.json'
        f = open(path, 'w')
        f.write(json.dumps(self.risk_level_bounds))
        f.close()
        print(colored(f'[ {self.scenario} ] Risk levels have been generated and saved to ' + path, 'green'))

    def generate_cases(self):
        self.executed_cases = [0 for _ in self.risk_levels]
        for ind, risk_level in enumerate(self.risk_levels):
            case_num = math.floor(self.AV_speed / 2 / self.v_res[risk_level]) * math.floor((self.risk_level_bounds['dec'][ind+1] - self.risk_level_bounds['dec'][ind]) / self.dec_res[risk_level]) * max(math.floor((self.risk_level_bounds['acc'][ind] - self.risk_level_bounds['acc'][ind + 1])) / self.acc_res[risk_level], 1)
            for _ in range(int(case_num)):
                self.cases[risk_level].append(
                    {
                        'acc': random.random() * (self.risk_level_bounds['acc'][ind] - self.risk_level_bounds['acc'][ind+1]) + self.risk_level_bounds['acc'][ind+1],
                        'dec': random.random() * (self.risk_level_bounds['dec'][ind+1] - self.risk_level_bounds['dec'][ind]) + self.risk_level_bounds['dec'][ind],
                        'sp': random.random() * self.AV_speed / 2 + self.AV_speed / 2
                    }
                )
        self.cases = {risk_level: self.cases[risk_level] for risk_level in ['low', 'mid', 'high'] if risk_level in self.cases.keys()}
        print(colored(f'[ {self.scenario} ] Cases for each risk level have been generated!', 'green'))

    def save_cases(self):
        data = json.dumps(self.cases)
        file = self.case_path + '/car_following.json'
        f = open(file, 'w')
        f.write(data)
        f.close()
        print(colored(f'[ {self.scenario} ] Cases have been save to ' + file, 'green'))
    
    def plot_cases(self):
        fig_format_setting(self.fontsize)
        y = np.concatenate([np.ones(100) * 0, np.ones(100) * 10])
        risk_level_bound = [-8] + self.risk_level_bounds['dec'] + [0]
        for i in range(len(risk_level_bound) - 1):
            x = np.linspace(risk_level_bound[i], risk_level_bound[i + 1], 100)
            x = np.concatenate([x, x[::-1]])
            plt.fill(x, y, facecolor=self.colors[i], alpha=self.alphas[i])
        for risk_level in self.cases.keys():
            for case in self.cases[risk_level]:
                plt.plot(case['dec'], case['sp'], 'k.')
        plt.xlabel(r'BV deceleration ($m/s^2$)')
        plt.ylabel(r'BV velocity ($m/s$)')
        path = self.case_path + '/car_following.svg'
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

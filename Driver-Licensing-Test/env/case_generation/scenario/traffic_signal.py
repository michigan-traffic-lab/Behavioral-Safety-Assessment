import matplotlib.pyplot as plt
from termcolor import colored
import json
import os

from env.case_generation.covering_suite import *

from utils import *


class TrafficSignalGen:
    def __init__(self, settings):
        self.scenario = 'Traffic Signal'
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

        self.thw_max = settings.scenario_info[self.scenario]['max thw']
        self.thw_min = settings.scenario_info[self.scenario]['min thw']
        self.thw_res = settings.scenario_info[self.scenario]['thw resolution']

        self.v_AV = settings.AV_speed_list[settings.scenario_list.index(self.scenario)]

        self.risk_level_bounds = []
        self.risk_levels = settings.risk_levels[1:4]
        self.cases = {risk_level: [] for risk_level in self.risk_levels}
        self.case_num = 0

        self.fontsize = settings.fontsize

    def init(self):
        flag = False
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

    def generate_cases(self):
        print(f'[ {self.scenario} ] Generating cases...')
        case_num = int((self.thw_max - self.thw_min) / self.thw_res['low'])
        self.cases = {
            'low': [{'thw': random.random() * (self.thw_max - self.thw_min) + self.thw_min} for _ in range(case_num)],
            'mid': [],
            'high': []
        }
        self.cases = {risk_level: self.cases[risk_level] for risk_level in ['low', 'mid', 'high'] if risk_level in self.cases.keys()}
        print(colored(f'[ {self.scenario} ] Cases for each risk level have been generated!', 'green'))

    def save_cases(self):
        data = json.dumps(self.cases)
        path = self.case_path + '/' + self.name + '.json'
        f = open(path, 'w')
        f.write(data)
        f.close()
        print(colored(f'[ {self.scenario} ] Cases have been save to ' + path, 'green'))

    def plot_cases(self):
        fig_format_setting(self.fontsize, 'Traffic Signal')
        for risk_level in self.risk_levels:
            for one_case in self.cases[risk_level]:
                plt.plot(one_case['thw'], 0, 'k.')
        plt.xlabel(r'Time headway ($s$)')
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

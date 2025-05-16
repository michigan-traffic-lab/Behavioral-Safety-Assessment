import os
import math

from env.case_generation.scenario.right_turn_case_generation import *
from env.case_generation.scenario.right_turn_turn_feasibility_analysis import *
from env.case_generation.scenario.right_turn_feasibility_plot import *


class RightTurnTurnGen:
    def __init__(self, settings):
        self.scenario = 'Right Turn (AV turns right)'
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

        self.min_AV_sp = settings.scenario_info[self.scenario]['min AV speed']
        self.max_AV_sp = settings.scenario_info[self.scenario]['max AV speed']
        self.turning_radius = settings.scenario_info[self.scenario]['turning radius']
        self.v_AV = settings.AV_speed_list[settings.scenario_list.index(self.scenario)]
        scenario_info = settings.scenario_info[self.scenario]
        self.risk_levels = settings.risk_levels[1:4]
        self.risk_levels.reverse()

        self.feasibility_analysis = AnalyzeFeasibility(self.scenario, self.name, settings)
        scenario_info['min initial distance'] = self.feasibility_analysis.min_dis
        risk_levels = settings.risk_levels
        risk_levels.reverse()
        self.case_gen = GenerateCases(self.scenario, self.name, scenario_info, self.v_AV, risk_levels)
        self.case_plot = PlotCases(self.scenario, self.name, scenario_info, self.v_AV, risk_levels, settings.colors, settings.alphas, settings.fontsize)

        self.AV_length = settings.AV_length
        self.AV_width = settings.AV_width

    def init(self):
        dec_feasibility_data_file = self.feasibility_path + '/' + self.name + '_dec.npz'
        acc_feasibility_data_file = self.feasibility_path + '/' + self.name + '_acc.npz'
        flag = False
        if not os.path.exists(dec_feasibility_data_file) or not os.path.exists(acc_feasibility_data_file):
            print(colored(f'[ {self.scenario} ] There is no feasibility data!', 'red'))
            self.feasibility_analysis.analyze_feasibility()
            self.feasibility_analysis.save(self.feasibility_path)
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
            self.generate_cases()
            self.plot_cases()
            openfile = open(cases_data_file, 'r')
            self.cases = json.load(openfile)

    def analyze_feasibility(self, v_AV_min=None, v_AV_max=None):
        self.feasibility_analysis.analyze_feasibility(v_AV_min, v_AV_max)
        self.feasibility_analysis.save(self.feasibility_path)

    def generate_cases(self):
        flag = False
        while not flag:
            flag = self.case_gen.speed_check()
            if not flag:
                if self.v_AV > self.max_AV_sp:
                    self.analyze_feasibility(v_AV_max=int(self.v_AV) + 1)
                if self.v_AV < self.min_AV_sp:
                    self.analyze_feasibility(v_AV_min=int(self.v_AV))
        self.case_gen.generate_risk_level_bounds(self.feasibility_path, a_flag='dec')
        self.case_gen.generate_risk_level_bounds(self.feasibility_path, a_flag='acc')
        self.case_gen.merge_risk_level_bounds()
        self.case_gen.save_risk_levels(self.risk_level_path)
        self.case_gen.generate_merged_cases()
        self.case_gen.save(self.case_path)

    def plot_cases(self):
        self.case_plot.plot_cases(self.case_gen.risk_level_bounds['dec'], self.case_gen.cases, self.case_path, self.name + '_dec')
        self.case_plot.plot_cases(self.case_gen.risk_level_bounds['acc'], self.case_gen.cases, self.case_path, self.name + '_acc')
        self.case_plot.plot_merged_cases(self.case_gen.merged_risk_level_bounds, self.case_gen.cases, self.case_path)

    def choose_case(self, round_num=-1):
        for risk_level, cases in self.cases.items():
            if round_num < len(cases):
                selected_case = cases[round_num]
                real_case = {
                    'dis': selected_case['dis'] - self.turning_radius + self.turning_radius * math.pi / 2 - self.AV_length / 2,  # distance to the conflict point along the track
                    'sp': selected_case['sp']
                }
                return risk_level, real_case, selected_case
            else:
                round_num -= len(cases)
        return "None", {}, {}

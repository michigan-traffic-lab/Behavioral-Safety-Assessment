import os
import json

from env.case_generation.scenario.roundabout_case_generation import GenerateCases
from env.case_generation.scenario.roundabout_feasibility_analysis import *
from env.case_generation.scenario.roundabout_feasibility_plot import *

from utils import cal_dis_along_path


class RoundaboutGen:
    def __init__(self, scenario, settings):
        self.scenario = scenario
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

        ind = 0
        for scenarios in settings.BV_scenarios:
            if self.scenario in scenarios:
                break
            ind += 1
        settings.scenario_info[self.scenario]['stop line to conflict point distance'] = cal_dis_along_path(settings.scenario_info[self.scenario]['challenger']['init ind'], settings.scenario_info[self.scenario]['challenger']['conflict ind'], settings.BV_routes[ind])

        scenario_info = settings.scenario_info[self.scenario]
        self.risk_levels = settings.risk_levels[1:4]
        self.v_AV = settings.AV_speed_list[settings.scenario_list.index(self.scenario)]
        self.v_AV_max = settings.scenario_info[self.scenario]['max AV speed']
        self.v_AV_min = settings.scenario_info[self.scenario]['min AV speed']

        all_risk_levels = settings.risk_levels
        all_risk_levels.reverse()
        self.case_gen = GenerateCases(self.scenario, self.name, scenario_info, self.v_AV, all_risk_levels)
        self.feasibility_analysis = AnalyzeFeasibility(self.scenario, self.name, settings)
        self.case_plot = PlotCases(self.scenario, self.name, scenario_info, self.v_AV, all_risk_levels, settings.colors, settings.alphas, settings.fontsize)

        self.cases = {risk_level: [] for risk_level in self.risk_levels}
        self.case_num = 0

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
                print(colored(f'[ {self.scenario} ] Case speed is wrong! Expected ' + str(self.cases['v_AV']) + ' but now is ' + str(self.v_AV) + ' exist.', 'red'))
                flag = True
        if flag:
            self.generate_cases()
            self.plot_cases()
            openfile = open(cases_data_file, 'r')
            self.cases = json.load(openfile)

        self.unexecuted_cases = self.cases
        self.unexecuted_risk_levels = []
        for key in self.risk_levels:
            if len(self.unexecuted_cases[key]) > 0:
                self.unexecuted_risk_levels.append(key)
        self.executed_cases = {risk_level: [] for risk_level in self.risk_levels}

    def analyze_feasibility(self, v_AV_min=None, v_AV_max=None):
        self.feasibility_analysis.analyze_feasibility(v_AV_min, v_AV_max)
        self.feasibility_analysis.save(self.feasibility_path)

    def generate_cases(self):
        flag = False
        while not flag:
            flag = self.case_gen.speed_check(self.v_AV)
            if not flag:
                if self.v_AV > self.v_AV_max:
                    self.analyze_feasibility(v_AV_max=int(self.v_AV) + 1)
                if self.v_AV < self.v_AV_min:
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
                return risk_level, selected_case, selected_case
            else:
                round_num -= len(cases)
        return "None", {}, {}

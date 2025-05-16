from numpy import arange, concatenate
import matplotlib.pyplot as plt
from termcolor import colored

from env.case_generation.scenario.right_turn_case_generation import *
from utils import *



class PlotCases():
    def __init__(self,
                 scenario,
                 name,
                 scenario_info,
                 v_AV,
                 risk_levels,
                 colors,
                 alphas,
                 fontsize,
                 ):
        self.scenario = scenario
        self.name = name
        self.v_AV = v_AV
        self.risk_levels = risk_levels
        self.colors = colors
        self.alphas = alphas
        self.fontsize = fontsize

        self.min_dis = scenario_info['min initial distance']
        self.max_dis = scenario_info['max initial distance']
        self.min_AV_sp = scenario_info['min AV speed']
        self.max_AV_sp = scenario_info['max AV speed']
        self.min_BV_sp = scenario_info['min BV speed']
        self.max_BV_sp = scenario_info['max BV speed']

        self.dis_step = scenario_info['distance step']
        self.AV_sp_step = scenario_info['AV speed step']
        self.BV_sp_step = scenario_info['BV speed step']
        self.acc_step = scenario_info['acceleration step']

    def plot_cases(self, risk_level_bounds, cases, figure_path, fig_name):
        fig_format_setting(self.fontsize)
        x = list(arange(self.min_BV_sp, self.max_BV_sp, self.BV_sp_step))
        x.append(self.max_BV_sp)
        x = concatenate((x, x[::-1]))
        for ind in range(len(risk_level_bounds) - 1):
            y = concatenate((risk_level_bounds[ind], risk_level_bounds[ind + 1][::-1]))
            plt.fill(x, y, facecolor=self.colors[self.risk_levels[ind]], alpha=self.alphas[self.risk_levels[ind]])
        for risk_level in self.risk_levels[1:4]:
            for one_case in cases[risk_level]:
                plt.plot(one_case['sp'], one_case['dis'], 'k.')
        plt.xlabel(r'BV speed ($m/s$)')
        plt.ylabel(r'Longitudinal distance ($m$)')
        path = figure_path + '/' + fig_name + '.svg'
        plt.savefig(path, format='svg', bbox_inches='tight')
        print(colored(f'[ {self.scenario} ] Case figure has been save to ' + path, 'green'))
        plt.close()
    
    def plot_merged_cases(self, merged_risk_level_bounds, cases, figure_path):
        fig_format_setting(self.fontsize)
        for bound in merged_risk_level_bounds:
            x = concatenate((bound['v_BV'], bound['v_BV'][::-1]))
            y = concatenate((bound['upper'], bound['lower'][::-1]))
            plt.fill(x, y, facecolor=self.colors[bound['risk_level']], alpha=self.alphas[bound['risk_level']])
        for risk_level in self.risk_levels[1:4]:
            for one_case in cases[risk_level]:
                plt.plot(one_case['sp'], one_case['dis'], 'k.')
        plt.xlabel(r'BV speed ($m/s$)')
        plt.ylabel(r'Longitudinal distance ($m$)')
        path = figure_path + '/' + self.name + '.svg'
        plt.savefig(path, format='svg', bbox_inches='tight')
        print(colored(f'[ {self.scenario} ] Merged case figure has been save to ' + path, 'green'))
        plt.close()

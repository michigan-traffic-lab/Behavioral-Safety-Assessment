from numpy import load, transpose
from termcolor import colored
import json

from env.case_generation.covering_suite import *
from utils import *



class GenerateCases():
    def __init__(self,
                 scenario,
                 name,
                 scenario_info,
                 v_AV,
                 risk_levels,
                 ):
        self.scenario = scenario
        self.name = name
        self.v_AV = v_AV
        self.flag = True
        self.risk_level_bounds = {
            'acc': [],
            'dec': []
        }
        self.merged_risk_level_bounds = []
        self.cases = {}
        self.turning_radius = scenario_info['turning radius']
        self.acc_react_time = scenario_info['acceleration reaction time']
        self.dec_react_time = scenario_info['deceleration reaction time']
        self.min_dis = scenario_info['min initial distance']
        self.max_dis = scenario_info['max initial distance']
        self.min_AV_sp = scenario_info['min AV speed']
        self.max_AV_sp = scenario_info['max AV speed']
        self.min_BV_sp = scenario_info['min BV speed']
        self.max_BV_sp = scenario_info['max BV speed']
        self.acc_bound = scenario_info['acceleration bound']
        self.dec_bound = scenario_info['deceleration bound']

        self.dis_step = scenario_info['distance step']
        self.AV_sp_step = scenario_info['AV speed step']
        self.BV_sp_step = scenario_info['BV speed step']
        self.acc_step = scenario_info['acceleration step']

        self.dis_num = (self.max_dis - self.min_dis) / self.dis_step + 1
        self.AV_sp_num = (self.max_AV_sp - self.min_AV_sp) / self.AV_sp_step + 1
        self.BV_sp_num = (self.max_BV_sp - self.min_BV_sp) / self.BV_sp_step + 1
        
        self.risk_levels = risk_levels
        self.d_res = scenario_info['distance resolution']
        self.v_res = scenario_info['speed resolution']

        self.bounds = {
            'acc': scenario_info['acceleration bound'],
            'dec': scenario_info['deceleration bound']
        }

        self.covering_suite = CoveringSuite()

    def speed_check(self):
        if self.v_AV > self.max_AV_sp or self.v_AV < self.min_AV_sp:
            print(colored(f'[ {self.scenario} ] AV speed is out of range!!!', 'red'))
            self.flag = False
        else:
            self.flag = True
        return self.flag

    def generate_risk_level_bounds(self, feasibility_path, a_flag='dec'):
        feasibility_data = feasibility_path + '/' + self.name + '_' + a_flag + '.npz'
        if self.flag:
            print(f'[ {self.scenario} ] Generating risk level...')
            record = load(feasibility_data)
            record = record['arr_0']
            step = (self.max_AV_sp - self.min_AV_sp) / (self.AV_sp_num - 1)
            ind = (self.v_AV - self.min_AV_sp) / step
            if abs(round(ind) - ind) < 0.01:
                ind = round(ind)
                a_matrix = record[:, ind, :]
            else:
                ind1 = int(ind)
                ind2 = ind1 + 1
                v1_1 = self.min_AV_sp + step * ind1
                v1_2 = v1_1 + step
                a_matrix = ratio(self.v_AV, [v1_1, v1_2], [record[:, ind1, :], record[:, ind2, :]])

            a_matrix = transpose(a_matrix)
            self.risk_level_bounds[a_flag] = []
            for bound in self.bounds[a_flag]:
                temp = []
                for accs in a_matrix:
                    flag = True
                    for ind in range(len(accs)):
                        if (accs[ind] < bound and a_flag == 'dec') or (accs[ind] > bound and a_flag == 'acc'):
                            if ind == 0:
                                temp.append(self.min_dis)
                            elif ind == len(accs) - 1:
                                temp.append(self.max_dis)
                            else:
                                d_1 = (self.max_dis - self.min_dis) / self.dis_num * (ind - 1) + self.min_dis
                                d_2 = (self.max_dis - self.min_dis) / self.dis_num * ind + self.min_dis
                                bound_d = ratio(bound, [accs[ind - 1], accs[ind]], [d_1, d_2])
                                temp.append(bound_d)
                            flag = False
                            break
                    if flag:
                        temp.append(self.max_dis)
                self.risk_level_bounds[a_flag].append(temp)
            print(colored(f'[ {self.scenario} ] Risk levels have been generated!', 'green'))
        else:
            print(colored(f'[ {self.scenario} ] AV speed is out of range!!!', 'red'))

    def merge_risk_level_bounds(self):
        print('Merging risk levels...')
        for dec_ind in range(len(self.risk_level_bounds['dec']) - 1):
            same_risk_level_bounds = self.merge_same_risk_level(dec_ind, dec_ind)
            deleted_risk_level_bounds = self.del_risk_level(same_risk_level_bounds, dec_ind)
            for bound in deleted_risk_level_bounds:
                self.merged_risk_level_bounds.append(bound)
        print(colored(f'[ {self.scenario} ] Merged risk levels have been generated!', 'green'))
        
    def save_risk_levels(self, risk_level_path):
        path = risk_level_path + '/' + self.name + '.json'
        f = open(path, 'w')
        f.write(json.dumps(self.merged_risk_level_bounds))
        f.close()
        print(colored(f'[ {self.scenario} ] Risk levels have been generated and saved to ' + path, 'green'))

    def del_risk_level(self, risk_level_bounds, dec_ind):
        cand_bounds = []
        for bound in risk_level_bounds:
            cand_bound = bound
            for merged_risk_level_bound in self.merged_risk_level_bounds:
                if len(cand_bound['v_BV']) > 0:
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
                    if not (cand_v_min >= merged_v_max or cand_v_max <= merged_v_min) and cand_v_min < cand_v_max:
                        for ind in range(len(cand_bound['v_BV'])):
                            v_BV = cand_bound['v_BV'][ind]
                            upper_bound_dec = cand_bound['upper'][ind]
                            lower_bound_dec = cand_bound['lower'][ind]
                            if v_BV in merged_risk_level_bound['v_BV']:
                                merged_ind = merged_risk_level_bound['v_BV'].index(v_BV)
                                upper_bound_acc = merged_risk_level_bound['upper'][merged_ind]
                                lower_bound_acc = merged_risk_level_bound['lower'][merged_ind]
                                if lower_bound_dec >= upper_bound_acc:
                                    bound1['upper'].append(upper_bound_dec)
                                    bound1['lower'].append(lower_bound_dec)
                                    bound1['v_BV'].append(cand_bound['v_BV'][ind])
                                elif upper_bound_dec > upper_bound_acc and upper_bound_acc > lower_bound_dec >= lower_bound_acc:
                                    bound1['upper'].append(upper_bound_dec)
                                    bound1['lower'].append(upper_bound_acc)
                                    bound1['v_BV'].append(cand_bound['v_BV'][ind])
                                elif upper_bound_dec > upper_bound_acc and lower_bound_acc > lower_bound_dec:
                                    bound1['upper'].append(upper_bound_dec)
                                    bound1['lower'].append(upper_bound_acc)
                                    bound1['v_BV'].append(cand_bound['v_BV'][ind])
                                    bound2['upper'].append(lower_bound_acc)
                                    bound2['lower'].append(lower_bound_dec)
                                    bound2['v_BV'].append(cand_bound['v_BV'][ind])
                                elif upper_bound_acc >= upper_bound_dec >= lower_bound_acc and upper_bound_acc >= lower_bound_dec >= lower_bound_acc:
                                    pass
                                elif upper_bound_acc >= upper_bound_dec > lower_bound_acc and lower_bound_acc > lower_bound_dec:
                                    bound1['upper'].append(lower_bound_acc)
                                    bound1['lower'].append(lower_bound_dec)
                                    bound1['v_BV'].append(cand_bound['v_BV'][ind])
                                elif lower_bound_acc >= upper_bound_dec:
                                    bound1['upper'].append(upper_bound_dec)
                                    bound1['lower'].append(lower_bound_dec)
                                    bound1['v_BV'].append(cand_bound['v_BV'][ind])
                            else:
                                bound1['upper'].append(upper_bound_dec)
                                bound1['lower'].append(lower_bound_dec)
                                bound1['v_BV'].append(cand_bound['v_BV'][ind])
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
                        bound['v_BV'].append((self.max_BV_sp - self.min_BV_sp) / (self.BV_sp_num - 1) * ind + self.min_BV_sp)
                        same_risk_level_bounds.append(bound)
                        bound = {
                            'risk_level': self.risk_levels[dec_ind],
                            'upper': [],
                            'lower': [],
                            'v_BV': []
                        }
                bound['upper'].append(max(upper_bound_dec, upper_bound_acc))
                bound['lower'].append(max(lower_bound_dec, lower_bound_acc))
                bound['v_BV'].append((self.max_BV_sp - self.min_BV_sp) / (self.BV_sp_num - 1) * ind + self.min_BV_sp)
                bound2['upper'].append(min(upper_bound_dec, upper_bound_acc))
                bound2['lower'].append(min(lower_bound_dec, lower_bound_acc))
                bound2['v_BV'].append((self.max_BV_sp - self.min_BV_sp) / (self.BV_sp_num - 1) * ind + self.min_BV_sp)
                flag = 2
            else:
                if flag == 2 and (len(bound['upper']) > 1 or len(bound2['upper']) > 0):
                    if len(bound['upper']) > 0:
                        bound['upper'].append(max(upper_bound_dec, upper_bound_acc))
                        bound['lower'].append(max(lower_bound_dec, lower_bound_acc))
                        bound['v_BV'].append((self.max_BV_sp - self.min_BV_sp) / (self.BV_sp_num - 1) * ind + self.min_BV_sp)
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
                        bound2['v_BV'].append((self.max_BV_sp - self.min_BV_sp) / (self.BV_sp_num - 1) * ind + self.min_BV_sp)
                        same_risk_level_bounds.append(bound2)
                        bound2 = {
                            'risk_level': self.risk_levels[dec_ind],
                            'upper': [],
                            'lower': [],
                            'v_BV': []
                        }
                bound['upper'].append(max(upper_bound_dec, upper_bound_acc))
                bound['lower'].append(min(lower_bound_dec, lower_bound_acc))
                bound['v_BV'].append((self.max_BV_sp - self.min_BV_sp) / (self.BV_sp_num - 1) * ind + self.min_BV_sp)
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

    def save(self, case_path):
        data = json.dumps(self.cases)
        path = case_path + '/' + self.name + '.json'
        f = open(path, 'w')
        f.write(data)
        f.close()
        print(colored(f'[ {self.scenario} ] Cases have been save to ' + path, 'green'))

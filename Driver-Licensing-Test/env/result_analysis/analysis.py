import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from rich import progress
import copy
import json

from utils import *


class Analysis():
    def __init__(self,
                 settings) -> None:
        self.settings = settings
        self.scenario_list = settings.scenario_list
        self.AV_speed_list = settings.AV_speed_list
        self.save_path = settings.evaluation_path
        self.risk_level_path = settings.risk_level_path

        self.AV_length = settings.AV_length
        self.AV_width = settings.AV_width
        self.BV_length = settings.BV_length
        self.BV_width = settings.BV_width
        self.VRU_length = settings.VRU_length
        self.VRU_width = settings.VRU_width

        self.scenario_info = settings.scenario_info

        self.BV_scenarios = settings.BV_scenarios
        self.AV_route = {}
        self.AV_route_seg_len = {}
        self.AV_conflict_ind = {}
        self.BV_routes = []
        self.BV_route_seg_len = []
        self.BV_conflict_ind = []
        self.BV_length = []
        self.BV_width = []

        # BV scenarios
        for scenario_list, route, length, width in zip(self.BV_scenarios, settings.BV_routes, settings.BV_length, settings.BV_width):
            bv_routes = {}
            bv_route_seg_len = {}
            bv_conflict_ind = {}
            bv_length = {}
            bv_width = {}
            for scenario in scenario_list:
                av_start_ind = settings.scenario_info[scenario]['AV']['start ind']
                av_end_ind = settings.scenario_info[scenario]['AV']['end ind']
                av_route = settings.AV_route[av_start_ind:av_end_ind + 1]
                bv_start_ind = settings.scenario_info[scenario]['challenger']['start ind']
                bv_end_ind = settings.scenario_info[scenario]['challenger']['end ind']
                bv_route = route[bv_start_ind:bv_end_ind + 1]

                AV_conflict_ind, BV_conflict_ind = cal_route_conflict_ind(av_route, self.AV_length, self.AV_width, bv_route, length, width)
                self.AV_route[scenario] = av_route
                self.AV_route_seg_len[scenario] = cal_route_segment_length(av_route)
                self.AV_conflict_ind[scenario] = AV_conflict_ind
                bv_routes[scenario] = bv_route
                bv_route_seg_len[scenario] = cal_route_segment_length(bv_route)
                bv_conflict_ind[scenario] = BV_conflict_ind
                bv_length[scenario] = length
                bv_width[scenario] = width
            self.BV_routes.append(bv_routes)
            self.BV_route_seg_len.append(bv_route_seg_len)
            self.BV_conflict_ind.append(bv_conflict_ind)
            self.BV_length.append(bv_length)
            self.BV_width.append(bv_width)

        # VRU scenarios
        self.VRU_scenarios = settings.VRU_scenarios
        self.VRU_routes = []
        self.VRU_length = []
        self.VRU_width = []
        for scenario_list, length, width in zip(self.VRU_scenarios, settings.VRU_length, settings.VRU_width):
            vru_length = {}
            vru_width = {}
            for scenario in scenario_list:
                av_start_ind = settings.scenario_info[scenario]['AV']['start ind']
                av_end_ind = settings.scenario_info[scenario]['AV']['end ind']
                av_route = settings.AV_route[av_start_ind:av_end_ind + 1]

                self.AV_route[scenario] = av_route
                vru_length[scenario] = length
                vru_width[scenario] = width
            self.VRU_length.append(vru_length)
            self.VRU_width.append(vru_width)
        
        # the scenarios not involved with BV or VRU
        bv_scenarios = []
        for scenario_list in self.BV_scenarios:
            bv_scenarios += scenario_list
        vru_scenarios = []
        for scenario_list in self.VRU_scenarios:
            vru_scenarios += scenario_list
        rest_AV_scenarios = list(set(self.scenario_list) - set(bv_scenarios) - set(vru_scenarios))
        for scenario in rest_AV_scenarios:
            av_start_ind = settings.scenario_info[scenario]['AV']['start ind']
            av_end_ind = settings.scenario_info[scenario]['AV']['end ind']
            av_route = settings.AV_route[av_start_ind:av_end_ind + 1]
            self.AV_route[scenario] = av_route
            self.AV_route_seg_len[scenario] = cal_route_segment_length(av_route)
            self.AV_conflict_ind[scenario] = -1

        self.same_route_scenarios = ['Cut-In', 'Car Following', 'Lane Departure (opposite direction)', 'Lane Departure (same direction)', 'VRU Crossing the Street without Crosswalk', 'VRU Crossing the Street at the Crosswalk', 'Vehicle Encroachment']

        self.plot_path = settings.plot_path
        self.fontsize = settings.fontsize

    def analyze_results(self):
        passed_num_set = {}
        failed_num_set = {}
        passed_cases_set = {}
        failed_cases_set = {}
        avg_min_dis_set = {}
        avg_min_ttc_set = {}
        avg_reaction_time_set = {}
        max_reaction_time_set = {}
        avg_reaction_dis_set = {}
        min_reaction_dis_set = {}
        all_min_dis_set = {}
        all_min_ttc_set = {}
        all_reaction_time_set = {}
        all_reaction_dis_set = {}
        initial_dis_err_set = {}
        initial_sp_err_set = {}
        av_init_sp_set = {}
        for scenario, AV_speed in zip(self.scenario_list, self.AV_speed_list):
            scenario_folder_names = [get_scenario_folder_name(scenario)]
            for scenario_folder_name in scenario_folder_names:
                data_path = self.settings.test_data_path + '/' + scenario_folder_name
                evaluation_path = self.save_path + '/' + scenario_folder_name
                os.makedirs(evaluation_path, exist_ok=True)
                statistical_key, statistical_data, track_key, track_data, flag = self._load_data(data_path, scenario)
                all_final_time, _ = self._find_final_time(scenario, statistical_key, statistical_data, track_key, track_data)
                if scenario not in ['Traffic Signal']:
                    _, _, all_reaction_timing_set = self._analyze_reaction_timing(scenario, statistical_key, statistical_data, track_key, track_data)
                else:
                    all_reaction_timing_set = {}
                self._plot_case(scenario, statistical_key, statistical_data, track_key, track_data, all_final_time, all_reaction_timing_set)
                if flag:
                    passed_cases, failed_cases, passed_statistical_data, passed_track_data, failed_statistical_data, failed_track_data, passed_num, failed_num = self._analyze_pass_or_fail(scenario, statistical_key, statistical_data, track_key, track_data, evaluation_path)
                    if scenario in ['Vehicle Encroachment']:
                        stoped_cases, stoped_statistical_data, stoped_track_data = self._check_stop(scenario, statistical_key, passed_statistical_data, track_key, passed_track_data)
                    if scenario in ['Traffic Signal']:
                        self._analyze_traffic_signal(scenario, statistical_key, passed_statistical_data, track_key, passed_track_data, evaluation_path)
                        min_dis_set = {}
                        min_ttc_set = {}
                        reaction_time_set = {}
                        max_reaction_time = 0
                        reaction_dis_set = {}
                        min_reaction_dis = 0
                        reaction_timing_set = {}
                    else:
                        final_time, duration = self._find_final_time(scenario, statistical_key, passed_statistical_data, track_key, passed_track_data)
                        _, _, real_dis, real_sp = self._analyze_initial_error(statistical_key, passed_statistical_data, evaluation_path)
                        min_ttc_set, init_ttc_set = self._analyze_minimum_time_to_collision(scenario, statistical_key, passed_statistical_data, track_key, passed_track_data, final_time, evaluation_path, real_dis, real_sp)
                        min_dis_set = self._analyze_minimum_distance(scenario, statistical_key, passed_statistical_data, track_key, passed_track_data, final_time, evaluation_path, real_dis, real_sp, init_ttc_set)
                        reaction_time_set, reaction_dis_set, reaction_timing_set = self._analyze_reaction_timing(scenario, statistical_key, passed_statistical_data, track_key, passed_track_data)
                    dis_errs, sp_errs, _, _ = self._analyze_initial_error(statistical_key, statistical_data, evaluation_path)
                    av_init_sp = self._analyze_AV_initial_speed(scenario, statistical_key, statistical_data, evaluation_path)

                    if scenario == 'Car Following':
                        self._analyze_following_distance(scenario, statistical_key, passed_statistical_data, track_key, passed_track_data, final_time, evaluation_path)

                    passed_num_set[scenario] = passed_num
                    failed_num_set[scenario] = failed_num
                    passed_cases_set[scenario] = passed_cases
                    failed_cases_set[scenario] = failed_cases
                    avg_min_dis_set[scenario] = self._cal_avg_value(min_dis_set)
                    avg_min_ttc_set[scenario] = self._cal_avg_value(min_ttc_set)
                    avg_reaction_time_set[scenario] = self._cal_avg_value(reaction_time_set)
                    reaction_time = []
                    for value in reaction_time_set.values():
                        reaction_time += value
                    if len(reaction_time) > 0:
                        tmp_reaction_time = [value for value in reaction_time if value < math.inf]
                        if len(tmp_reaction_time) > 0:
                            max_reaction_time = max(tmp_reaction_time)
                        else:
                            max_reaction_time = 0
                    else:
                        max_reaction_time = 0
                    max_reaction_time_set[scenario] = max_reaction_time
                    avg_reaction_dis_set[scenario] = self._cal_avg_value(reaction_dis_set)
                    reaction_dis = []
                    for value in reaction_dis_set.values():
                        reaction_dis += value
                    if len(reaction_dis) > 0:
                        min_reaction_dis = min(reaction_dis)
                    else:
                        min_reaction_dis = 0
                    min_reaction_dis_set[scenario] = min_reaction_dis

                    all_min_dis_set[scenario] = min_dis_set
                    all_min_ttc_set[scenario] = min_ttc_set
                    all_reaction_time_set[scenario] = reaction_time_set
                    all_reaction_dis_set[scenario] = reaction_dis_set
                    initial_dis_err_set[scenario] = dis_errs
                    initial_sp_err_set[scenario] = sp_errs
                    av_init_sp_set[scenario] = av_init_sp
        return passed_num_set, failed_num_set, passed_cases_set, failed_cases_set, avg_min_dis_set, avg_min_ttc_set, avg_reaction_time_set, max_reaction_time_set, avg_reaction_dis_set, min_reaction_dis_set, all_min_dis_set, all_min_ttc_set, all_reaction_time_set, all_reaction_dis_set, initial_dis_err_set, initial_sp_err_set, av_init_sp_set

    def _check_stop(self, scenario, statistical_key, statistical_data, track_key, track_data):
        stop_cases = []
        stop_statistical_data = []
        stop_track_data = []
        if scenario == 'Vehicle Encroachment':
            av_sp_ind = track_key.index('AV sp')
            for statistic, track in zip(statistical_data, track_data):
                if track[-1][av_sp_ind] < 1:
                    stop_cases.append(statistic)
                    stop_statistical_data.append(statistic)
                    stop_track_data.append(track)
        return stop_cases, stop_statistical_data, stop_track_data
    
    def _cal_avg_value(self, value_set):
        tmp_set = []
        for value in value_set.values():
            tmp_set += value
        avg_value = 0
        count = 0
        for v in tmp_set:
            if v < math.inf:
                avg_value += v
                count += 1
        if count > 0:
            avg_value /= count
        return avg_value

    def _load_data(self, data_path, scenario):
        statistical_key, statistical_data = read_csv(data_path + '/record.csv')
        initial_timestamp_ind = statistical_key.index('init timestamp')
        track_data = []
        files = os.listdir(data_path)
        N = len(files)
        ite = 0
        for n in progress.track(range(N-1), description='Loading data from ' + data_path):
            track_file = data_path + '/' + str(ite) + '.csv'
            while not os.path.exists(track_file):
                print('File ', track_file, ' does not exist!')
                ite += 1
                track_file = data_path + '/' + str(ite) + '.csv'
            track_file = data_path + '/' + str(ite) + '.csv'
            track_key, value = read_csv(track_file)
            track_data.append(value)
            ite += 1

        effective_init_condition = []
        effective_ind = []
        for i in range(len(statistical_data)):
            if statistical_data[i][1:initial_timestamp_ind] not in effective_init_condition:
                effective_init_condition.append(statistical_data[i][1:initial_timestamp_ind])
                effective_ind.append(i)
        if len(effective_init_condition) == N - 1:
            statistical_data = [statistical_data[ind] for ind in effective_ind]
            track_data = [track_data[i] for i in effective_ind]
            return statistical_key, statistical_data, track_key, track_data, True
        else:
            print('There are ', len(effective_init_condition), ' cases, but only ', N - 1, ' of them are recorded!')
            return statistical_key, statistical_data, track_key, track_data, False

    def _determine_risk_level(self, risk_level_bound, acc):
        tmp = risk_level_bound + [acc]
        tmp.sort()
        risk_level_ind = tmp.index(acc)
        if len(risk_level_bound) == 6:
            risk_level_ind -= 1
            if risk_level_ind < 0:
                risk_level_ind = 0
            elif risk_level_ind > 4:
                risk_level_ind = 4
        real_risk_level = ['trivial', 'low', 'mid', 'high', 'infs'][risk_level_ind]
        return real_risk_level, risk_level_ind

    def _analyze_traffic_signal(self, scenario, statistical_key, statistical_data, track_key, track_data, path):
        risk_level_ind = statistical_key.index('risk level')
        timestamp_ind = track_key.index('timestamp')
        av_x_ind = track_key.index('AV x')
        av_y_ind = track_key.index('AV y')
        av_heading_ind = track_key.index('AV heading')
        traffic_light_state_ind = track_key.index('traffic light state')

        traffic_light_pos = self.settings.AV_route[self.settings.scenario_info[scenario]['AV']['traffic light ind']]
        traffic_light_ctrl_inds = []
        for traffic_light_scenarios in self.settings.test_info['traffic signal']['scenarios']:
            if scenario in traffic_light_scenarios:
                traffic_light_ctrl_ind = self.settings.test_info['traffic signal']['index'][self.settings.test_info['traffic signal']['scenarios'].index(traffic_light_scenarios)][0]
                break
        init_dis = {}
        init_sp = {}
        pass_stop = {}
        for statistic, track in zip(statistical_data, track_data):
            risk_level = statistic[risk_level_ind]
            if risk_level not in list(pass_stop.keys()):
                pass_stop[risk_level] = []
            if risk_level not in list(init_dis.keys()):
                init_dis[risk_level] = []
            if risk_level not in list(init_sp.keys()):
                init_sp[risk_level] = []
            turn_to_yellow = False
            turn_to_red = False
            for data in track:
                if data[traffic_light_state_ind][traffic_light_ctrl_ind] == 'y' and not turn_to_yellow:
                    init_dis[risk_level].append(cal_dis([data[av_x_ind], data[av_y_ind]], traffic_light_pos))
                    init_sp[risk_level].append(data[track_key.index('AV sp')])
                    turn_to_yellow = True
                if data[traffic_light_state_ind][traffic_light_ctrl_ind] == 'r':
                    turn_to_red = True
                    if data[av_y_ind] > traffic_light_pos[1]:
                        pass_stop[risk_level].append('pass')
                        break
                    else:
                        pass_stop[risk_level].append('stop')
                        break
            if not turn_to_red:
                if track[-1][av_y_ind] > traffic_light_pos[1]:
                    pass_stop[risk_level].append('pass')
                else:
                    pass_stop[risk_level].append('stop')

        return 

    def _analyze_pass_or_fail(self, scenario, statistical_key, statistical_data, track_key, track_data, path):
        risk_level_ind = statistical_key.index('risk level')
        init_condition_1_ind = None
        init_condition_2_ind = None
        if scenario in ['Cut-In', 'Lane Departure (same direction)']:
            risk_level_bound = self.settings.scenario_info[scenario]['min acceleration']
            risk_level_bound = abs(np.asarray(risk_level_bound))
            risk_level_bound = list(risk_level_bound)
            risk_level_bound.sort()
            init_condition_1_ind = statistical_key.index('relative sp')
            init_condition_2_ind = statistical_key.index('dis')
            init_condition_1 = r'Relative speed ($m/s$)'
            init_condition_2 = r'Relative distance ($m$)'
        elif scenario in ['Lane Departure (opposite direction)']:
            risk_level_bound = self.settings.scenario_info[scenario]['lane change duration']
            risk_level_bound.sort()
            init_condition_1_ind = statistical_key.index('relative sp')
            init_condition_2_ind = statistical_key.index('dis')
            init_condition_1 = r'Relative speed ($m/s$)'
            init_condition_2 = r'Relative distance ($m$)'
        elif scenario in ['Left Turn (AV goes straight)', 'Left Turn (AV turns left)']:
            challenger_sp_ind = statistical_key.index('sp')
            acc_reaction_time = self.settings.scenario_info[scenario]['acceleration reaction time'][0]
            dec_reaction_time = self.settings.scenario_info[scenario]['deceleration reaction time'][0]
            acc_risk_level_bound = self.settings.scenario_info[scenario]['max acceleration']
            dec_risk_level_bound = self.settings.scenario_info[scenario]['min acceleration']
            acc_risk_level_bound = abs(np.asarray(acc_risk_level_bound))
            dec_risk_level_bound = abs(np.asarray(dec_risk_level_bound))
            acc_risk_level_bound = list(acc_risk_level_bound)
            dec_risk_level_bound = list(dec_risk_level_bound)
            acc_risk_level_bound.sort()
            dec_risk_level_bound.sort()
            risk_level_bound = dec_risk_level_bound
            init_condition_1_ind = statistical_key.index('sp')
            init_condition_2_ind = statistical_key.index('dis')
            init_condition_1 = r'BV speed ($m/s$)'
            init_condition_2 = r'Longitudinal distance ($m$)'
        elif scenario in ['Right Turn (AV goes straight)', 'Right Turn (AV turns right)', 'AV Merging into the Roundabout', 'BV Merging into the Roundabout']:
            challenger_sp_ind = statistical_key.index('sp')
            acc_reaction_time = self.settings.scenario_info[scenario]['acceleration reaction time']
            dec_reaction_time = self.settings.scenario_info[scenario]['deceleration reaction time']
            acc_risk_level_bound = self.settings.scenario_info[scenario]['acceleration bound']
            dec_risk_level_bound = self.settings.scenario_info[scenario]['deceleration bound']
            acc_risk_level_bound = abs(np.asarray(acc_risk_level_bound))
            dec_risk_level_bound = abs(np.asarray(dec_risk_level_bound))
            acc_risk_level_bound = list(acc_risk_level_bound)
            dec_risk_level_bound = list(dec_risk_level_bound)
            acc_risk_level_bound.sort()
            dec_risk_level_bound.sort()
            risk_level_bound = dec_risk_level_bound
            init_condition_1_ind = statistical_key.index('sp')
            init_condition_2_ind = statistical_key.index('dis')
            init_condition_1 = r'BV speed ($m/s$)'
            init_condition_2 = r'Longitudinal distance ($m$)'
        elif scenario in ['VRU Crossing the Street at the Crosswalk', 'VRU Crossing the Street without Crosswalk']:
            risk_level_bound = self.settings.scenario_info[scenario]['deceleration bound']
            risk_level_bound = abs(np.asarray(risk_level_bound))
            risk_level_bound = list(risk_level_bound)
            risk_level_bound.sort()
            init_condition_1_ind = statistical_key.index('crossing sp')
            init_condition_2_ind = statistical_key.index('triggered dis')
            init_condition_1 = r'VRU speed ($m/s$)'
            init_condition_2 = r'Longitudinal distance ($m$)'
        elif scenario in ['Car Following']:
            detailed_risk_level_file = self.risk_level_path + '/' + get_scenario_folder_name(scenario) + '/' + get_scenario_folder_name(scenario) + '.json'
            with open(detailed_risk_level_file, 'r') as f:
                detailed_risk_level_bounds = json.load(f)
            risk_level_bound = detailed_risk_level_bounds['dec']
            risk_level_bound = abs(np.asarray(risk_level_bound))
            risk_level_bound = list(risk_level_bound)
            risk_level_bound.sort()
            init_condition_1_ind = statistical_key.index('dec')
            init_condition_2_ind = statistical_key.index('sp')
            init_condition_1 = r'BV deceleration ($m/s^2$)'
            init_condition_2 = r'BV speed ($m/s$)'
        elif scenario in ['Vehicle Encroachment']:
            risk_level_bound = [-5.05164471455172, -2.450781893794381, -1.534516401802792, -0.07866796019828293]
            init_condition_1_ind = statistical_key.index('dis')
            init_condition_2_ind = statistical_key.index('angle')
            init_condition_1 = r'Distance between BV center and lane center($m$)'
            init_condition_2 = r'Angle ($^{\circ}$)'
        elif scenario in ['Traffic Signal']:
            risk_level_bound = [-5.05164471455172, -2.450781893794381, -1.534516401802792, -0.07866796019828293]

        timestamp_ind = track_key.index('timestamp')
        passed_cases = {}
        passed_num = {'all': 0}
        failed_cases = {}
        failed_num = {'all': 0}
        passed_statistical_data = []
        passed_track_data = []
        failed_statistical_data = []
        failed_track_data = []
        if scenario not in ['Traffic Signal']:
            for statistic, track, num in zip(statistical_data, track_data, range(len(statistical_data))):
                risk_level = statistic[risk_level_ind]
                if risk_level not in list(passed_cases.keys()):
                    passed_cases[risk_level] = []
                if risk_level not in list(failed_cases.keys()):
                    failed_cases[risk_level] = []
                if risk_level not in list(passed_num.keys()):
                    passed_num[risk_level] = 0
                if risk_level not in list(failed_num.keys()):
                    failed_num[risk_level] = 0
                if track[-1][timestamp_ind] == 0:
                    failed_cases[risk_level].append(num)
                    failed_num['all'] += 1
                    failed_num[risk_level] += 1
                    failed_statistical_data.append(statistic)
                    failed_track_data.append(track)
                else:
                    passed_cases[risk_level].append(num)
                    passed_num['all'] += 1
                    passed_num[risk_level] += 1
                    passed_statistical_data.append(statistic)
                    passed_track_data.append(track)
        else:
            scenario_ind = self.settings.scenario_list.index(scenario)
            av_path = self.AV_route[scenario][self.settings.scenario_info[scenario]['AV']['start ind']:self.settings.scenario_info[scenario]['AV']['end ind'] + 1]
            traffic_light_ind_on_route = self.settings.scenario_info[scenario]['AV']['traffic light ind'] - self.settings.scenario_info[scenario]['AV']['start ind']
            traffic_light_ind = self.settings.traffic_signal_index[scenario_ind][0]
            traffic_light_state_ind = track_key.index('traffic light state')
            for statistic, track, num in zip(statistical_data, track_data, range(len(statistical_data))):
                risk_level = statistic[risk_level_ind]
                if risk_level not in list(passed_cases.keys()):
                    passed_cases[risk_level] = []
                if risk_level not in list(failed_cases.keys()):
                    failed_cases[risk_level] = []
                if risk_level not in list(passed_num.keys()):
                    passed_num[risk_level] = 0
                if risk_level not in list(failed_num.keys()):
                    failed_num[risk_level] = 0
                av_x_ind = track_key.index('AV x')
                av_y_ind = track_key.index('AV y')
                NNPN = -1
                traffic_light_flag = False
                pass_flag = True
                for point in track:
                    NNPN = find_next_route_points(av_path, [point[av_x_ind], point[av_y_ind]], NNPN)
                    if NNPN > traffic_light_ind_on_route and not traffic_light_flag:
                        traffic_light_flag = True
                        if point[traffic_light_state_ind][traffic_light_ind] == 'r':
                            failed_cases[risk_level].append(num)
                            failed_num['all'] += 1
                            failed_num[risk_level] += 1
                            failed_statistical_data.append(statistic)
                            failed_track_data.append(track)
                            pass_flag = False
                            break
                if pass_flag:
                    passed_cases[risk_level].append(num)
                    passed_num['all'] += 1
                    passed_num[risk_level] += 1
                    passed_statistical_data.append(statistic)
                    passed_track_data.append(track)

        print('Pass percent: ', passed_num['all'] / (passed_num['all'] + failed_num['all']) * 100, '%')
        for risk_level in list(passed_num.keys()):
            print(risk_level, ': passed cases / all cases: ', passed_num[risk_level], '/', passed_num[risk_level] + failed_num[risk_level])

        return passed_cases, failed_cases, passed_statistical_data, passed_track_data, failed_statistical_data, failed_track_data, passed_num, failed_num

    def _analyze_initial_error(self, statistical_key, statistical_data, path):
        risk_level_ind = statistical_key.index('risk level')
        dis_err_ind = statistical_key.index('dis err')
        sp_err_ind = statistical_key.index('sp err')
        initial_timestamp_ind = statistical_key.index('init timestamp')
        dis_ind = -1
        sp_ind = -1
        for key in statistical_key:
            if 'dis' in key and 'err' not in key and 'edge' not in key and 'crossing' not in key:
                dis_ind = statistical_key.index(key)
                break
        for key in statistical_key:
            if 'sp' in key and 'err' not in key and 'alongside' not in key:
                sp_ind = statistical_key.index(key)
                break
        dis_errs = {}
        sp_errs = {}
        real_dis = {}
        real_sp = {}
        for line in statistical_data:
            if line[initial_timestamp_ind] == 0:
                continue
            risk_level = line[risk_level_ind]
            if risk_level not in list(dis_errs.keys()):
                dis_errs[risk_level] = []
            if risk_level not in list(sp_errs.keys()):
                sp_errs[risk_level] = []
            dis_errs[risk_level].append(line[dis_err_ind])
            sp_errs[risk_level].append(line[sp_err_ind])
            if dis_ind != -1:
                if risk_level not in list(real_dis.keys()):
                    real_dis[risk_level] = []
                real_dis[risk_level].append(line[dis_ind] + line[dis_err_ind])
            if sp_ind != -1:
                if risk_level not in list(real_sp.keys()):
                    real_sp[risk_level] = []
                real_sp[risk_level].append(line[sp_ind])

        # plot initial distance error
        fig, ax = fig_format_setting(self.fontsize)
        plt.boxplot(list(dis_errs.values()), patch_artist=True)
        plt.setp(ax, xticklabels=list(dis_errs.keys()))
        ax.set_xlabel(r'Risk level')
        ax.set_ylabel(r'Initial distance error ($m$)')
        plt.savefig(path + '/initial_distance_error.svg', format='svg', bbox_inches='tight')
        plt.close()

        return dis_errs, sp_errs, real_dis, real_sp
    
    def _analyze_AV_initial_speed(self, scenario, statistical_key, statistical_data, path):
        risk_level_ind = statistical_key.index('risk level')
        av_sp_ind = statistical_key.index('AV init sp')
        initial_timestamp_ind = statistical_key.index('init timestamp')
        scenario_ind = self.settings.scenario_list.index(scenario)
        av_init_sp = {}
        for line in statistical_data:
            if line[initial_timestamp_ind] == 0:
                continue
            risk_level = line[risk_level_ind]
            if risk_level not in list(av_init_sp.keys()):
                av_init_sp[risk_level] = []
            av_init_sp[risk_level].append(line[av_sp_ind])
        
        # plot initial distance error
        fig, ax = fig_format_setting(self.fontsize)
        plt.boxplot(list(av_init_sp.values()), patch_artist=True)
        plt.setp(ax, xticklabels=list(av_init_sp.keys()))
        xlim = ax.get_xlim()
        plt.hlines(y=self.settings.AV_speed_list[scenario_ind], xmin=xlim[0], xmax=xlim[1], colors='r', linestyles='dashed')
        plt.xlim(xlim)
        ax.set_xlabel(r'Risk level')
        ax.set_ylabel(r'AV initial speed ($m/s$)')
        plt.savefig(path + '/AV_initial_speed.svg', format='svg', bbox_inches='tight')
        plt.close()
        return av_init_sp

    def _analyze_minimum_distance(self, scenario, statistical_key, statistical_data, track_key, track_data, final_time, path, real_init_dis, real_init_sp, init_ttc_set):
        risk_level_ind = statistical_key.index('risk level')
        initial_timestamp_ind = statistical_key.index('init timestamp')
        timestamp_ind = track_key.index('timestamp')
        av_x_ind = track_key.index('AV x')
        av_y_ind = track_key.index('AV y')
        av_heading_ind = track_key.index('AV heading')
        challenger_x_ind = track_key.index('challenger x')
        challenger_y_ind = track_key.index('challenger y')
        challenger_heading_ind = track_key.index('challenger heading')
        flag = True
        for BV_scenario_list in self.BV_scenarios:
            if scenario in BV_scenario_list:
                ind = self.BV_scenarios.index(BV_scenario_list)
                challenger_length = self.BV_length[ind][scenario]
                challenger_width = self.BV_width[ind][scenario]
                flag = False
                break
        for VRU_scenario_list in self.VRU_scenarios:
            if scenario in VRU_scenario_list:
                ind = self.VRU_scenarios.index(VRU_scenario_list)
                challenger_length = self.VRU_length[ind][scenario]
                challenger_width = self.VRU_width[ind][scenario]
                flag = False
                break
        if flag:
            return {}

        min_dis_set = {}
        min_dis_list = []
        for statistic, track, end_t in zip(statistical_data, track_data, final_time):
            if statistic[initial_timestamp_ind] == 0:
                continue
            risk_level = statistic[risk_level_ind]
            if risk_level not in list(min_dis_set.keys()):
                min_dis_set[risk_level] = []
            min_dis = math.inf
            for data in track:
                if end_t >= data[timestamp_ind] >= statistic[initial_timestamp_ind]:
                    # dis = cal_dis([data[av_x_ind], data[av_y_ind]], [data[challenger_x_ind], data[challenger_y_ind]])
                    dis = cal_min_dis_of_rectangles([data[av_x_ind], data[av_y_ind], self.AV_length, self.AV_width, (90 - data[av_heading_ind]) / 180 * math.pi],
                                                    [data[challenger_x_ind], data[challenger_y_ind], challenger_length, challenger_width, (90 - data[challenger_heading_ind]) / 180 * math.pi])
                    if dis < min_dis:
                        min_dis = copy.deepcopy(dis)
            min_dis_set[risk_level].append(min_dis)
            min_dis_list.append(min_dis)

        return min_dis_set

    def _analyze_minimum_time_to_collision(self, scenario, statistical_key, statistical_data, track_key, track_data, final_time, path, real_init_dis, real_init_sp):
        risk_level_ind = statistical_key.index('risk level')
        initial_timestamp_ind = statistical_key.index('init timestamp')
        timestamp_ind = track_key.index('timestamp')
        av_x_ind = track_key.index('AV x')
        av_y_ind = track_key.index('AV y')
        av_sp_ind = track_key.index('AV sp')
        av_heading_ind = track_key.index('AV heading')
        challenger_x_ind = track_key.index('challenger x')
        challenger_y_ind = track_key.index('challenger y')
        challenger_sp_ind = track_key.index('challenger sp')
        challenger_heading_ind = track_key.index('challenger heading')

        AV_route = self.AV_route[scenario]
        flag = True
        for BV_scenario_list in self.BV_scenarios:
            if scenario in BV_scenario_list:
                AV_conflict_ind = self.AV_conflict_ind[scenario]
                AV_seg_len = self.AV_route_seg_len[scenario]
                ind = self.BV_scenarios.index(BV_scenario_list)
                challenger_route = self.BV_routes[ind][scenario]
                challenger_conflict_ind = self.BV_conflict_ind[ind][scenario]
                challenger_seg_len = self.BV_route_seg_len[ind][scenario]
                challenger_length = self.BV_length[ind][scenario]
                challenger_width = self.BV_width[ind][scenario]
                flag = False
                break
        for VRU_scenario_list in self.VRU_scenarios:
            if scenario in VRU_scenario_list:
                ind = self.VRU_scenarios.index(VRU_scenario_list)
                challenger_length = self.VRU_length[ind][scenario]
                challenger_width = self.VRU_width[ind][scenario]
                flag = False
                break
        if flag:
            return {}

        min_ttc_set = {}
        init_ttc_set = {}
        for i in progress.track(range(len(statistical_data)), description='Analyzing TTC'):
            if statistical_data[i][initial_timestamp_ind] == 0:
                continue
            init_ttc_flag = False
            statistic, one_track, end_t = statistical_data[i], track_data[i], final_time[i]
            risk_level = statistic[risk_level_ind]
            if risk_level not in list(min_ttc_set.keys()):
                min_ttc_set[risk_level] = []
            if risk_level not in list(init_ttc_set.keys()):
                init_ttc_set[risk_level] = []
            min_ttc = math.inf
            min_ind = 0
            for data in one_track:
                if end_t >= data[timestamp_ind] >= statistic[initial_timestamp_ind] and math.cos((data[av_heading_ind] - data[challenger_heading_ind]) / 180 * math.pi) * data[challenger_sp_ind] < data[av_sp_ind]:
                    ttc = math.inf
                    if data[av_sp_ind] > 0 and data[challenger_sp_ind] >= 0:
                        if scenario in self.same_route_scenarios:
                            if scenario in ['VRU Crossing the Street without Crosswalk', 'VRU Crossing the Street at the Crosswalk']:
                                av_corners = cal_corners([data[av_x_ind], data[av_y_ind], self.AV_length, self.AV_width, (90 - data[av_heading_ind]) / 180 * math.pi])
                                left_points = [av_corners[0], av_corners[1]]
                                right_points = [av_corners[2], av_corners[3]]
                                if not are_points_on_same_side(left_points[0], left_points[1], [data[av_x_ind], data[av_y_ind]], [data[challenger_x_ind], data[challenger_y_ind]]) or \
                                    not are_points_on_same_side(right_points[0], right_points[1], [data[av_x_ind], data[av_y_ind]], [data[challenger_x_ind], data[challenger_y_ind]]):
                                    continue
                            relative_sp = data[av_sp_ind] - data[challenger_sp_ind] * math.cos((data[challenger_heading_ind] - data[av_heading_ind]) / 180 * math.pi)
                            if relative_sp > 0:
                                
                                rotated_challenger_position = rotate_point2_around_point1([data[av_x_ind], data[av_y_ind]], [data[challenger_x_ind], data[challenger_y_ind]], (90 - data[av_heading_ind]) / 180 * math.pi)
                                rotated_challenger_heading = (data[av_heading_ind] - data[challenger_heading_ind]) / 180 * math.pi
                                ttc = (rotated_challenger_position[0] - data[av_x_ind] - self.AV_length / 2 - challenger_length / 2 * abs(math.cos(rotated_challenger_heading)) - challenger_width / 2 * abs(math.sin(rotated_challenger_heading))) / relative_sp
                                if ttc < 0:
                                    ttc = math.inf
                            else:
                                ttc = math.inf
                        else:
                            AV_rectangle = [data[av_x_ind], data[av_y_ind], self.AV_length, self.AV_width, (90 - data[av_heading_ind]) / 180 * math.pi]
                            challenger_rectangle = [data[challenger_x_ind], data[challenger_y_ind], challenger_length, challenger_width, (90 - data[challenger_heading_ind]) / 180 * math.pi]
                            _, ttc = cal_TTC_with_route(AV_rectangle, data[av_sp_ind], AV_route, AV_conflict_ind, AV_seg_len, challenger_rectangle, data[challenger_sp_ind], challenger_route, challenger_conflict_ind, challenger_seg_len)
                    if ttc < min_ttc:
                        min_ttc = copy.deepcopy(ttc)
                        min_ind = one_track.index(data)
                    if not init_ttc_flag:
                        init_ttc_flag = True
                        init_ttc_set[risk_level].append(ttc)
            if min_ttc > 10:
                tmp = 0
            if not init_ttc_flag:
                init_ttc_set[risk_level].append(math.inf)
                init_ttc_flag = True
            min_ttc_set[risk_level].append(min_ttc)

        return min_ttc_set, init_ttc_set

    def _analyze_reaction_timing(self, scenario, statistical_key, statistical_data, track_key, track_data):
        risk_level_ind = statistical_key.index('risk level')
        initial_timestamp_ind = statistical_key.index('init timestamp')
        initial_av_sp_ind = statistical_key.index('AV init sp')
        timestamp_ind = track_key.index('timestamp')
        av_x_ind = track_key.index('AV x')
        av_y_ind = track_key.index('AV y')
        av_sp_ind = track_key.index('AV sp')
        av_heading_ind = track_key.index('AV heading')
        av_acceleration_ind = track_key.index('AV acc')
        av_lon_acceleration_ind = track_key.index('AV lon acc')
        challenger_x_ind = track_key.index('challenger x')
        challenger_y_ind = track_key.index('challenger y')
        challenger_sp_ind = track_key.index('challenger sp')
        challenger_acceleration_ind = track_key.index('challenger acc')
        challenger_heading_ind = track_key.index('challenger heading')
        flag = True
        if scenario in ['Cut-In', 'Lane Departure (same direction)']:
            risk_level_bound = self.settings.scenario_info[scenario]['min acceleration']
            risk_level_bound = abs(np.asarray(risk_level_bound))
            risk_level_bound = list(risk_level_bound)
            risk_level_bound.sort()
            init_condition_1_ind = statistical_key.index('relative sp')
            init_condition_2_ind = statistical_key.index('dis')
            init_condition_1 = r'Relative speed ($m/s$)'
            init_condition_2 = r'Distance ($m$)'
        elif scenario in ['Lane Departure (opposite direction)']:
            risk_level_bound = self.settings.scenario_info[scenario]['lane change duration']
            risk_level_bound.sort()
            init_condition_1_ind = statistical_key.index('relative sp')
            init_condition_2_ind = statistical_key.index('dis')
            init_condition_1 = r'Relative speed ($m/s$)'
            init_condition_2 = r'Distance ($m$)'
        elif scenario in ['Left Turn (AV goes straight)', 'Left Turn (AV turns left)']:
            challenger_sp_ind = statistical_key.index('sp')
            acc_reaction_time = self.settings.scenario_info[scenario]['acceleration reaction time'][0]
            dec_reaction_time = self.settings.scenario_info[scenario]['deceleration reaction time'][0]
            acc_risk_level_bound = self.settings.scenario_info[scenario]['max acceleration']
            dec_risk_level_bound = self.settings.scenario_info[scenario]['min acceleration']
            acc_risk_level_bound = abs(np.asarray(acc_risk_level_bound))
            dec_risk_level_bound = abs(np.asarray(dec_risk_level_bound))
            acc_risk_level_bound = list(acc_risk_level_bound)
            dec_risk_level_bound = list(dec_risk_level_bound)
            acc_risk_level_bound.sort()
            dec_risk_level_bound.sort()
            risk_level_bound = dec_risk_level_bound
            init_condition_1_ind = statistical_key.index('sp')
            init_condition_2_ind = statistical_key.index('dis')
            init_condition_1 = r'BV speed ($m/s$)'
            init_condition_2 = r'Distance ($m$)'
        elif scenario in ['Right Turn (AV goes straight)', 'Right Turn (AV turns right)', 'AV Merging into the Roundabout', 'BV Merging into the Roundabout']:
            challenger_sp_ind = statistical_key.index('sp')
            acc_reaction_time = self.settings.scenario_info[scenario]['acceleration reaction time']
            dec_reaction_time = self.settings.scenario_info[scenario]['deceleration reaction time']
            acc_risk_level_bound = self.settings.scenario_info[scenario]['acceleration bound']
            dec_risk_level_bound = self.settings.scenario_info[scenario]['deceleration bound']
            acc_risk_level_bound = abs(np.asarray(acc_risk_level_bound))
            dec_risk_level_bound = abs(np.asarray(dec_risk_level_bound))
            acc_risk_level_bound = list(acc_risk_level_bound)
            dec_risk_level_bound = list(dec_risk_level_bound)
            acc_risk_level_bound.sort()
            dec_risk_level_bound.sort()
            risk_level_bound = dec_risk_level_bound
            init_condition_1_ind = statistical_key.index('sp')
            init_condition_2_ind = statistical_key.index('dis')
            init_condition_1 = r'BV speed ($m/s$)'
            init_condition_2 = r'Distance ($m$)'
        elif scenario in ['VRU Crossing the Street at the Crosswalk', 'VRU Crossing the Street without Crosswalk']:
            risk_level_bound = self.settings.scenario_info[scenario]['deceleration bound']
            risk_level_bound = abs(np.asarray(risk_level_bound))
            risk_level_bound = list(risk_level_bound)
            risk_level_bound.sort()
            init_condition_1_ind = statistical_key.index('crossing sp')
            init_condition_2_ind = statistical_key.index('triggered dis')
            init_condition_1 = r'VRU speed ($m/s$)'
            init_condition_2 = r'Distance'
        elif scenario in ['Car Following']:
            detailed_risk_level_file = self.risk_level_path + '/' + get_scenario_folder_name(scenario) + '/' + get_scenario_folder_name(scenario) + '.json'
            with open(detailed_risk_level_file, 'r') as f:
                detailed_risk_level_bounds = json.load(f)
            risk_level_bound = detailed_risk_level_bounds['dec']
            risk_level_bound = abs(np.asarray(risk_level_bound))
            risk_level_bound = list(risk_level_bound)
            risk_level_bound.sort()
            init_condition_1_ind = statistical_key.index('dec')
            init_condition_2_ind = statistical_key.index('sp')
            init_condition_1 = r'BV deceleration ($m/s^2$)'
            init_condition_2 = r'BV speed ($m/s$)'
        elif scenario in ['Vehicle Encroachment']:
            risk_level_bound = [-5.05164471455172, -2.450781893794381, -1.534516401802792, -0.07866796019828293]
            init_condition_1_ind = statistical_key.index('dis')
            init_condition_2_ind = statistical_key.index('angle')
            init_condition_1 = r'Distance ($m$)'
            init_condition_2 = r'Angle ($^{\circ}$)'
        elif scenario in ['Traffic Signal']:
            risk_level_bound = [-5.05164471455172, -2.450781893794381, -1.534516401802792, -0.07866796019828293]
            init_condition_1_ind = statistical_key.index('thw')
            init_condition_1 = r'THW ($s$)'
        for BV_scenario_list in self.BV_scenarios:
            if scenario in BV_scenario_list:
                AV_conflict_ind = self.AV_conflict_ind[scenario]
                AV_seg_len = self.AV_route_seg_len[scenario]
                ind = self.BV_scenarios.index(BV_scenario_list)
                challenger_route = self.BV_routes[ind][scenario]
                challenger_conflict_ind = self.BV_conflict_ind[ind][scenario]
                challenger_seg_len = self.BV_route_seg_len[ind][scenario]
                challenger_length = self.BV_length[ind][scenario]
                challenger_width = self.BV_width[ind][scenario]
                flag = False
                break
        for VRU_scenario_list in self.VRU_scenarios:
            if scenario in VRU_scenario_list:
                ind = self.VRU_scenarios.index(VRU_scenario_list)
                challenger_length = self.VRU_length[ind][scenario]
                challenger_width = self.VRU_width[ind][scenario]
                flag = False
                break
        if flag:
            return {}
        reaction_time_acc_set = {}
        reaction_time_set = {}
        reaction_dis_set = {}
        reaction_ttc_set = {}
        reaction_AV_position_set = {}
        reaction_challenger_position_set = {}
        reaction_timing_set = {}
        reaction_time_list = []
        reaction_dis_list = []
        reaction_flag_set = {}
        init_condition_1_set = {}
        init_condition_2_set = {}
        AV_route = self.AV_route[scenario]
        for statistic, track in zip(statistical_data, track_data):
            risk_level = statistic[risk_level_ind]
            if risk_level not in list(reaction_time_acc_set.keys()):
                reaction_time_acc_set[risk_level] = []
            if risk_level not in list(reaction_time_set.keys()):
                reaction_time_set[risk_level] = []
            if risk_level not in list(reaction_dis_set.keys()):
                reaction_dis_set[risk_level] = []
            if risk_level not in list(reaction_ttc_set.keys()):
                reaction_ttc_set[risk_level] = []
            if risk_level not in list(reaction_AV_position_set.keys()):
                reaction_AV_position_set[risk_level] = [[], []]
            if risk_level not in list(reaction_challenger_position_set.keys()):
                reaction_challenger_position_set[risk_level] = [[], []]
            if risk_level not in list(reaction_timing_set.keys()):
                reaction_timing_set[risk_level] = []
            if risk_level not in list(reaction_flag_set.keys()):
                reaction_flag_set[risk_level] = []
            if risk_level not in list(init_condition_1_set.keys()):
                init_condition_1_set[risk_level] = []
            if risk_level not in list(init_condition_2_set.keys()):
                init_condition_2_set[risk_level] = []
            if statistic[initial_timestamp_ind] == 0:
                reaction_time_set[risk_level].append(math.inf)
                reaction_dis_set[risk_level].append(math.inf)
                reaction_ttc_set[risk_level].append(copy.deepcopy(math.inf))
                reaction_AV_position_set[risk_level][0].append(0)
                reaction_AV_position_set[risk_level][1].append(0)
                reaction_challenger_position_set[risk_level][0].append(0)
                reaction_challenger_position_set[risk_level][1].append(0)
                reaction_timing_set[risk_level].append(0)
                reaction_time_list.append(math.inf)
                reaction_dis_list.append(math.inf)
                reaction_flag_set[risk_level].append(False)
                continue
            if scenario in ['Traffic Signal']:
                init_condition_1_set[risk_level].append(statistic[init_condition_1_ind])
            else:
                init_condition_1_set[risk_level].append(statistic[init_condition_1_ind])
                init_condition_2_set[risk_level].append(statistic[init_condition_2_ind])
            if scenario == 'Car Following':
                init_challenger_sp_ind = statistical_key.index('sp')
                initial_challenger_dec_ind = statistical_key.index('dec')
                challenger_acc_start_flag = False
                av_acc_start_flag = False
                challenger_const_sp_flag = False
                challenger_dec_start_flag = False
                acc_start_timestamp = 0
                start_timestamp = 0
                init_av_sp = 0
                for data in track:
                    flag = False
                    if not challenger_acc_start_flag and data[challenger_acceleration_ind] > 0 and data[challenger_sp_ind] < .5 and data[av_sp_ind] < 1:
                        challenger_acc_start_flag = True
                        acc_start_timestamp = data[timestamp_ind]
                    if challenger_acc_start_flag and not av_acc_start_flag and data[av_lon_acceleration_ind] > 0:
                        av_acc_start_flag = True
                        reaction_time_acc_set[risk_level].append(data[timestamp_ind] - acc_start_timestamp)
                    if av_acc_start_flag and not challenger_dec_start_flag and abs(data[challenger_sp_ind] - statistic[init_challenger_sp_ind]) < .1 and abs(data[challenger_acceleration_ind]) < 0.1:
                        challenger_const_sp_flag = True
                    if challenger_const_sp_flag and not challenger_dec_start_flag and data[challenger_acceleration_ind] < -.1:
                        challenger_dec_start_flag = True
                        start_timestamp = data[timestamp_ind]
                        init_av_sp = data[av_sp_ind]
                    if challenger_dec_start_flag:
                        if data[av_lon_acceleration_ind] < -.1 and data[av_sp_ind] < init_av_sp - .1:
                            av_rectangle = [data[av_x_ind], data[av_y_ind], self.AV_length, self.AV_width, (90 - data[av_heading_ind]) / 180 * math.pi]
                            bv_rectangle = [data[challenger_x_ind], data[challenger_y_ind], challenger_length, challenger_width, (90 - data[challenger_heading_ind]) / 180 * math.pi]
                            dis = cal_min_dis_of_rectangles(av_rectangle, bv_rectangle)
                            relative_sp = data[av_sp_ind] - data[challenger_sp_ind] * math.cos((data[challenger_heading_ind] - data[av_heading_ind]) / 180 * math.pi)
                            if relative_sp > 0:
                                rotated_challenger_position = rotate_point2_around_point1([data[av_x_ind], data[av_y_ind]], [data[challenger_x_ind], data[challenger_y_ind]], (90 - data[av_heading_ind]) / 180 * math.pi)
                                rotated_challenger_heading = (data[av_heading_ind] - data[challenger_heading_ind]) / 180 * math.pi
                                ttc = (rotated_challenger_position[0] - data[av_x_ind] - self.AV_length / 2 - challenger_length / 2 * abs(math.cos(rotated_challenger_heading)) - challenger_width / 2 * abs(math.sin(rotated_challenger_heading))) / relative_sp
                            else:
                                ttc = math.inf
                            reaction_time_set[risk_level].append(data[timestamp_ind] - start_timestamp)
                            reaction_dis_set[risk_level].append(dis)
                            reaction_ttc_set[risk_level].append(copy.deepcopy(ttc))
                            reaction_AV_position_set[risk_level][0].append(data[av_x_ind])
                            reaction_AV_position_set[risk_level][1].append(data[av_y_ind])
                            reaction_challenger_position_set[risk_level][0].append(data[challenger_x_ind])
                            reaction_challenger_position_set[risk_level][1].append(data[challenger_y_ind])
                            reaction_timing_set[risk_level].append(data[timestamp_ind])
                            reaction_time_list.append(data[timestamp_ind] - start_timestamp)
                            reaction_dis_list.append(dis)
                            reaction_flag_set[risk_level].append(True)
                            flag = True
                            break
                if not flag:
                    print('AV does not react to challenger!')
                    reaction_time_set[risk_level].append(math.inf)
                    reaction_dis_set[risk_level].append(math.inf)
                    reaction_ttc_set[risk_level].append(copy.deepcopy(math.inf))
                    reaction_AV_position_set[risk_level][0].append(0)
                    reaction_AV_position_set[risk_level][1].append(0)
                    reaction_challenger_position_set[risk_level][0].append(0)
                    reaction_challenger_position_set[risk_level][1].append(0)
                    reaction_timing_set[risk_level].append(0)
                    reaction_time_list.append(math.inf)
                    reaction_dis_list.append(math.inf)
                    reaction_flag_set[risk_level].append(False)
                tmp = 0
            else:
                ttc = math.inf
                flag = True
                pass_yield_flag = 'yield'
                if scenario in ['Left Turn (AV goes straight)', 'Left Turn (AV turns left)', 'Right Turn (AV goes straight)', 'Right Turn (AV turns right)', 'BV Merging into the Roundabout', 'Roundaout (outside)']:
                    AV_conflict_ind = self.settings.scenario_info[scenario]['AV']['conflict ind'] - self.settings.scenario_info[scenario]['AV']['start ind']
                    challenger_conflict_ind = self.settings.scenario_info[scenario]['challenger']['conflict ind'] - self.settings.scenario_info[scenario]['challenger']['start ind']
                    AV_route = self.AV_route[scenario]
                    for routes in self.BV_routes:
                        if scenario in routes.keys():
                            challenger_route = routes[scenario]
                            break
                    AV_NNPN = 0
                    challenger_NNPN = 0
                    for data in track:
                        if data[timestamp_ind] >= statistic[initial_timestamp_ind]:
                            AV_NNPN = find_next_route_points(AV_route, [data[av_x_ind], data[av_y_ind]], AV_NNPN)
                            challenger_NNPN = find_next_route_points(challenger_route, [data[challenger_x_ind], data[challenger_y_ind]], challenger_NNPN)
                            if AV_NNPN >= AV_conflict_ind and challenger_NNPN < challenger_conflict_ind:
                                pass_yield_flag = 'pass'
                                break
                            elif AV_NNPN < AV_conflict_ind and challenger_NNPN >= challenger_conflict_ind:
                                pass_yield_flag = 'yield'
                                break
                reaction_flag = False
                if pass_yield_flag == 'yield':
                    if scenario in ['Vehicle Encroachment', 'Trafic Signal']:
                        av_baseline_sp = max([line[av_sp_ind] for line in track])
                        start_flag = False
                    else:
                        av_baseline_sp = statistic[initial_av_sp_ind]
                        start_flag = True
                    for data in track:
                        if scenario in ['Vehicle Encroachment', 'Trafic Signal'] and data[av_sp_ind] == av_baseline_sp and not start_flag:
                            start_flag = True
                        if start_flag and data[timestamp_ind] >= statistic[initial_timestamp_ind] and data[av_acceleration_ind] < -0.1 and data[av_sp_ind] < av_baseline_sp - 0.1:
                            # judgement is not correct in some cases
                            if self.settings.test_name == 'Autoware.universe_sim' and scenario == 'Right Turn (AV turns right)' and track_data.index(track) in [26, 27]:
                                if data[av_lon_acceleration_ind] > -1:
                                    continue
                            else:
                                tmp_ind = track.index(data)
                                if scenario not in ['Vehicle Encroachment']:
                                    if abs(track[max(tmp_ind - 1, 0)][av_lon_acceleration_ind] - data[av_lon_acceleration_ind]) > 1 or abs(data[av_lon_acceleration_ind] - track[min(tmp_ind + 1, len(track) - 1)][av_lon_acceleration_ind]) > 1:
                                        continue
                                    if abs(track[max(tmp_ind - 1, 0)][av_acceleration_ind] - data[av_acceleration_ind]) > 1 or abs(data[av_acceleration_ind] - track[min(tmp_ind + 1, len(track) - 1)][av_acceleration_ind]) > 1:
                                        continue
                            av_rectangle = [data[av_x_ind], data[av_y_ind], self.AV_length, self.AV_width, (90 - data[av_heading_ind]) / 180 * math.pi]
                            bv_rectangle = [data[challenger_x_ind], data[challenger_y_ind], challenger_length, challenger_width, (90 - data[challenger_heading_ind]) / 180 * math.pi]
                            dis = cal_min_dis_of_rectangles(av_rectangle, bv_rectangle)
                            if scenario in self.same_route_scenarios:
                                relative_sp = data[av_sp_ind] - data[challenger_sp_ind] * math.cos((data[challenger_heading_ind] - data[av_heading_ind]) / 180 * math.pi)
                                if relative_sp > 0:
                                    rotated_challenger_position = rotate_point2_around_point1([data[av_x_ind], data[av_y_ind]], [data[challenger_x_ind], data[challenger_y_ind]], (90 - data[av_heading_ind]) / 180 * math.pi)
                                    rotated_challenger_heading = (data[av_heading_ind] - data[challenger_heading_ind]) / 180 * math.pi
                                    ttc = (rotated_challenger_position[0] - data[av_x_ind] - self.AV_length / 2 - challenger_length / 2 * abs(math.cos(rotated_challenger_heading)) - challenger_width / 2 * abs(math.sin(rotated_challenger_heading))) / relative_sp
                                else:
                                    ttc = math.inf
                            else:
                                AV_rectangle = [data[av_x_ind], data[av_y_ind], 5, 2, (90 - data[av_heading_ind]) / 180 * math.pi]
                                challenger_rectangle = [data[challenger_x_ind], data[challenger_y_ind], 5, 2, (90 - data[challenger_heading_ind]) / 180 * math.pi]
                                if data[av_sp_ind] > 0 and data[challenger_sp_ind] > 0:
                                    _, ttc = cal_TTC_with_route(AV_rectangle, data[av_sp_ind], AV_route, AV_conflict_ind, AV_seg_len, challenger_rectangle, data[challenger_sp_ind], challenger_route, challenger_conflict_ind, challenger_seg_len)
                            if data == track[-1]:
                                print('AV does not react to challenger!')
                            else:
                                reaction_flag = True
                                reaction_time_set[risk_level].append(data[timestamp_ind] - statistic[initial_timestamp_ind])
                                reaction_dis_set[risk_level].append(dis)
                                reaction_ttc_set[risk_level].append(copy.deepcopy(ttc))
                                reaction_AV_position_set[risk_level][0].append(data[av_x_ind])
                                reaction_AV_position_set[risk_level][1].append(data[av_y_ind])
                                reaction_challenger_position_set[risk_level][0].append(data[challenger_x_ind])
                                reaction_challenger_position_set[risk_level][1].append(data[challenger_y_ind])
                                reaction_timing_set[risk_level].append(data[timestamp_ind])
                                reaction_time_list.append(data[timestamp_ind] - statistic[initial_timestamp_ind])
                                reaction_dis_list.append(dis)
                                flag = False
                            break
                elif pass_yield_flag == 'pass':
                    pass
                if flag:
                    reaction_time_set[risk_level].append(math.inf)
                    reaction_dis_set[risk_level].append(math.inf)
                    reaction_ttc_set[risk_level].append(copy.deepcopy(math.inf))
                    reaction_AV_position_set[risk_level][0].append(0)
                    reaction_AV_position_set[risk_level][1].append(0)
                    reaction_challenger_position_set[risk_level][0].append(0)
                    reaction_challenger_position_set[risk_level][1].append(0)
                    reaction_timing_set[risk_level].append(0)
                    reaction_time_list.append(math.inf)
                    reaction_dis_list.append(math.inf)

                reaction_flag_set[risk_level].append(reaction_flag)

        return reaction_time_set, reaction_dis_set, reaction_timing_set

    def _analyze_following_distance(self, scenario, statistical_key, statistical_data, track_key, track_data, final_time, path):
        risk_level_ind = statistical_key.index('risk level')
        init_challenger_sp_ind = statistical_key.index('sp')
        initial_timestamp_ind = statistical_key.index('init timestamp')
        timestamp_ind = track_key.index('timestamp')
        av_x_ind = track_key.index('AV x')
        av_y_ind = track_key.index('AV y')
        av_sp_ind = track_key.index('AV sp')
        av_acc_ind = track_key.index('AV lon acc')
        challenger_x_ind = track_key.index('challenger x')
        challenger_y_ind = track_key.index('challenger y')
        challenger_sp_ind = track_key.index('challenger sp')
        challenger_acc_ind = track_key.index('challenger acc')

        following_dis_set = {}
        following_sp_set = {}
        for statistic, track, end_t in zip(statistical_data, track_data, final_time):
            risk_level = statistic[risk_level_ind]
            if risk_level not in list(following_dis_set.keys()):
                following_dis_set[risk_level] = []
            if risk_level not in list(following_sp_set.keys()):
                following_sp_set[risk_level] = []
            dis = []
            for data in track:
                if end_t >= data[timestamp_ind] >= statistic[initial_timestamp_ind] and \
                    abs(statistic[init_challenger_sp_ind] - data[challenger_sp_ind]) <= .1 and \
                    abs(data[av_acc_ind]) < .5 and \
                    abs(data[challenger_acc_ind]) < .1:
                    # abs(data[av_sp_ind] - data[challenger_sp_ind]) <= .5 and \
                    dis.append(cal_dis([data[av_x_ind], data[av_y_ind]], [data[challenger_x_ind], data[challenger_y_ind]]))
            if len(dis) > 0:
                following_dis_set[risk_level].append(sum(dis) / len(dis))
                following_sp_set[risk_level].append(statistic[init_challenger_sp_ind])
            else:
                print('No valid following distance data! Please check the data!')

        return

    def _find_final_time(self, scenario, statistical_key, statistical_data, track_key, track_data):
        initial_timestamp_ind = statistical_key.index('init timestamp')
        initial_av_sp_ind = statistical_key.index('AV init sp')
        timestamp_ind = track_key.index('timestamp')
        av_x_ind = track_key.index('AV x')
        av_y_ind = track_key.index('AV y')
        av_sp_ind = track_key.index('AV sp')
        av_acc_ind = track_key.index('AV lon acc')
        av_heading_ind = track_key.index('AV heading')
        if len(self.BV_scenarios) + len(self.VRU_scenarios) > 0:
            challenger_x_ind = track_key.index('challenger x')
            challenger_y_ind = track_key.index('challenger y')
            challenger_sp_ind = track_key.index('challenger sp')
            challenger_acc_ind = track_key.index('challenger acc')
        final_time = []
        duration = []
        if scenario == 'Cut-In':
            initial_challenger_sp_ind = statistical_key.index('relative sp')
            for statistic, track in zip(statistical_data, track_data):
                initial_BV_speed = statistic[initial_av_sp_ind] - statistic[initial_challenger_sp_ind]
                flag = True
                for data in track:
                    if data[timestamp_ind] > statistic[initial_timestamp_ind] and abs(data[challenger_acc_ind]) > 0.5:
                        final_time.append(data[timestamp_ind])
                        flag = False
                        break
                if flag:
                    ind = statistical_data.index(statistic)
                    final_time.append(track[-2][0])
                duration.append(final_time[-1] - statistic[initial_timestamp_ind])
            return final_time, duration
        elif scenario == 'Car Following':
            initial_challenger_dec_ind = statistical_key.index('dec')
            for statistic, track in zip(statistical_data, track_data):
                initial_BV_dec = statistic[initial_challenger_dec_ind]
                flag = True
                start_flag = False
                dec_flag = False
                for data in track:
                    if not start_flag and data[av_sp_ind] < 0.1:
                        start_flag = True
                        statistic[initial_timestamp_ind] = data[timestamp_ind]
                    if start_flag and data[timestamp_ind] > statistic[initial_timestamp_ind]:
                        if abs(data[challenger_acc_ind] - initial_BV_dec) < .1:
                            dec_flag = True
                        if dec_flag and data[challenger_acc_ind] > 0:
                            final_time.append(data[timestamp_ind])
                            flag = False
                            break
                if flag:
                    final_time.append(track[-2][0])
                duration.append(final_time[-1] - statistic[initial_timestamp_ind])
            return final_time, duration
        elif scenario == 'VRU Crossing the Street without Crosswalk':
            initial_challenger_sp_ind = statistical_key.index('crossing sp')
            for statistic, track in zip(statistical_data, track_data):
                initial_VRU_speed = statistic[initial_challenger_sp_ind]
                flag = True
                for data in track:
                    if data[timestamp_ind] - statistic[initial_timestamp_ind] > 3 and abs(data[challenger_sp_ind] - initial_VRU_speed) > 0.1:
                        final_time.append(data[timestamp_ind])
                        flag = False
                        break
                if flag:
                    ind = statistical_data.index(statistic)
                    final_time.append(track[-2][0])
                duration.append(final_time[-1] - statistic[initial_timestamp_ind])
            return final_time, duration
        elif scenario == 'VRU Crossing the Street at the Crosswalk':
            initial_challenger_sp_ind = statistical_key.index('crossing sp')
            for statistic, track in zip(statistical_data, track_data):
                initial_VRU_speed = statistic[initial_challenger_sp_ind]
                flag = True
                for data in track:
                    if data[timestamp_ind] - statistic[initial_timestamp_ind] > 3 and abs(data[challenger_sp_ind] - initial_VRU_speed) > 0.1:
                        final_time.append(data[timestamp_ind])
                        flag = False
                        break
                if flag:
                    ind = statistical_data.index(statistic)
                    final_time.append(track[-2][0])
                duration.append(final_time[-1] - statistic[initial_timestamp_ind])
            return final_time, duration
        elif scenario == 'Lane Departure (same direction)':
            initial_challenger_sp_ind = statistical_key.index('relative sp')
            for statistic, track in zip(statistical_data, track_data):
                initial_BV_speed = statistic[initial_av_sp_ind] - statistic[initial_challenger_sp_ind]
                flag = True
                for data in track:
                    if data[timestamp_ind] > statistic[initial_timestamp_ind] and abs(data[challenger_acc_ind]) > 0.5:
                        final_time.append(data[timestamp_ind])
                        flag = False
                        break
                if flag:
                    ind = statistical_data.index(statistic)
                    final_time.append(track[-2][0])
                duration.append(final_time[-1] - statistic[initial_timestamp_ind])
            return final_time, duration
        elif scenario == 'Lane Departure (opposite direction)':
            initial_challenger_sp_ind = statistical_key.index('relative sp')
            for statistic, track in zip(statistical_data, track_data):
                initial_BV_speed = statistic[initial_challenger_sp_ind] - statistic[initial_av_sp_ind]
                flag = True
                for data in track:
                    if data[timestamp_ind] > statistic[initial_timestamp_ind] and abs(data[challenger_acc_ind]) > 0.5:
                        final_time.append(data[timestamp_ind])
                        flag = False
                        break
                if flag:
                    ind = statistical_data.index(statistic)
                    final_time.append(track[-2][0])
                duration.append(final_time[-1] - statistic[initial_timestamp_ind])
            return final_time, duration
        elif scenario == 'Left Turn (AV goes straight)':
            initial_challenger_sp_ind = statistical_key.index('sp')
            for statistic, track in zip(statistical_data, track_data):
                initial_BV_speed = statistic[initial_challenger_sp_ind]
                flag = True
                for data in track:
                    if data[timestamp_ind] > statistic[initial_timestamp_ind] and (abs(data[challenger_sp_ind] - initial_BV_speed) > 0.1 or data[challenger_y_ind] > 240):
                        final_time.append(data[timestamp_ind])
                        flag = False
                        break
                if flag:
                    ind = statistical_data.index(statistic)
                    final_time.append(track[-2][0])
                duration.append(final_time[-1] - statistic[initial_timestamp_ind])
            return final_time, duration
        elif scenario == 'Left Turn (AV turns left)':
            initial_challenger_sp_ind = statistical_key.index('sp')
            for statistic, track in zip(statistical_data, track_data):
                initial_BV_speed = statistic[initial_challenger_sp_ind]
                flag = True
                for data in track:
                    if data[timestamp_ind] > statistic[initial_timestamp_ind] and (abs(data[challenger_sp_ind] - initial_BV_speed) > 0.1 or data[challenger_y_ind] > 240):
                        final_time.append(data[timestamp_ind])
                        flag = False
                        break
                if flag:
                    ind = statistical_data.index(statistic)
                    final_time.append(track[-2][0])
                duration.append(final_time[-1] - statistic[initial_timestamp_ind])
            return final_time, duration
        elif scenario == 'Right Turn (AV goes straight)':
            initial_challenger_sp_ind = statistical_key.index('sp')
            for statistic, track in zip(statistical_data, track_data):
                initial_BV_speed = statistic[initial_challenger_sp_ind]
                flag = True
                for data in track:
                    if data[timestamp_ind] > statistic[initial_timestamp_ind] and (abs(data[challenger_sp_ind] - initial_BV_speed) > 0.1 or data[challenger_y_ind] > 240):
                        final_time.append(data[timestamp_ind])
                        flag = False
                        break
                if flag:
                    ind = statistical_data.index(statistic)
                    final_time.append(track[-2][0])
                duration.append(final_time[-1] - statistic[initial_timestamp_ind])
            return final_time, duration
        elif scenario == 'Right Turn (AV turns right)':
            initial_challenger_sp_ind = statistical_key.index('sp')
            for statistic, track in zip(statistical_data, track_data):
                initial_BV_speed = statistic[initial_challenger_sp_ind]
                flag = True
                for data in track:
                    if data[timestamp_ind] > statistic[initial_timestamp_ind] and data[challenger_y_ind] > 210:
                        final_time.append(data[timestamp_ind])
                        flag = False
                        break
                if flag:
                    ind = statistical_data.index(statistic)
                    final_time.append(track[-2][0])
                duration.append(final_time[-1] - statistic[initial_timestamp_ind])
            return final_time, duration
        elif scenario == 'AV Merging into the Roundabout':
            initial_challenger_sp_ind = statistical_key.index('sp')
            for statistic, track in zip(statistical_data, track_data):
                initial_BV_speed = statistic[initial_challenger_sp_ind]
                flag = True
                for data in track:
                    if data[timestamp_ind] - statistic[initial_timestamp_ind] > 2 and ((abs(data[challenger_sp_ind] - initial_BV_speed) > 0.1 and data[challenger_sp_ind] != 0) or data[challenger_y_ind] > 240):
                        final_time.append(data[timestamp_ind])
                        flag = False
                        break
                if flag:
                    ind = statistical_data.index(statistic)
                    final_time.append(track[-2][0])
                duration.append(final_time[-1] - statistic[initial_timestamp_ind])
            return final_time, duration
        elif scenario == 'BV Merging into the Roundabout':
            initial_challenger_sp_ind = statistical_key.index('sp')
            for statistic, track in zip(statistical_data, track_data):
                initial_BV_speed = statistic[initial_challenger_sp_ind]
                flag = True
                for data in track:
                    if data[timestamp_ind] > statistic[initial_timestamp_ind] and (abs(data[challenger_sp_ind] - initial_BV_speed) > 0.1 or data[challenger_y_ind] > 240):
                        final_time.append(data[timestamp_ind])
                        flag = False
                        break
                if flag:
                    ind = statistical_data.index(statistic)
                    final_time.append(track[-2][0])
            return final_time, duration
        elif scenario == 'Vehicle Encroachment':
            for statistic, track in zip(statistical_data, track_data):
                flag = True
                start_flag = False
                for data in track:
                    if not start_flag and data[av_sp_ind] > 2:
                        start_flag = True
                        statistic[initial_timestamp_ind] = data[timestamp_ind]
                    if start_flag and data[timestamp_ind] - statistic[initial_timestamp_ind] > 2 and (data[av_sp_ind] < 0.2 or data[av_y_ind] > data[challenger_y_ind]):
                        final_time.append(data[timestamp_ind])
                        flag = False
                        break
                if flag:
                    ind = statistical_data.index(statistic)
                    final_time.append(track[-2][0])
                duration.append(final_time[-1] - statistic[initial_timestamp_ind])
            return final_time, duration
        elif scenario == 'Traffic Signal':
            for statistic, track in zip(statistical_data, track_data):
                flag = True
                if flag:
                    ind = statistical_data.index(statistic)
                    final_time.append(track[-2][0])
                duration.append(final_time[-1] - statistic[initial_timestamp_ind])
            return final_time, duration
        else:
            return [track[-1][0] for track in range(len(track_data))]

    def _plot_case(self, scenario, statistical_key, statistical_data, track_key, track_data, all_final_time, all_reaction_timing_set):
        initial_timestamp_ind = statistical_key.index('init timestamp')
        av_sp_ind = track_key.index('AV sp')
        av_lon_acceleration_ind = track_key.index('AV lon acc')
        av_lat_acceleration_ind = track_key.index('AV lat acc')
        timestamp_ind = track_key.index('timestamp')
        path = self.plot_path + '/' + get_scenario_folder_name(scenario)
        os.makedirs(path, exist_ok=True)

        all_reaction_timing_list = []
        for key in all_reaction_timing_set.keys():
            all_reaction_timing_list += all_reaction_timing_set[key]
        
        key_timestamps = []
        additional_time = 2

        for i in range(len(statistical_data)):
            lines = []
            time = [line[timestamp_ind] for line in track_data[i][:-1] if statistical_data[i][initial_timestamp_ind] - additional_time <= line[timestamp_ind] <= all_final_time[i] + additional_time]
            fig, ax1 = fig_format_setting(self.fontsize, tight_layout=False, constrained_layout=True)
            av_sp = [line[av_sp_ind] for line in track_data[i][:-1] if statistical_data[i][initial_timestamp_ind] - additional_time <= line[timestamp_ind] <= all_final_time[i] + additional_time]
            line1, = ax1.plot(time, av_sp, color='b', label='AV sp.')
            lines.append(line1)
            ax1.set_xlabel(r'Timestamp ($s$)')
            ax1.set_ylabel(r'AV speed ($m/s$)')

            ax2 = ax1.twinx()
            av_acc = [line[av_lon_acceleration_ind] for line in track_data[i][:-1] if statistical_data[i][initial_timestamp_ind] - additional_time <= line[timestamp_ind] <= all_final_time[i] + additional_time]
            line2, = ax2.plot(time, av_acc, color='g', label='AV long. acc.')
            lines.append(line2)
            if scenario in ['Lane Departure (opposite direction)']:
                av_lat_acc = [line[av_lat_acceleration_ind] for line in track_data[i][:-1] if statistical_data[i][initial_timestamp_ind] - additional_time <= line[timestamp_ind] <= all_final_time[i] + additional_time]
                line2_2, = ax2.plot(time, av_lat_acc, color='c', label='AV lat. acc.')
                lines.append(line2_2)
            line3 = ax2.axvline(x=statistical_data[i][initial_timestamp_ind], color='r', linestyle='--', label='Start ts.')
            lines.append(line3)
            if len(all_reaction_timing_list) > 0:
                if all_reaction_timing_list[i] != 0:
                    line4 = ax2.axvline(x=all_reaction_timing_list[i], color='y', linestyle='--', label='Reaction ts.')
                    lines.append(line4)
            if track_data[i][-1][timestamp_ind] == 0:
                line5 = ax2.axvline(x=all_final_time[i], color='m', linestyle='--', label='Collision ts.')
            else:
                line5 = ax2.axvline(x=all_final_time[i], color='m', linestyle='--', label='Final ts.')
            lines.append(line5)
            ylim = plt.gca().get_ylim()
            ylim = [max(ylim[0], -10), min(ylim[1], 10)]
            ax2.set_ylim(ylim)
            ax2.set_ylabel(r'AV acceleration ($m/s^2$)')
            ax1.legend(lines, [l.get_label() for l in lines], fontsize=18)

            ylim = ax2.get_ylim()
            if ylim[1] - ylim[0] < 0.2:
                c = sum(ylim) / 2
                c = round(c, 1)
                ax2.set_ylim([c - 0.1, c + 0.1])

            plt.savefig(path + '/' + str(i + 1) + '.svg', format='svg', bbox_inches='tight')
            plt.close()

            key_timestamps.append([i+1, statistical_data[i][initial_timestamp_ind], all_reaction_timing_list[i] if len(all_reaction_timing_list) > 0 else 0, all_final_time[i]])

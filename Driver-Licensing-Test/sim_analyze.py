import json
import os
from functools import reduce
import matplotlib.pyplot as plt

from env.result_analysis.analysis import Analysis
from settings import Settings
from utils import save_csv, read_csv, fig_format_setting, get_scenario_folder_name


def get_overall_results(overall_metrics, overall_passed_num, overall_failed_num,
                          overall_initial_dis_err, overall_initial_sp_err, overall_av_init_sp):
    overall_results = {
        'overall_metrics': overall_metrics,
        'overall_passed_num': overall_passed_num,
        'overall_failed_num': overall_failed_num,
        'overall_initial_dis_err': overall_initial_dis_err,
        'overall_initial_sp_err': overall_initial_sp_err,
        'overall_av_init_sp': overall_av_init_sp
    }
    return overall_results


if __name__ == '__main__':
    test_name = 'Autoware.Universe'
    total_round_num = 3

    overall_metrics = []
    overall_passed_cases = {}
    overall_failed_cases = {}
    overall_initial_dis_err = {}
    overall_initial_sp_err = {}
    overall_av_init_sp = {}

    settings = Settings(test_name, 1, 'cut_in')
    overall_evaluation_path = os.path.join(settings.root, 'output', test_name, 'evaluation', 'overall')
    os.makedirs(overall_evaluation_path, exist_ok=True)

    keys = ['scenario', 'passed_num', 'failed_num', 'avg_min_dis', 'avg_min_ttc', 'avg_reaction_time', 'max_reaction_time', 'avg_reaction_dis', 'min_reaction_dis']
    for round_num in range(total_round_num):
        settings = Settings(test_name, round_num, 'cut_in')
        if os.path.exists(settings.evaluation_path + '/metrics.csv'):
            keys, metrics = read_csv(settings.evaluation_path + '/metrics.csv')
            with open(settings.evaluation_path + '/passed_cases.json', 'r') as f:
                passed_cases = json.load(f)
            with open(settings.evaluation_path + '/failed_cases.json', 'r') as f:
                failed_cases = json.load(f)
            with open(settings.evaluation_path + '/initial_dis_err.json', 'r') as f:
                initial_dis_err = json.load(f)
            with open(settings.evaluation_path + '/initial_sp_err.json', 'r') as f:
                initial_sp_err = json.load(f)
            with open(settings.evaluation_path + '/av_init_sp.json', 'r') as f:
                av_init_sp = json.load(f)
        else:
            passed_num = {}
            failed_num = {}
            passed_cases = {}
            failed_cases = {}
            avg_min_dis = {}
            avg_min_ttc = {}
            avg_reaction_time = {}
            max_reaction_time = {}
            avg_reaction_dis = {}
            min_reaction_dis = {}
            all_min_dis = {}
            all_min_ttc = {}
            all_reaction_time = {}
            all_reaction_dis = {}
            initial_dis_err = {}
            initial_sp_err = {}
            av_init_sp = {}
            metrics = []
            for folder in ['cut_in', 'car_following', 'lane_departure_same', 'lane_departure_opposite', 'left_turn_straight', 'left_turn_turn',
                           'right_turn_straight', 'right_turn_turn', 'vru_without_crosswalk', 'vru_at_crosswalk',
                           'roundabout_av_inside', 'roundabout_av_outside', 'vehicle_encroachment', 'traffic_signal']:
                settings = Settings(test_name, round_num, folder)
                analyzer = Analysis(settings)
                passed_num_set, failed_num_set, passed_cases_set, failed_cases_set, avg_min_dis_set, \
                    avg_min_ttc_set, avg_reaction_time_set, max_reaction_time_set, \
                    avg_reaction_dis_set, min_reaction_dis_set, all_min_dis_set, \
                    all_min_ttc_set, all_reaction_time_set, all_reaction_dis_set, \
                    initial_dis_err_set, initial_sp_err_set, av_init_sp_set = analyzer.analyze_results()

                result = {scenario: [scenario] for scenario in passed_num_set.keys()}
                for scenario, num in passed_num_set.items():
                    passed_num[scenario] = num
                    result[scenario].append(num['all'])
                for scenario, num in failed_num_set.items():
                    failed_num[scenario] = num
                    result[scenario].append(num['all'])
                for scenario, dis in avg_min_dis_set.items():
                    avg_min_dis[scenario] = dis
                    result[scenario].append(dis)
                for scenario, ttc in avg_min_ttc_set.items():
                    avg_min_ttc[scenario] = ttc
                    result[scenario].append(ttc)
                for scenario, t in avg_reaction_time_set.items():
                    avg_reaction_time[scenario] = t
                    result[scenario].append(t)
                for scenario, t in max_reaction_time_set.items():
                    max_reaction_time[scenario] = t
                    result[scenario].append(t)
                for scenario, d in avg_reaction_dis_set.items():
                    avg_reaction_dis[scenario] = d
                    result[scenario].append(d)
                for scenario, d in min_reaction_dis_set.items():
                    min_reaction_dis[scenario] = d
                    result[scenario].append(d)

                for scenario, dis in all_min_dis_set.items():
                    all_min_dis[scenario] = dis
                for scenario, ttc in all_min_ttc_set.items():
                    all_min_ttc[scenario] = ttc
                for scenario, t in all_reaction_time_set.items():
                    all_reaction_time[scenario] = t
                for scenario, d in all_reaction_dis_set.items():
                    all_reaction_dis[scenario] = d
                for scenario, cases in passed_cases_set.items():
                    passed_cases[scenario] = cases
                for scenario, cases in failed_cases_set.items():
                    failed_cases[scenario] = cases
                for scenario, d in initial_dis_err_set.items():
                    initial_dis_err[scenario] = d
                for scenario, sp in initial_sp_err_set.items():
                    initial_sp_err[scenario] = sp
                for scenario, sp in av_init_sp_set.items():
                    av_init_sp[scenario] = sp

                for value in result.values():
                    metrics.append(value)
            if len(metrics) > 0:
                save_csv(metrics, keys, settings.evaluation_path, 'metrics', False)

                f = open(settings.evaluation_path + '/all_min_dis.json', 'w')
                f.write(json.dumps(all_min_dis))
                f.close()

                f = open(settings.evaluation_path + '/all_min_ttc.json', 'w')
                f.write(json.dumps(all_min_ttc))
                f.close()

                f = open(settings.evaluation_path + '/all_reaction_time.json', 'w')
                f.write(json.dumps(all_reaction_time))
                f.close()

                f = open(settings.evaluation_path + '/all_reaction_dis.json', 'w')
                f.write(json.dumps(all_reaction_dis))
                f.close()

                f = open(settings.evaluation_path + '/passed_cases.json', 'w')
                f.write(json.dumps(passed_cases))
                f.close()

                f = open(settings.evaluation_path + '/failed_cases.json', 'w')
                f.write(json.dumps(failed_cases))
                f.close()

                f = open(settings.evaluation_path + '/initial_dis_err.json', 'w')
                f.write(json.dumps(initial_dis_err))
                f.close()

                f = open(settings.evaluation_path + '/initial_sp_err.json', 'w')
                f.write(json.dumps(initial_sp_err))
                f.close()

                f = open(settings.evaluation_path + '/av_init_sp.json', 'w')
                f.write(json.dumps(av_init_sp))
                f.close()

        overall_metrics.append(metrics)
        overall_passed_cases[round_num] = passed_cases
        overall_failed_cases[round_num] = failed_cases
        overall_initial_dis_err[round_num] = initial_dis_err
        overall_initial_sp_err[round_num] = initial_sp_err
        overall_av_init_sp[round_num] = av_init_sp

    intersect_passed_cases = {}
    intersect_failed_cases = {}
    final_initial_dis_err = {}
    final_initial_sp_err = {}
    final_av_init_sp = {}
    for scenario in overall_passed_cases[0].keys():
        scenario_passed_cases = {}
        scenario_failed_cases = {}
        scenario_initial_dis_err = {}
        scenario_initial_sp_err = {}
        scenario_av_init_sp = {}
        for round_num in overall_passed_cases.keys():
            for risk_level in overall_passed_cases[round_num][scenario].keys():
                if risk_level not in scenario_passed_cases.keys():
                    scenario_passed_cases[risk_level] = [overall_passed_cases[round_num][scenario][risk_level]]
                else:
                    scenario_passed_cases[risk_level].append(overall_passed_cases[round_num][scenario][risk_level])
            for risk_level in overall_failed_cases[round_num][scenario].keys():
                if risk_level not in scenario_failed_cases.keys():
                    scenario_failed_cases[risk_level] = [overall_failed_cases[round_num][scenario][risk_level]]
                else:
                    scenario_failed_cases[risk_level].append(overall_failed_cases[round_num][scenario][risk_level])
            for risk_level in overall_initial_dis_err[round_num][scenario].keys():
                if risk_level not in scenario_initial_dis_err.keys():
                    scenario_initial_dis_err[risk_level] = overall_initial_dis_err[round_num][scenario][risk_level]
                else:
                    scenario_initial_dis_err[risk_level] += overall_initial_dis_err[round_num][scenario][risk_level]
            for risk_level in overall_initial_sp_err[round_num][scenario].keys():
                if risk_level not in scenario_initial_sp_err.keys():
                    scenario_initial_sp_err[risk_level] = overall_initial_sp_err[round_num][scenario][risk_level]
                else:
                    scenario_initial_sp_err[risk_level] += overall_initial_sp_err[round_num][scenario][risk_level]
            for risk_level in overall_av_init_sp[round_num][scenario].keys():
                if risk_level not in scenario_av_init_sp.keys():
                    scenario_av_init_sp[risk_level] = overall_av_init_sp[round_num][scenario][risk_level]
                else:
                    scenario_av_init_sp[risk_level] += overall_av_init_sp[round_num][scenario][risk_level]
        
        scenario_passed_cases_num = 0
        scenario_failed_cases_num = 0
        for risk_level in scenario_passed_cases.keys():
            passed_cases = reduce(lambda x, y: set(x) & set(y), scenario_passed_cases[risk_level])
            scenario_passed_cases_num += len(passed_cases)
        for risk_level in scenario_failed_cases.keys():
            failed_cases = reduce(lambda x, y: set(x).union(y), scenario_failed_cases[risk_level])
            scenario_failed_cases_num += len(failed_cases)
        intersect_passed_cases[scenario] = scenario_passed_cases_num
        intersect_failed_cases[scenario] = scenario_failed_cases_num

        final_initial_dis_err[scenario] = scenario_initial_dis_err
        final_initial_sp_err[scenario] = scenario_initial_sp_err
        final_av_init_sp[scenario] = scenario_av_init_sp

    final_metrics = []
    for i in range(len(overall_metrics[0])):
        scenario_data = [metrics[i] for metrics in overall_metrics]
        final_scenario_data = [
            scenario_data[0][0],
            list(intersect_passed_cases.values())[i],
            list(intersect_failed_cases.values())[i]
        ]
        final_scenario_data.extend([sum(scenario_data[j][i] for j in range(len(scenario_data))) / len(scenario_data) for i in range(3, len(scenario_data[0]))])
        final_metrics.append(final_scenario_data)
    save_csv(final_metrics, keys, overall_evaluation_path, 'overall_metrics', False)

    for scenario in final_initial_dis_err.keys():
        settings = Settings(test_name, 0, get_scenario_folder_name(scenario))
        path = os.path.join(overall_evaluation_path, get_scenario_folder_name(scenario))
        os.makedirs(path, exist_ok=True)

        # plot initial distance error
        fig, ax = fig_format_setting(settings.fontsize)
        plt.boxplot(list(final_initial_dis_err[scenario].values()), patch_artist=True)
        plt.setp(ax, xticklabels=list(final_initial_dis_err[scenario].keys()))
        ax.set_xlabel(r'Risk level')
        ax.set_ylabel(r'Initial distance error ($m$)')
        plt.savefig(path + '/initial_distance_error.svg', format='svg', bbox_inches='tight')
        plt.close()

        # plot AV initial speed
        fig, ax = fig_format_setting(settings.fontsize)
        plt.boxplot(list(final_av_init_sp[scenario].values()), patch_artist=True)
        plt.setp(ax, xticklabels=list(final_av_init_sp[scenario].keys()))
        xlim = ax.get_xlim()
        plt.hlines(y=settings.AV_speed_list[0], xmin=xlim[0], xmax=xlim[1], colors='r', linestyles='dashed')
        plt.xlim(xlim)
        ax.set_xlabel(r'Risk level')
        ax.set_ylabel(r'AV initial speed ($m/s$)')
        plt.savefig(path + '/av_initial_speed.svg', format='svg', bbox_inches='tight')
        plt.close()

import matplotlib.pyplot as plt
from rich import progress
import matplotlib.animation as animation
from tqdm import tqdm
import numpy as np

from settings import *
from utils import *


class Plot():
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
        self.BV_length = []
        self.BV_width = []
        for scenario_list, length, width in zip(self.BV_scenarios, settings.BV_length, settings.BV_width):
            BV_length = {}
            BV_width = {}
            for scenario in scenario_list:
                BV_length[scenario] = length
                BV_width[scenario] = width
            self.BV_length.append(BV_length)
            self.BV_width.append(BV_width)

        self.VRU_scenarios = settings.VRU_scenarios
        self.VRU_length = []
        self.VRU_width = []
        for scenario_list, length, width in zip(self.VRU_scenarios, settings.VRU_length, settings.VRU_width):
            vru_length = {}
            vru_width = {}
            for scenario in scenario_list:
                vru_length[scenario] = length
                vru_width[scenario] = width
            self.VRU_length.append(vru_length)
            self.VRU_width.append(vru_width)

        self.plot_path = settings.plot_path

    def plot_results(self):
        for scenario in self.scenario_list:
            scenario_folder_name = get_scenario_folder_name(scenario)
            data_path = self.settings.test_data_path + '/' + scenario_folder_name
            if not os.path.exists(data_path):
                continue
            evaluation_path = self.save_path + '/' + scenario_folder_name
            if not os.path.exists(evaluation_path):
                os.mkdir(evaluation_path)
            statistical_key, statistical_data, track_key, track_data, flag = self._load_data(data_path)
            all_final_time, _ = self._find_final_time(scenario, statistical_key, statistical_data, track_key, track_data)
            all_reaction_timing_set = self._analyze_reaction_timing(scenario, statistical_key, statistical_data, track_key, track_data)
            self._plot_case(scenario, statistical_key, statistical_data, track_key, track_data, all_final_time, all_reaction_timing_set)

    def _load_data(self, data_path):
        statistical_key, statistical_data = read_csv(data_path + '/record.csv')
        track_data = []
        files = os.listdir(data_path)
        N = len(files)
        if 'case id' in statistical_key:
            case_id_ind = statistical_key.index('case id')
        else:
            case_id_ind = -1
        for line in progress.track(statistical_data, description='Loading data from ' + data_path):
            if case_id_ind == -1:
                case_id = statistical_data.index(line) + 1
            else:
                case_id = int(line[case_id_ind])
            track_file = data_path + '/' + str(case_id) + '.csv'
            track_key, value = read_csv(track_file)
            track_data.append(value)
        return statistical_key, statistical_data, track_key, track_data, True

    def _analyze_reaction_timing(self, scenario, statistical_key, statistical_data, track_key, track_data):
        risk_level_ind = statistical_key.index('risk level')
        initial_timestamp_ind = statistical_key.index('init timestamp')
        initial_AV_sp_ind = statistical_key.index('AV init sp')
        timestamp_ind = track_key.index('timestamp')
        AV_x_ind = track_key.index('AV x')
        AV_y_ind = track_key.index('AV y')
        AV_sp_ind = track_key.index('AV sp')
        AV_heading_ind = track_key.index('AV heading')
        AV_acceleration_ind = track_key.index('AV acc')
        challenger_x_ind = track_key.index('challenger x')
        challenger_y_ind = track_key.index('challenger y')
        challenger_sp_ind = track_key.index('challenger sp')
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
        reaction_timing_set = {}
        # AV_route = self.AV_route[scenario]
        for statistic, track in zip(statistical_data, track_data):
            risk_level = statistic[risk_level_ind]
            if risk_level not in list(reaction_timing_set.keys()):
                reaction_timing_set[risk_level] = []

            challenger_route_ori = []
            AV_route_ori = []
            for line in track:
                if line[challenger_sp_ind] > 0.5:
                    if len(AV_route_ori) == 0:
                        AV_route_ori.append([line[AV_x_ind], line[AV_y_ind], line[AV_heading_ind] % 360])
                    else:
                        if cal_dis([line[AV_x_ind], line[AV_y_ind]], AV_route_ori[-1][:2]) > 0.5:
                            AV_route_ori.append([line[AV_x_ind], line[AV_y_ind], line[AV_heading_ind] % 360])
                    if len(challenger_route_ori) == 0:
                        challenger_route_ori.append([line[challenger_x_ind], line[challenger_y_ind], line[challenger_heading_ind] % 360])
                    else:
                        if cal_dis([line[challenger_x_ind], line[challenger_y_ind]], challenger_route_ori[-1][:2]) > 0.5:
                            challenger_route_ori.append([line[challenger_x_ind], line[challenger_y_ind], line[challenger_heading_ind] % 360])
            heading = sum(point[-1] for point in AV_route_ori[-5:]) / 5
            for _ in range(60):
                x = AV_route_ori[-1][0] + 0.5 * math.cos((90 - heading) / 180 * math.pi)
                y = AV_route_ori[-1][1] + 0.5 * math.sin((90 - heading) / 180 * math.pi)
                AV_route_ori.append([x, y, heading])
            challenger_route_ori = np.array(challenger_route_ori)
            AV_route_ori = np.array(AV_route_ori)

            challenger_route = np.array(challenger_route_ori)
            challenger_route = challenger_route[::5]

            AV_route = np.array(AV_route_ori)
            AV_route = AV_route[::5]
            AV_conflict_ind = cal_route_conflict_ind(AV_route, self.AV_length, self.AV_width, challenger_route, challenger_length, challenger_width)[0]
            challenger_conflict_ind = cal_route_conflict_ind(challenger_route, challenger_length, challenger_width, AV_route, self.AV_length, self.AV_width)[0]

            pass_yield_flag = 'yield'
            if scenario in ['Left Turn (AV goes straight)']:
                AV_NNPN = 0
                challenger_NNPN = 0
                for data in track:
                    if data[timestamp_ind] >= statistic[initial_timestamp_ind]:
                        AV_NNPN = find_next_route_points(AV_route, [data[AV_x_ind], data[AV_y_ind]], AV_NNPN)
                        challenger_NNPN = find_next_route_points(challenger_route, [data[challenger_x_ind], data[challenger_y_ind]], challenger_NNPN)
                        if AV_NNPN >= AV_conflict_ind and challenger_NNPN < challenger_conflict_ind:
                            pass_yield_flag = 'pass'
                            break
                        elif AV_NNPN < AV_conflict_ind and challenger_NNPN >= challenger_conflict_ind:
                            pass_yield_flag = 'yield'
                            break

            if pass_yield_flag == 'yield':
                for data in track:
                    if data[timestamp_ind] >= statistic[initial_timestamp_ind] and data[AV_acceleration_ind] < -0.1 and data[AV_sp_ind] < statistic[initial_AV_sp_ind] - 0.5:
                        if data[timestamp_ind] - statistic[initial_timestamp_ind] > 5:
                            reaction_timing_set[risk_level].append(math.inf)
                        else:
                            reaction_timing_set[risk_level].append(data[timestamp_ind])
                        break
            else:
                reaction_timing_set[risk_level].append(data[timestamp_ind])

        return reaction_timing_set

    def _find_final_time(self, scenario, statistical_key, statistical_data, track_key, track_data):
        initial_timestamp_ind = statistical_key.index('init timestamp')
        timestamp_ind = track_key.index('timestamp')
        AV_x_ind = track_key.index('AV x')
        challenger_x_ind = track_key.index('challenger x')
        challenger_sp_ind = track_key.index('challenger sp')
        final_time = []
        duration = []
        if scenario == 'VRU Crossing the Street without Crosswalk':
            for statistic, track in zip(statistical_data, track_data):
                flag = True
                for data in track:
                    if data[timestamp_ind] > statistic[initial_timestamp_ind] and (data[challenger_x_ind] > data[AV_x_ind] + self.AV_width / 2 or data[challenger_sp_ind] < 0.2):
                        final_time.append(data[timestamp_ind])
                        flag = False
                        break
                if flag:
                    final_time.append(track[-1][0])
                duration.append(final_time[-1] - statistic[initial_timestamp_ind])
            return final_time, duration
        elif scenario == 'Left Turn (AV goes straight)':
            for statistic, track in zip(statistical_data, track_data):
                flag = True
                for data in track:
                    if data[timestamp_ind] - statistic[initial_timestamp_ind] > 2 and (data[challenger_x_ind] > 130 or data[challenger_sp_ind] < 0.2):
                        final_time.append(data[timestamp_ind])
                        flag = False
                        break
                if flag:
                    final_time.append(track[-1][0])
                duration.append(final_time[-1] - statistic[initial_timestamp_ind])
            return final_time, duration
        else:
            return [track[-1][0] for track in range(len(track_data))]

    def _plot_case(self, scenario, statistical_key, statistical_data, track_key, track_data, all_final_time, all_reaction_timing_set):
        initial_timestamp_ind = statistical_key.index('init timestamp')
        AV_x_ind = track_key.index('AV x')
        AV_y_ind = track_key.index('AV y')
        AV_sp_ind = track_key.index('AV sp')
        AV_lon_acceleration_ind = track_key.index('AV acc')
        AV_lat_acceleration_ind = track_key.index('AV lat acc')
        challenger_x_ind = track_key.index('challenger x')
        challenger_y_ind = track_key.index('challenger y')
        challenger_sp_ind = track_key.index('challenger sp')
        timestamp_ind = track_key.index('timestamp')
        if scenario in ['Jaywalking', 'VRU at Crosswalk']:
            challenger_length_cand = self.VRU_length
        else:
            challenger_length_cand = self.BV_length
        challenger_length = 0
        for challenger_lengths in challenger_length_cand:
            for key in challenger_lengths.keys():
                if key == scenario:
                    challenger_length = challenger_lengths[key]
                    break
            if challenger_length != 0:
                break
        if 'case id' in statistical_key:
            case_id_ind = statistical_key.index('case id')
        else:
            case_id_ind = -1
        path = self.plot_path + '/' + get_scenario_folder_name(scenario)
        if not os.path.exists(path):
            os.mkdir(path)

        all_reaction_timing_list = []
        for key in all_reaction_timing_set.keys():
            all_reaction_timing_list += all_reaction_timing_set[key]
        
        key_timestamps = []

        additional_time = 5  # 5 seconds more before and after the case

        figsize = (10, 10)
        for i in range(len(statistical_data)):
            case_id = int(statistical_data[i][case_id_ind])

            plt.figure(figsize=figsize)
            time = [line[timestamp_ind] for line in track_data[i][:-1] if statistical_data[i][initial_timestamp_ind] - additional_time <= line[timestamp_ind] <= all_final_time[i] + additional_time]
            ylim = {}

            plt.subplot(3, 1, 1)
            long_dis = [line[AV_y_ind] - line[challenger_y_ind] - self.AV_length / 2 - challenger_length / 2 for line in track_data[i][:-1] if statistical_data[i][initial_timestamp_ind] - additional_time <= line[timestamp_ind] <= all_final_time[i] + additional_time]
            plt.plot(time, long_dis, color='b', label='Longitudinal distance')
            plt.axvline(x=statistical_data[i][initial_timestamp_ind], color='r', linestyle='--', label='start timestamp')
            if len(all_reaction_timing_list) > 0:
                if all_reaction_timing_list[i] != 0:
                    plt.axvline(x=all_reaction_timing_list[i], color='y', linestyle='--', label='reaction timestamp')
            if track_data[i][-1][timestamp_ind] == 0:
                plt.axvline(x=all_final_time[i], color='m', linestyle='--', label='collision timestamp')
            else:
                plt.axvline(x=all_final_time[i], color='m', linestyle='--', label='final timestamp')
            plt.ylabel('Longitudinal distance (m)')
            plt.legend()
            ylim[0] = plt.ylim()

            plt.subplot(3, 1, 2)
            AV_sp = [line[AV_sp_ind] for line in track_data[i][:-1] if statistical_data[i][initial_timestamp_ind] - additional_time <= line[timestamp_ind] <= all_final_time[i] + additional_time]
            challenger_sp = [line[challenger_sp_ind] for line in track_data[i][:-1] if statistical_data[i][initial_timestamp_ind] - additional_time <= line[timestamp_ind] <= all_final_time[i] + additional_time]
            plt.plot(time, AV_sp, color='b', label='AV speed')
            plt.plot(time, challenger_sp, color='g', label='Challenger speed')
            plt.axvline(x=statistical_data[i][initial_timestamp_ind], color='r', linestyle='--', label='start timestamp')
            if len(all_reaction_timing_list) > 0:
                if all_reaction_timing_list[i] != 0:
                    plt.axvline(x=all_reaction_timing_list[i], color='y', linestyle='--', label='reaction timestamp')
            if track_data[i][-1][timestamp_ind] == 0:
                plt.axvline(x=all_final_time[i], color='m', linestyle='--', label='collision timestamp')
            else:
                plt.axvline(x=all_final_time[i], color='m', linestyle='--', label='final timestamp')
            plt.ylabel('Speed (m/s)')
            plt.legend()
            ylim[1] = plt.ylim()

            plt.subplot(3, 1, 3)
            if scenario in ['Lane Departure (opposite)']:
                AV_lat_acc = [line[AV_lat_acceleration_ind] for line in track_data[i][:-1] if statistical_data[i][initial_timestamp_ind] - additional_time <= line[timestamp_ind] <= all_final_time[i] + additional_time]
                plt.plot(time, AV_lat_acc, color='g', label='AV lateral acceleration')
                plt.ylabel(r'AV lateral acceleration ($m/s^2$)')
            else:
                AV_lon_acc = [line[AV_lon_acceleration_ind] for line in track_data[i][:-1] if statistical_data[i][initial_timestamp_ind] - additional_time <= line[timestamp_ind] <= all_final_time[i] + additional_time]
                plt.plot(time, AV_lon_acc, color='b', label='AV longitudinal acceleration')
                plt.ylabel(r'AV longitudinal acceleration ($m/s^2$)')
            plt.axvline(x=statistical_data[i][initial_timestamp_ind], color='r', linestyle='--', label='start timestamp')
            if len(all_reaction_timing_list) > 0:
                if all_reaction_timing_list[i] != 0:
                    plt.axvline(x=all_reaction_timing_list[i], color='y', linestyle='--', label='reaction timestamp')
            if track_data[i][-1][timestamp_ind] == 0:
                plt.axvline(x=all_final_time[i], color='m', linestyle='--', label='collision timestamp')
            else:
                plt.axvline(x=all_final_time[i], color='m', linestyle='--', label='final timestamp')
            plt.xlabel('timestamp (s)')
            plt.legend()
            ylim[2] = plt.ylim()

            if case_id_ind == -1:
                plt.savefig(path + '/' + str(i + 1) + '.png')
            else:
                plt.savefig(path + '/' + str(case_id) + '.png')
            plt.close()

            key_timestamps.append([i+1, statistical_data[i][initial_timestamp_ind], all_reaction_timing_list[i] if len(all_reaction_timing_list) > 0 else 0, all_final_time[i]])

            subfigure_num = 3
            if scenario in ['Lane Departure (opposite)']:
                animation_data = {
                    0: [[cal_dis([line[AV_x_ind], line[AV_y_ind]], [line[challenger_x_ind], line[challenger_y_ind]]) for line in track_data[i][:-1] if statistical_data[i][initial_timestamp_ind] - additional_time <= line[timestamp_ind] <= all_final_time[i] + additional_time]],
                    1: [[line[AV_sp_ind] for line in track_data[i][:-1] if statistical_data[i][initial_timestamp_ind] - additional_time <= line[timestamp_ind] <= all_final_time[i] + additional_time],
                        [line[challenger_sp_ind] for line in track_data[i][:-1] if statistical_data[i][initial_timestamp_ind] - additional_time <= line[timestamp_ind] <= all_final_time[i] + additional_time]],
                    2: [[line[AV_lat_acceleration_ind] for line in track_data[i][:-1] if statistical_data[i][initial_timestamp_ind] - additional_time <= line[timestamp_ind] <= all_final_time[i] + additional_time]]
                }
                animation_labels = {
                    0: ['Distance between the AV center and the BV center'],
                    1: ['AV speed', 'Challenger speed'],
                    2: [r'AV lateral acceleration']
                }
            else:
                animation_data = {
                    0: [[cal_dis([line[AV_x_ind], line[AV_y_ind]], [line[challenger_x_ind], line[challenger_y_ind]]) for line in track_data[i][:-1] if statistical_data[i][initial_timestamp_ind] - additional_time <= line[timestamp_ind] <= all_final_time[i] + additional_time]],
                    1: [[line[AV_sp_ind] for line in track_data[i][:-1] if statistical_data[i][initial_timestamp_ind] - additional_time <= line[timestamp_ind] <= all_final_time[i] + additional_time],
                        [line[challenger_sp_ind] for line in track_data[i][:-1] if statistical_data[i][initial_timestamp_ind] - additional_time <= line[timestamp_ind] <= all_final_time[i] + additional_time]],
                    2: [[line[AV_lon_acceleration_ind] for line in track_data[i][:-1] if statistical_data[i][initial_timestamp_ind] - additional_time <= line[timestamp_ind] <= all_final_time[i] + additional_time]]
                }
                if scenario in ['jaywalking_child']:
                    animation_labels = {
                        0: ['Distance between the AV center and the VRU center'],
                        1: ['AV speed', 'VRU speed'],
                        2: [r'AV longitudinal acceleration']
                    }
                else:
                    animation_labels = {
                        0: ['Distance between the AV center and the BV center'],
                        1: ['AV speed', 'BV speed'],
                        2: [r'AV longitudinal acceleration']
                    }
            data_colors = ['b', 'g']
            special_timestamps = [statistical_data[i][initial_timestamp_ind], all_final_time[i]]
            timestamp_labels = ['start timestamp', 'final timestamp']
            timestamp_colors = ['r', 'y', 'm']
            xlim = [min(time), max(time)]
            xlabel = 'Timestamp (s)'
            ylabel = {
                0: 'Distance (m)',
                1: 'Speed (m/s)',
                2: r'Acceleration ($m/s^2$)'
            }
            title = {
                0: 'Distance',
                1: 'Speed',
                2: 'Acceleration'
            }
            if case_id_ind == -1:
                filename = path + '/' + str(i + 1) + '.mp4'
            else:
                filename = path + '/' + str(case_id) + '.mp4'
            fps = 100
            self._generate_variable_animation(
                figsize, subfigure_num, case_id, time, animation_data, animation_labels, data_colors, special_timestamps, timestamp_labels, timestamp_colors,
                xlim, ylim, xlabel, ylabel, title, FONTSIZE, filename, fps
            )

        save_csv(key_timestamps, ['case id', 'start timestamp', 'reaction timestamp', 'final timestamp'], path, 'timestamps', False)

    def _generate_variable_animation(
            self, figsize, subfigure_num, case_id, time, animation_data, animation_labels, data_colors, special_timestamps, timestamp_labels, timestamp_colors,
            xlim, ylim, xlabel, ylabel, title, fontsize, filename, fps):
        # Create a figure with subplots
        fig, axes = plt.subplots(subfigure_num, 1, figsize=figsize)
        if subfigure_num == 1:
            axes = [axes]
        
        # Adjust the spacing between subplots
        plt.subplots_adjust(hspace=0.4)  # Adjust the value of hspace as needed

        # Set up each subplot
        lines = []
        vertical_lines = []

        for i in range(subfigure_num):
            ax = axes[i]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim[i])
            if i == subfigure_num - 1:
                ax.set_xlabel(xlabel, fontsize=fontsize)
            ax.set_ylabel(ylabel[i], fontsize=fontsize)
            ax.set_title(title[i], fontsize=fontsize)

            # Plot data series for each subplot
            line_handles = []
            for j, data_series in enumerate(animation_data[i]):
                line, = ax.plot([], [], color=data_colors[j], label=animation_labels[i][j])
                line_handles.append(line)
            lines.append(line_handles)

            # Plot vertical lines for special timestamps in each subplot
            vlines = []
            for timestamp, label, color in zip(special_timestamps, timestamp_labels, timestamp_colors):
                if timestamp > 0:
                    vline = ax.axvline(timestamp, color=color, linestyle='--', label=label)
                    vline.set_visible(False)
                    vlines.append(vline)
            vertical_lines.append(vlines)

            # Add legend to each subplot without timestamp labels initially
            ax.legend([line for line in line_handles], [label for label in animation_labels[i]], fontsize=fontsize)

        fig.tight_layout()  # Automatically adjusts spacing to prevent overlap

        # Initialization function
        def init():
            for line_handles in lines:
                for line in line_handles:
                    line.set_data([], [])
            for vlines in vertical_lines:
                for vline in vlines:
                    vline.set_visible(False)
            return [line for line_handles in lines for line in line_handles] + [vline for vlines in vertical_lines for vline in vlines]

        # Animation function
        def update(frame):
            for i in range(subfigure_num):
                for j, data_series in enumerate(animation_data[i]):
                    lines[i][j].set_data(time[:frame + 1], data_series[:frame + 1])
                for k, vline in enumerate(vertical_lines[i]):
                    if time[frame] >= special_timestamps[k]:
                        vline.set_visible(True)
                # Update the legend to include only visible vertical lines
                handles, labels = axes[i].get_legend_handles_labels()
                visible_handles = [handle for handle in handles if handle.get_visible()]
                visible_labels = [label for handle, label in zip(handles, labels) if handle.get_visible()]
                axes[i].legend(visible_handles, visible_labels, loc='upper left', bbox_to_anchor=(0.008, 0.97), borderaxespad=0.)
            return [line for line_handles in lines for line in line_handles] + [vline for vlines in vertical_lines for vline in vlines]

        # Initialize tqdm progress bar
        num_frames = len(animation_data[0][0])
        progress_bar = tqdm(total=num_frames, desc=f"Generating animation of case {case_id}", unit="frame")

        # Wrapper for updating tqdm during animation
        def update_with_progress(frame):
            result = update(frame)
            progress_bar.update(1)
            return result

        # Create the animation
        ani = animation.FuncAnimation(
            fig, update_with_progress, frames=num_frames, init_func=init, blit=True, repeat=False
        )

        # Save the animation as a video
        ani.save(filename, writer="ffmpeg", fps=fps)

        # Close the progress bar after completion
        progress_bar.close()


if __name__ == '__main__':
    test_name = 'Tesla'

    for folder in ['left_turn_straight', 'vru_without_crosswalk']:
        settings = Settings(test_name=test_name, test_round=1, scenario_info_folder=folder)
        analyzer = Plot(settings)
        analyzer.plot_results()

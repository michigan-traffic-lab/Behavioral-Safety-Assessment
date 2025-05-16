import sys
import os

# Get the directory of the current script
dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(dir)

import copy
from sklearn.linear_model import LinearRegression
import argparse
from multiprocessing import Pool, Manager, cpu_count
from tqdm import tqdm

from env.data_process.extract_lanes import *
from plot_scenario import *
from utils import *
from settings import *


def vehicle_departing_lane_exist(track, grouped_straight_lane_segments, lane_width):
    flag = False
    start_ind, end_ind = [], []
    if track.object_type == 'vehicle':
        start_point = track.object_states[0]
        end_point = track.object_states[-1]
        for vector_straight_lane_segments, ind in zip(grouped_straight_lane_segments, range(len(grouped_straight_lane_segments))):
            for key, value in vector_straight_lane_segments.items():
                if check_point_inside_polygon(start_point.position, value.left_lane_marking.polyline[:, :2].tolist() + value.right_lane_marking.polyline[::-1, :2].tolist()):
                    start_ind.append(ind)
                if check_point_inside_polygon(end_point.position, value.left_lane_marking.polyline[:, :2].tolist() + value.right_lane_marking.polyline[::-1, :2].tolist()):
                    end_ind.append(ind)
        ind_flag = True
        for ind in start_ind:
            if ind in end_ind:
                ind_flag = False
                break
        neighbor_flag = False
        if ind_flag:
            for ind2 in end_ind:
                for key1, value1 in grouped_straight_lane_segments[ind].items():
                    for key2, value2 in grouped_straight_lane_segments[ind2].items():
                        if key1 == value2.left_neighbor_id or key1 == value2.right_neighbor_id or key2 == value1.left_neighbor_id or key2 == value1.right_neighbor_id:
                            neighbor_flag = True
                    if neighbor_flag:
                        break
                if neighbor_flag:
                    break
        if len(start_ind) > 0 and len(end_ind) > 0 and neighbor_flag:
            flag = True
    return flag, [start_ind, end_ind]


def extract_straight_tracks_cutted_in_by_lane_departuring_track(straight_tracks, lane_departure_track, start_lane_segments, start_lane_width, end_lane_segments, end_lane_width):
    same_direction_tracks = []
    opposite_direction_tracks = []
    other_tracks = []
    for straight_track in straight_tracks:
        for lane, width in zip(end_lane_segments, end_lane_width):
            if not check_track_in_lane(straight_track, lane, width):
                continue
            departure_point = find_departure_point(lane_departure_track, end_lane_segments)
            time_step = departure_point.timestep
            if not (straight_track.object_states[0].timestep <= time_step <= straight_track.object_states[-1].timestep):
                continue
            for ind in range(len(straight_track.object_states)):
                if straight_track.object_states[ind].timestep == time_step:
                    straight_point = straight_track.object_states[ind]
                    break
            straight_heading, _ = cal_track_start_heading(straight_track)
            lane_departure_heading, _ = cal_track_start_heading(lane_departure_track)
            rotated_departure_point = rotate_point2_around_point1(lane_departure_track.object_states[0].position, departure_point.position, lane_departure_heading)
            rotated_straight_point = rotate_point2_around_point1(lane_departure_track.object_states[0].position, straight_point.position, lane_departure_heading)
            dis = rotated_departure_point[0] - rotated_straight_point[0]

            if abs(straight_heading - lane_departure_heading) < 20 / 180 * math.pi and dis > 0:
                same_direction_tracks.append(straight_track)
            elif abs(abs(straight_heading - lane_departure_heading) - math.pi) < 20 / 180 * math.pi and dis < 0:
                opposite_direction_tracks.append(straight_track)
            else:
                other_tracks.append(straight_track)
            break
    return same_direction_tracks, opposite_direction_tracks, other_tracks


def extract_lane_departure_key_variable(lane_departure_track, straight_track, start_lane_segments, end_lane_segments, direction='same'):
    lane_change_flag = False
    start_heading, _ = cal_track_start_heading(lane_departure_track)
    start_point = lane_departure_track.object_states[0].position
    extracted_data = []
    for state in lane_departure_track.object_states:
        position = state.position
        dis2left_list = dis2right_list = []
        if not lane_change_flag:
            for lane_segments in start_lane_segments:
                for id, lane in lane_segments.items():
                    if check_point_inside_polygon(position, lane.left_lane_marking.polyline[:, :2].tolist() + lane.right_lane_marking.polyline[::-1, :2].tolist()):
                        dis2left_list.append(cal_dis_from_point_to_line(position, lane.left_lane_boundary.xyz[0], lane.left_lane_boundary.xyz[-1]))
                        dis2right_list.append(cal_dis_from_point_to_line(position, lane.right_lane_boundary.xyz[0], lane.right_lane_boundary.xyz[-1]))
                        break
            if len(dis2left_list) == 0:
                lane_change_flag = True
            else:
                dis2left = min(dis2left_list)
                dis2right = min(dis2right_list)
        else:
            for lane_segments in end_lane_segments:
                for id, lane in lane_segments.items():
                    if check_point_inside_polygon(position, lane.left_lane_marking.polyline[:, :2].tolist() + lane.right_lane_marking.polyline[::-1, :2].tolist()):
                        dis2left_list.append(-cal_dis_from_point_to_line(position, lane.left_lane_boundary.xyz[0], lane.left_lane_boundary.xyz[-1]))
                        dis2right_list.append(-cal_dis_from_point_to_line(position, lane.right_lane_boundary.xyz[0], lane.right_lane_boundary.xyz[-1]))
                        break
            if len(dis2left_list) == 0:
                dis2lane = 10000
                for lane_segments in end_lane_segments:
                    for id, lane in lane_segments.items():
                        if cal_dis(position[:2], lane.left_lane_boundary.xyz[0, :2]) < dis2lane:
                            dis2lane = cal_dis(position[:2], lane.left_lane_boundary.xyz[0, :2])
                            dis2left = cal_dis_from_point_to_line(position, lane.left_lane_boundary.xyz[0], lane.left_lane_boundary.xyz[-1])
                            dis2right = cal_dis_from_point_to_line(position, lane.right_lane_boundary.xyz[0], lane.right_lane_boundary.xyz[-1])
            else:
                dis2left = max(dis2left_list)
                dis2right = max(dis2right_list)

        if abs(dis2left) < 1 or abs(dis2right) < 1:
            time_step = state.timestep
            if not (straight_track.object_states[0].timestep <= time_step <= straight_track.object_states[-1].timestep):
                continue
            rotated_lane_departure_point = rotate_point2_around_point1(start_point, position, start_heading)
            lane_departure_speed = cal_dis(state.velocity, [0, 0])
            straight_speed_list = []
            for straight_state in straight_track.object_states:
                if time_step - 5 <= straight_state.timestep <= time_step + 5:
                    straight_speed_list.append(cal_dis(straight_state.velocity, [0, 0]))
                if time_step == straight_state.timestep:
                    straight_point = straight_state.position
                    straight_speed = cal_dis(straight_state.velocity, [0, 0])
            rotated_straight_point = rotate_point2_around_point1(start_point, straight_point, start_heading)
            straight_acc, _ = np.polyfit([DELTA_T * i for i in range(len(straight_speed_list))], straight_speed_list, 1)
            dis = rotated_lane_departure_point[0] - rotated_straight_point[0]
            lateral_dis = rotated_lane_departure_point[1] - rotated_straight_point[1]
            if straight_speed <= 0:
                continue
            if direction == 'same' and dis > 0:
                extracted_data.append([dis, lateral_dis, lane_departure_speed, straight_speed, straight_acc])
            elif direction == 'opposite' and dis < 0:
                extracted_data.append([abs(dis), lateral_dis, lane_departure_speed, straight_speed, straight_acc])
    return extracted_data


def find_closest_straight_track(lane_departure_track, straight_tracks):
    straight_track = None
    lane_departure_point = lane_departure_track.object_states[0].position
    dis = 10000
    for track in straight_tracks:
        point = track.object_states[0].position
        if cal_dis(lane_departure_point, point) < dis:
            straight_track = copy.deepcopy(track)
            dis = cal_dis(lane_departure_point, point)
    return straight_track


def locate_start_moment(data, direction='same'):
    lateral_dis = [line[1] for line in data]
    for i in range(2, len(lateral_dis) - 2):
        model = LinearRegression()
        model.fit(np.transpose([list(range(5))]), np.transpose([lateral_dis[i - 2:i + 3]]))
        slope = model.coef_[0][0]
        if slope > 0.1 or slope < -0.1:
            if direction == 'same':
                return data[i][0], data[i][3] - data[i][2]
            elif direction == 'opposite':
                return data[i][0], data[i][2] + data[i][3]
    return 0, 0


def extract_lane_change_key_variable(lane_departure_track, start_lane_segments, end_lane_segments):
    lane_change_flag = False
    start_heading, start_point_2 = cal_track_start_heading(lane_departure_track)
    start_point = lane_departure_track.object_states[0].position
    extracted_data = []
    start_position = [0, 0]
    final_position = [0, 0]
    for lane_segments in start_lane_segments:
        for id, lane in lane_segments.items():
            if check_point_inside_polygon(start_point, lane.left_lane_marking.polyline[:, :2].tolist() + lane.right_lane_marking.polyline[::-1, :2].tolist()):
                dis2left = cal_dis_from_point_to_line(start_point, lane.left_lane_boundary.xyz[0], lane.left_lane_boundary.xyz[-1])
                dis2right = cal_dis_from_point_to_line(start_point, lane.right_lane_boundary.xyz[0], lane.right_lane_boundary.xyz[-1])
                if dis2left > 1 and dis2right > 1:
                    for state in lane_departure_track.object_states:
                        position = state.position
                        dis2left_list = dis2right_list = []
                        if not lane_change_flag:
                            for lane_segments in start_lane_segments:
                                for id, lane in lane_segments.items():
                                    if check_point_inside_polygon(position, lane.left_lane_marking.polyline[:, :2].tolist() + lane.right_lane_marking.polyline[::-1, :2].tolist()):
                                        dis2left_list.append(cal_dis_from_point_to_line(position, lane.left_lane_boundary.xyz[0], lane.left_lane_boundary.xyz[-1]))
                                        dis2right_list.append(cal_dis_from_point_to_line(position, lane.right_lane_boundary.xyz[0], lane.right_lane_boundary.xyz[-1]))
                                        break
                            if len(dis2left_list) == 0:
                                lane_change_flag = True
                            else:
                                dis2left = min(dis2left_list)
                                dis2right = min(dis2right_list)
                        else:
                            for lane_segments in end_lane_segments:
                                for id, lane in lane_segments.items():
                                    if check_point_inside_polygon(position, lane.left_lane_marking.polyline[:, :2].tolist() + lane.right_lane_marking.polyline[::-1, :2].tolist()):
                                        dis2left_list.append(-cal_dis_from_point_to_line(position, lane.left_lane_boundary.xyz[0], lane.left_lane_boundary.xyz[-1]))
                                        dis2right_list.append(-cal_dis_from_point_to_line(position, lane.right_lane_boundary.xyz[0], lane.right_lane_boundary.xyz[-1]))
                                        break
                            if len(dis2left_list) == 0:
                                dis2lane = 10000
                                for lane_segments in end_lane_segments:
                                    for id, lane in lane_segments.items():
                                        if cal_dis(position[:2], lane.left_lane_boundary.xyz[0, :2]) < dis2lane:
                                            dis2lane = cal_dis(position[:2], lane.left_lane_boundary.xyz[0, :2])
                                            dis2left = cal_dis_from_point_to_line(position, lane.left_lane_boundary.xyz[0], lane.left_lane_boundary.xyz[-1])
                                            dis2right = cal_dis_from_point_to_line(position, lane.right_lane_boundary.xyz[0], lane.right_lane_boundary.xyz[-1])
                            else:
                                dis2left = max(dis2left_list)
                                dis2right = max(dis2right_list)
                        
                        if abs(dis2left) < 1 or abs(dis2right) < 1:
                            time_step = state.timestep
                            extracted_data.append(time_step)
                            final_position = copy.deepcopy(position)
                            if start_position == [0, 0]:
                                start_position = copy.deepcopy(position)

                    start_dis = cal_dis_from_point_to_line(start_position, start_point, start_point_2)
                    end_dis = cal_dis_from_point_to_line(final_position, start_point, start_point_2)
    lane_width = []
    for start_lane_segment in start_lane_segments:
        for id, lane in lane_segments.items():
            lane_width.append(cal_dis_from_point_to_line(lane.right_lane_boundary.xyz[0, :2], lane.left_lane_boundary.xyz[0, :2], lane.left_lane_boundary.xyz[-1, :2]))
    if len(extracted_data) > 1 and start_dis < sum(lane_width) / len(lane_width) / 2 - 1 and end_dis > sum(lane_width) / len(lane_width) / 2 + 1:
        return (max(extracted_data) - min(extracted_data)) * 0.1
    else:
        return 0


def process_folder(folder, train_path,
                   straight_heaading_threshold, shared_same_data, shared_opposite_data,
                   shared_same_direction_initial_conditions, shared_oposite_direction_initial_conditions):
    folder_path = os.path.join(train_path, folder)
    avm = load_argoverse_map(folder_path)
    vector_straight_lane_segments, straight_lane_width = extract_straight_lane(avm.vector_lane_segments)
    grouped_straight_lane_segments, grouped_straight_lane_width = group_lanes(vector_straight_lane_segments, straight_lane_width)
    city_name, all_tracks = load_argoverse_tracks(folder_path)
    straight_tracks = extract_straight_tracks(all_tracks, grouped_straight_lane_segments, grouped_straight_lane_width)

    local_opposite_data = []
    local_same_data = []
    local_same_direction_initial_conditions = []
    local_oposite_direction_initial_conditions = []

    for tracks in all_tracks:
        for one_track in tracks:
            start_heading, _ = cal_track_start_heading(one_track)
            end_heading, _ = cal_track_end_heading(one_track)
            diff = abs(start_heading - end_heading) % (2 * math.pi)
            diff = min(diff, 2 * math.pi - diff)
            if diff < straight_heaading_threshold:
                flag_lane_departure_exists, inds = vehicle_departing_lane_exist(one_track, grouped_straight_lane_segments, grouped_straight_lane_width)
                if flag_lane_departure_exists:
                    start_lane_segments = [grouped_straight_lane_segments[i] for i in inds[0]]
                    start_lane_width = [grouped_straight_lane_width[i] for i in inds[0]]
                    end_lane_segments = [grouped_straight_lane_segments[i] for i in inds[1]]
                    end_lane_width = [grouped_straight_lane_width[i] for i in inds[1]]
                    same_tracks, opposite_tracks, _ = extract_straight_tracks_cutted_in_by_lane_departuring_track(
                        straight_tracks, one_track, start_lane_segments, start_lane_width, end_lane_segments, end_lane_width)

                    lane_change_time = extract_lane_change_key_variable(one_track, start_lane_segments, end_lane_segments)
                    if lane_change_time != 0:
                        local_opposite_data.append([lane_change_time])

                    if len(same_tracks) > 0:
                        straight_track = find_closest_straight_track(one_track, same_tracks)
                        key_var = extract_lane_departure_key_variable(one_track, straight_track, start_lane_segments, end_lane_segments, 'same')
                        initial_dis, initial_relative_sp = locate_start_moment(key_var, 'same')
                        local_same_direction_initial_conditions.append([initial_dis, initial_relative_sp])
                        local_same_data.extend(key_var)
                    if len(opposite_tracks) > 0:
                        straight_track = find_closest_straight_track(one_track, opposite_tracks)
                        key_var = extract_lane_departure_key_variable(one_track, straight_track, start_lane_segments, end_lane_segments, 'opposite')
                        initial_dis, initial_relative_sp = locate_start_moment(key_var, 'opposite')
                        local_oposite_direction_initial_conditions.append([initial_dis, initial_relative_sp])

    shared_same_data.extend(local_same_data)
    shared_opposite_data.extend(local_opposite_data)
    shared_same_direction_initial_conditions.extend(local_same_direction_initial_conditions)
    shared_oposite_direction_initial_conditions.extend(local_oposite_direction_initial_conditions)


def process_folder_wrapper(args):
    return process_folder(*args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='/path/to/Argoverse2/dataset')
    parser.add_argument('--save-path', type=str, default='output/data_process')
    args = parser.parse_args()

    train_path = os.path.join(args.dataset_path, 'train')
    lane_departure_same_save_path = os.path.join(args.save_path, 'lane_departure_same')
    lane_departure_opposite_save_path = os.path.join(args.save_path, 'lane_departure_opposite')
    os.makedirs(lane_departure_same_save_path, exist_ok=True)
    os.makedirs(lane_departure_opposite_save_path, exist_ok=True)

    with open(os.path.join(args.save_path, 'folder_list.txt'), 'r') as f:
        folder_list = [line.strip() for line in f.readlines()]

    manager = Manager()
    same_data = manager.list()
    opposite_data = manager.list()
    same_direction_initial_conditions = manager.list()
    opposite_direction_initial_conditions = manager.list()

    num_workers = cpu_count()
    pool = Pool(processes=num_workers)

    folder_args = [
        (folder, train_path,
         straight_heaading_threshold, same_data, opposite_data, same_direction_initial_conditions, opposite_direction_initial_conditions)
        for folder in folder_list
    ]

    # for arg in folder_args:
    #     process_folder_wrapper(arg)

    with Pool(processes=8) as pool:
        for _ in tqdm(pool.imap_unordered(process_folder_wrapper, folder_args), total=len(folder_args), desc="Processing folders"):
            pass

    pool.close()
    pool.join()

    # Save results after parallel processing
    if same_data:
        save_csv(list(same_data), ['relative dis', 'lateral dis', 'BV speed', 'AV speed', 'AV acc'],
                 lane_departure_same_save_path, 'data', False)
    if opposite_data:
        save_csv(list(opposite_data), ['lane_change_time'],
                 lane_departure_opposite_save_path, 'data', False)
    if same_direction_initial_conditions:
        save_csv(list(same_direction_initial_conditions), ['relative dis', 'relative sp'],
                 lane_departure_same_save_path, 'init_conditions', False)
    if opposite_direction_initial_conditions:
        save_csv(list(opposite_direction_initial_conditions), ['relative dis', 'relative sp'],
                 lane_departure_opposite_save_path, 'init_conditions', False)

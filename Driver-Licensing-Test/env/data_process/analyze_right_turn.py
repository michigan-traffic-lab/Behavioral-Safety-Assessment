import sys
import os

# Get the directory of the current script
dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(dir)

import copy
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from env.data_process.extract_lanes import *
from plot_scenario import *


def right_turn_exist(track, vector_right_turn_lane_segments, lane_width):
    flag = False
    key = 0
    start_heading, _ = cal_track_start_heading(track)
    end_heading, _ = cal_track_end_heading(track)
    diff = abs(start_heading - math.pi / 2 - end_heading) % (2 * math.pi)
    diff = min(diff, 2 * math.pi - diff)
    if diff < turning_heading_threshold:
        return False, 0
    for state in track.object_states:
        position = state.position
        for [key, value], width in zip(vector_right_turn_lane_segments.items(), lane_width):
            dis2start = cal_dis(position, value.left_lane_boundary.xyz[0])
            dis2end = cal_dis(position, value.left_lane_boundary.xyz[-1])
            if dis2start < width:
                flag = True
                return flag, key
    return flag, key


def extract_straight_track_merging_to_turning_track(tracks, turning_tracks):
    straight_tracks = []
    for track in tracks:
        if track.object_type != 'vehicle':
            continue
        start_heading, _ = cal_track_start_heading(track)
        end_heading, _ = cal_track_end_heading(track)
        diff = abs(start_heading - end_heading) % (2 * math.pi)
        diff = min(diff, 2 * math.pi - diff)
        if diff > 0.2:
            continue
        for turning_track in turning_tracks:
            end_heading_for_turning_track, _ = cal_track_end_heading(turning_track)
            diff = abs(start_heading - end_heading_for_turning_track) % (2 * math.pi)
            diff = min(diff, 2 * math.pi - diff)
            if diff > 0.2:
                continue
            if collinear(track.object_states[0].position, track.object_states[-1].position, turning_track.object_states[-1].position, 0.02):
                straight_tracks.append(track)
    return straight_tracks


def extract_right_turn_key_variable(straight_track, right_turn_track):
    data = []
    straight_track_start_point = straight_track.object_states[0].position
    straight_track_end_point = straight_track.object_states[-1].position
    right_turn_track_start_point = right_turn_track.object_states[0].position
    right_turn_track_end_point = right_turn_track.object_states[-1].position
    for state in right_turn_track.object_states:
        dis = cal_dis(right_turn_track_start_point, state.position)
        if dis > 1:
            right_turn_track_point2 = state.position
            break
    intersection_point = line_intersection(straight_track_start_point, straight_track_end_point, right_turn_track_start_point, right_turn_track_point2)
    start_time = max(straight_track.object_states[0].timestep, right_turn_track.object_states[0].timestep)
    end_time = min(straight_track.object_states[-1].timestep, right_turn_track.object_states[-1].timestep)

    # rotate tracks around intersection point and make straight track be parallel to x-axis positive direction
    straight_track_angle = cal_angle(straight_track_end_point[0] - straight_track_start_point[0], straight_track_end_point[1] - straight_track_start_point[1])
    right_turn_track_angle = cal_angle(right_turn_track_point2[0] - right_turn_track_start_point[0], right_turn_track_point2[1] - right_turn_track_start_point[1])

    rotate_straight_track_start_point = rotate_point2_around_point1(intersection_point, straight_track_start_point, straight_track_angle)
    rotate_right_turn_track_start_point = rotate_point2_around_point1(intersection_point, right_turn_track_start_point, straight_track_angle)
    if rotate_straight_track_start_point[0] < rotate_right_turn_track_start_point[0]:
        # judge who pass the conflict point first
        dy = 10000
        for state in right_turn_track.object_states:
            dis = cal_dis_from_point_to_line(state.position, straight_track_start_point, straight_track_end_point)
            if dis < dy:
                dy = copy.deepcopy(dis)
                right_turn_conflict_state = copy.deepcopy(state)
                if dy < 1:
                    break
        straight_conflict_state = None
        for state in straight_track.object_states:
            if state.timestep == right_turn_conflict_state.timestep:
                straight_conflict_state = copy.deepcopy(state)
        if straight_conflict_state is not None:
            rotated_straight_cross_point = rotate_point2_around_point1(right_turn_conflict_state.position, straight_conflict_state.position, straight_track_angle)
            if rotated_straight_cross_point[0] < right_turn_conflict_state.position[0]:
                flag = 'turn'  # turning vehicle cross the conflict point first
            else:
                flag = 'straight'  # going straight vehicle cross the conflict point first
            cross_time_step = right_turn_conflict_state.timestep
            end_time = min(end_time, cross_time_step)

            straight_track_states = [state for state in straight_track.object_states if start_time <= state.timestep <= end_time]
            right_turn_track_states = [state for state in right_turn_track.object_states if start_time <= state.timestep <= end_time]

            for ind in range(end_time - start_time - 1):
                right_turn_state = right_turn_track_states[ind]
                rotate_right_turn_position = rotate_point2_around_point1(intersection_point, right_turn_state.position, right_turn_track_angle)
                # if abs(rotate_right_turn_position[1] - intersection_point[1]) > 2:
                straight_state = straight_track_states[ind]
                rotate_straight_position = rotate_point2_around_point1(intersection_point, straight_state.position, straight_track_angle)
                d1 = intersection_point[0] - rotate_straight_position[0]
                if abs(rotate_straight_position[0] - intersection_point[0]) < 50:  # and d1 > 0:
                    straight_state_ = straight_track_states[ind + 1]
                    rotate_straight_speed = rotate_point2_around_point1([0, 0], straight_state.velocity, straight_track_angle)
                    rotate_straight_speed_ = rotate_point2_around_point1([0, 0], straight_state_.velocity, straight_track_angle)

                    v1 = rotate_straight_speed[0]
                    start_ind = max(ind - 5, 0)
                    end_ind = min(ind + 5, end_time - start_time - 1)
                    v_list = [cal_dis(straight_track_states[i].velocity, [0, 0]) for i in range(start_ind, end_ind)]
                    time_list = [i * DELTA_T for i in range(start_ind, end_ind)]
                    if len(time_list) == 1:
                        break
                    a1, _ = np.polyfit(time_list, v_list, 1)
                    
                    right_turn_state_ = right_turn_track_states[ind + 1]
                    
                    rotate_right_turn_speed = rotate_point2_around_point1([0, 0], right_turn_state.velocity, right_turn_track_angle)
                    rotate_right_turn_speed_ = rotate_point2_around_point1([0, 0], right_turn_state_.velocity, right_turn_track_angle)
                    
                    d2 = abs(rotate_right_turn_position[0] - intersection_point[0])
                    v2 = rotate_right_turn_speed[0]
                    a2 = (rotate_right_turn_speed_[0] - rotate_right_turn_speed[0]) / DELTA_T / (right_turn_state_.timestep - right_turn_state.timestep)
                    v2 = cal_dis(right_turn_state.velocity, [0, 0])
                    v_list = [cal_dis(right_turn_track_states[i].velocity, [0, 0]) for i in range(start_ind, end_ind)]
                    a2, _ = np.polyfit(time_list, v_list, 1)
                    data.append([d1, v1, a1, d2, v2, a2, flag])
    return data


def process_folder(folder):
    folder_path = os.path.join(dataset_path, folder)
    avm = load_argoverse_map(folder_path)
    vector_right_turn_lane_segments, right_turn_lane_width = extract_right_turn_lane(avm.vector_lane_segments)

    city_name, all_tracks = load_argoverse_tracks(folder_path)
    local_data = {
        'traj': [],
        'init_conditions': []
    }

    for tracks in all_tracks:
        for one_track in tracks:
            flag, key = right_turn_exist(one_track, vector_right_turn_lane_segments, right_turn_lane_width)
            if not flag:
                continue
            straight_tracks = extract_straight_track_merging_to_turning_track(tracks, [one_track])
            for straight_track in straight_tracks:
                traj = extract_right_turn_key_variable(straight_track, one_track)
                if len(traj) > 0:
                    local_data['traj'].extend(traj)
                    local_data['init_conditions'].append(traj[0])

    return local_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='/path/to/Argoverse2/dataset')
    parser.add_argument('--save-path', type=str, default='output/data_process')
    args = parser.parse_args()

    dataset_path = os.path.join(args.dataset_path, 'train')
    save_path = os.path.join(args.save_path, 'right_turn')
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(args.save_path, 'folder_list.txt'), 'r') as f:
        folder_list = [folder.strip() for folder in f.readlines()]

    all_data = []
    init_conditions = []
    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_folder, folder_list), total=len(folder_list), desc='Processing folders'):
            all_data.extend(result['traj'])
            init_conditions.extend(result['init_conditions'])

    save_csv(all_data, ['d1', 'v1', 'a1', 'd2', 'v2', 'a2', 'flag'], save_path, 'data', False)
    save_csv(init_conditions, ['d1', 'v1', 'a1', 'd2', 'v2', 'a2', 'flag'], save_path, 'init_conditions', False)

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


def left_turn_exist(track, vector_left_turn_lane_segments, lane_width):
    flag = False
    key = 0
    start_heading, _ = cal_track_start_heading(track)
    end_heading, _ = cal_track_end_heading(track)
    diff = abs(start_heading + math.pi / 2 - end_heading) % (2 * math.pi)
    diff = min(diff, 2 * math.pi - diff)
    if diff > turning_heading_threshold:
        return False, 0
    for ind in range(len(track.object_states)):
        position = track.object_states[ind].position
        for [key, value], width in zip(vector_left_turn_lane_segments.items(), lane_width):
            if not check_point_inside_polygon(position, value.left_lane_marking.polyline[:, :2].tolist() + value.right_lane_marking.polyline[::-1, :2].tolist()):
                continue
            start_ind = max(ind - 5, 0)
            end_ind = min(start_ind + 10, len(track.object_states) - 1)
            track_heading = cal_angle(track.object_states[end_ind].position[0] - track.object_states[start_ind].position[0], track.object_states[end_ind].position[0] - track.object_states[start_ind].position[1])
            lane_heading = cal_angle(value.left_lane_boundary.xyz[-1, 0] - value.left_lane_boundary.xyz[0, 0], value.left_lane_boundary.xyz[-1, 1] - value.left_lane_boundary.xyz[0, 1])
            diff = abs(track_heading - lane_heading) % (2 * math.pi)
            diff = min(diff, 2 * math.pi - diff)
            if diff > turning_heading_threshold:
                continue
            flag = True
            return flag, key
    return flag, key


def extract_left_turn_lane(vector_lane_segments, flag_bike_lane_included=False):
    vector_left_turn_lane_segments = {}
    lane_width = []
    for key, value in vector_lane_segments.items():
        if (value.is_intersection and value.lane_type == 'VEHICLE') or flag_bike_lane_included:
            left_lane_marking = value.left_lane_boundary.xyz
            start_heading = cal_lane_start_heading(left_lane_marking)
            end_heading = cal_lane_end_heading(left_lane_marking)
            diff = abs(start_heading + math.pi / 2 - end_heading)
            diff = min(diff, abs(2 * math.pi - diff))
            if diff > turning_heading_threshold:
                continue
            vector_left_turn_lane_segments[key] = value
            right_lane_marking = value.right_lane_boundary.xyz
            lane_width.append(cal_dis(left_lane_marking[0], right_lane_marking[0]))
    return vector_left_turn_lane_segments, lane_width


def extract_straight_track_conflict_with_left_turning_track(tracks, turning_tracks):
    straight_tracks = []
    for track in tracks:
        if track.object_type != 'vehicle':
            continue
        start_heading, _ = cal_track_start_heading(track)
        end_heading, _ = cal_track_end_heading(track)
        for turning_track in turning_tracks:
            start_heading_for_turning_track, _ = cal_track_start_heading(turning_track)
            end_heading_for_turning_track, _ = cal_track_end_heading(turning_track)
            start_diff = abs(start_heading_for_turning_track + math.pi - start_heading) % (2 * math.pi)
            start_diff = min(start_diff, 2 * math.pi - start_diff)
            end_diff = abs(end_heading_for_turning_track + math.pi / 2 - end_heading) % (2 * math.pi)
            end_diff = min(end_diff, 2 * math.pi - end_diff)
            if start_diff > 0.2 or end_diff > 0.2:
                continue
            start_point_for_turning_track = turning_track.object_states[0].position
            end_point_for_turning_track = turning_track.object_states[-1].position
            start_point_for_straight_track = track.object_states[0].position
            end_point_for_straight_track = track.object_states[-1].position
            rotated_end_point_for_turning_track = rotate_point2_around_point1(start_point_for_turning_track, end_point_for_turning_track, start_heading_for_turning_track)
            rotated_start_point_for_straight_track = rotate_point2_around_point1(start_point_for_turning_track, start_point_for_straight_track, start_heading_for_turning_track)
            rotated_end_point_for_straight_track = rotate_point2_around_point1(start_point_for_turning_track, end_point_for_straight_track, start_heading_for_turning_track)
            if rotated_start_point_for_straight_track[0] > rotated_end_point_for_turning_track[0] and rotated_start_point_for_straight_track[1] < rotated_end_point_for_turning_track[1] \
                and rotated_start_point_for_straight_track[1] > start_point_for_turning_track[1] \
                and rotated_end_point_for_straight_track[1] < rotated_end_point_for_turning_track[1] and rotated_end_point_for_straight_track[1] > start_point_for_turning_track[1]:
                straight_tracks.append(track)
    return straight_tracks


def extract_left_turn_key_variable(straight_track, left_turn_track):
    data = []
    straight_track_start_point = straight_track.object_states[0].position
    straight_track_end_point = straight_track.object_states[-1].position
    left_turn_track_start_point = left_turn_track.object_states[0].position
    left_turn_track_end_point = left_turn_track.object_states[-1].position
    for state in left_turn_track.object_states:
        dis = cal_dis(left_turn_track_start_point, state.position)
        if dis > 1:
            left_turn_track_point1 = state.position
            break
    for state in left_turn_track.object_states[::-1]:
        dis = cal_dis(left_turn_track_end_point, state.position)
        if dis > 1:
            left_turn_track_point2 = state.position
            break
    intersection_point = line_intersection(straight_track_start_point, straight_track_end_point, left_turn_track_end_point, left_turn_track_point2)
    start_time = max(straight_track.object_states[0].timestep, left_turn_track.object_states[0].timestep)
    end_time = min(straight_track.object_states[-1].timestep, left_turn_track.object_states[-1].timestep)

    # rotate tracks around intersection point and make straight track be parallel to x-axis positive direction
    straight_track_angle = cal_angle(straight_track_end_point[0] - straight_track_start_point[0], straight_track_end_point[1] - straight_track_start_point[1])
    left_turn_track_angle = cal_angle(left_turn_track_point1[0] - left_turn_track_start_point[0], left_turn_track_point1[1] - left_turn_track_start_point[1])

    rotate_straight_track_start_point = rotate_point2_around_point1(intersection_point, straight_track_start_point, straight_track_angle)
    rotate_left_turn_track_start_point = rotate_point2_around_point1(intersection_point, left_turn_track_start_point, straight_track_angle)
    if rotate_straight_track_start_point[0] < rotate_left_turn_track_start_point[0]:
        # judge who pass the conflict point first
        dy = 10000
        left_turn_cross_point = None
        for state in left_turn_track.object_states:
            dis = cal_dis_from_point_to_line(state.position, straight_track_start_point, straight_track_end_point)
            if dis < dy:
                dy = copy.deepcopy(dis)
                left_turn_cross_state = copy.deepcopy(state)
        if dy < 0.5:
            straight_cross_state = None
            for state in straight_track.object_states:
                if state.timestep == left_turn_cross_state.timestep:
                    straight_cross_state = copy.deepcopy(state)
            if straight_cross_state is not None:
                rotated_straight_cross_point = rotate_point2_around_point1(left_turn_cross_state.position, straight_cross_state.position, straight_track_angle)
                if rotated_straight_cross_point[0] < left_turn_cross_state.position[0]:
                    flag = 'turn'  # turning vehicle cross the conflict point first
                else:
                    flag = 'straight'  # going straight vehicle cross the conflict point first
                cross_time_step = left_turn_cross_state.timestep

                straight_track_states = [state for state in straight_track.object_states if start_time <= state.timestep <= end_time]
                left_turn_track_states = [state for state in left_turn_track.object_states if start_time <= state.timestep <= end_time]

                for ind in range(end_time - start_time - 1):
                    left_turn_state = left_turn_track_states[ind]
                    if left_turn_state.timestep <= cross_time_step:
                            rotate_left_turn_position = rotate_point2_around_point1(left_turn_cross_state.position, left_turn_state.position, left_turn_track_angle)
                            straight_state = straight_track_states[ind]
                            rotate_straight_position = rotate_point2_around_point1(left_turn_cross_state.position, straight_state.position, straight_track_angle)
                            d1 = left_turn_cross_state.position[0] - rotate_straight_position[0]
                            if abs(rotate_straight_position[0] - left_turn_cross_state.position[0]) < 50 and d1 > 0:
                                straight_state_ = straight_track_states[ind + 1]
                                rotate_straight_speed = rotate_point2_around_point1([0, 0], straight_state.velocity, straight_track_angle)
                                rotate_straight_speed_ = rotate_point2_around_point1([0, 0], straight_state_.velocity, straight_track_angle)

                                v1 = rotate_straight_speed[0]
                                start_ind = max(ind - 5, 0)
                                end_ind = min(ind + 5, end_time - start_time - 1)
                                v_list = [cal_dis(straight_track_states[i].velocity, [0, 0]) for i in range(start_ind, end_ind)]
                                time_list = [i * DELTA_T for i in range(start_ind, end_ind)]
                                a1, _ = np.polyfit(time_list, v_list, 1)
                                d2 = abs(rotate_left_turn_position[0] - left_turn_cross_state.position[0])
                                v2 = cal_dis(left_turn_state.velocity, [0, 0])
                                v_list = [cal_dis(left_turn_track_states[i].velocity, [0, 0]) for i in range(start_ind, end_ind)]
                                a2, _ = np.polyfit(time_list, v_list, 1)
                                data.append([d1, v1, a1, d2, v2, a2, flag])
    return data


def process_folder(folder):
    folder_path = os.path.join(dataset_path, folder)
    avm = load_argoverse_map(folder_path)
    vector_left_turn_lane_segments, left_turn_lane_width = extract_left_turn_lane(avm.vector_lane_segments)
    vector_straight_lane_segments, straight_lane_width = extract_straight_lane(avm.vector_lane_segments)
    grouped_straight_lane_segments, grouped_straight_lane_width = group_lanes(vector_straight_lane_segments, straight_lane_width)

    city_name, all_tracks = load_argoverse_tracks(folder_path)
    straight_tracks = extract_straight_tracks(all_tracks, grouped_straight_lane_segments, grouped_straight_lane_width)

    data = {
        'traj': [],
        'init_conditions': []
    }
    for tracks in all_tracks:
        for one_track in tracks:
            if one_track in straight_tracks:
                continue

            flag, key = left_turn_exist(one_track, vector_left_turn_lane_segments, left_turn_lane_width)

            if flag:
                conflict_straight_tracks = extract_straight_track_conflict_with_left_turning_track(straight_tracks, [one_track])
                for straight_track in conflict_straight_tracks:
                    traj = extract_left_turn_key_variable(straight_track, one_track)
                    if len(traj) > 0:
                        data['traj'].extend(traj)
                        data['init_conditions'].extend([traj[0]])
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='/path/to/Argoverse2/dataset')
    parser.add_argument('--save-path', type=str, default='output/data_process')
    args = parser.parse_args()

    dataset_path = os.path.join(args.dataset_path, 'train')
    save_path = os.path.join(args.save_path, 'left_turn')
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(args.save_path, 'folder_list.txt'), 'r') as f:
        folder_list = [folder.strip() for folder in f.readlines()]

    all_left_turn_data = []
    init_conditions = []
    worker_num = cpu_count()
    with Pool(processes=worker_num) as pool:
        for data in tqdm(pool.imap_unordered(process_folder, folder_list), total=len(folder_list), desc='Processing folders'):
            all_left_turn_data.extend(data['traj'])
            init_conditions.extend(data['init_conditions'])

    # Save all results in one go
    save_csv(all_left_turn_data, ['d1', 'v1', 'a1', 'd2', 'v2', 'a2', 'flag'], save_path, 'data', False)
    save_csv(init_conditions, ['d1', 'v1', 'a1', 'd2', 'v2', 'a2', 'flag'], save_path, 'init_conditions', False)

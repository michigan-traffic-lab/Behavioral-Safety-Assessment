import sys
import os

# Get the directory of the current script
dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(dir)

import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from env.data_process.extract_lanes import *
from plot_scenario import *


def extract_straight_track_crossing_pedestrian_track(tracks, pedestrian_track, vector_straight_lane_segments, lane_width):
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
        flag = False
        for [_, lane], width in zip(vector_straight_lane_segments.items(), lane_width):
            lane_heading = cal_lane_start_heading(lane.left_lane_boundary.xyz)
            diff = abs(start_heading - lane_heading) % (2 * math.pi)
            diff = min(diff, 2 * math.pi - diff)
            if diff > 0.2:
                continue
            for state in track.object_states:
                dis2left = cal_dis_from_point_to_line(state.position, lane.left_lane_boundary.xyz[0], lane.left_lane_boundary.xyz[-1])
                dis2right = cal_dis_from_point_to_line(state.position, lane.right_lane_boundary.xyz[0], lane.right_lane_boundary.xyz[-1])
                if dis2left < width and dis2right < width:
                    straight_tracks.append(track)
                    flag = True
                    break
            if flag:
                break
    return straight_tracks


def pedestrian_crossing_exist(track, vector_straight_lane_segments, lane_width, vector_intersection_lane_segments):
    flag = False  # whether the pedestrian is crossing at an intersection
    keys = []
    if track.object_type not in ['pedestrian', 'cyclist']:
        return flag, keys
    for state in track.object_states:
        position = state.position
        for [key, value], width in zip(vector_straight_lane_segments.items(), lane_width):
            dis2left = cal_dis_from_point_to_line(position, value.left_lane_boundary.xyz[0], value.left_lane_boundary.xyz[-1])
            dis2right = cal_dis_from_point_to_line(position, value.right_lane_boundary.xyz[0], value.right_lane_boundary.xyz[-1])
            if dis2left > width or dis2right > width:
                continue
            start_heading, _ = cal_track_start_heading(track)
            end_heading, _ = cal_track_end_heading(track)
            lane_heading = cal_angle(value.left_lane_boundary.xyz[-1][0] - value.left_lane_boundary.xyz[0][0], value.left_lane_boundary.xyz[-1][1] - value.left_lane_boundary.xyz[0][1])
            diff = abs(end_heading - lane_heading) % (2 * math.pi)
            diff = min(diff, 2 * math.pi - diff)
            if abs(diff - math.pi / 2) > 0.2:
                continue
            if key not in keys:
                keys.append(key)
    for _, intersection_lane in vector_intersection_lane_segments.items():
        left_start = intersection_lane.left_lane_boundary.xyz[0]
        left_end = intersection_lane.left_lane_boundary.xyz[-1]
        right_start = intersection_lane.right_lane_boundary.xyz[0]
        right_end = intersection_lane.right_lane_boundary.xyz[-1]
        if (min(left_start[0], left_end[0]) - 5 <= track.object_states[-1].position[0] <= max(left_start[0], left_end[0]) + 5 and min(left_start[1], left_end[1]) - 5 <= track.object_states[-1].position[1] <= max(left_start[1], left_end[1]) + 5) or \
            (min(right_start[0], right_end[0]) - 5 <= track.object_states[-1].position[0] <= max(right_start[0], right_end[0]) + 5 and min(right_start[1], right_end[1]) - 5 <= track.object_states[-1].position[1] <= max(right_start[1], right_end[1]) + 5):
            flag = True
            break
    return flag, keys


def extract_crossing_data(VRU_track, AV_tracks):
    VRU_end_heading, _ = cal_track_end_heading(VRU_track)
    for ind in range(len(VRU_track.object_states)):
        heading = None
        for ind_ in range(ind, len(VRU_track.object_states)):
            dis = cal_dis(VRU_track.object_states[ind].position, VRU_track.object_states[ind_].position)
            if dis < 1:
                continue
            heading = cal_angle(VRU_track.object_states[ind_].position[0] - VRU_track.object_states[ind].position[0], VRU_track.object_states[ind_].position[1] - VRU_track.object_states[ind].position[1])
            break
        if heading is None:
            heading = cal_angle(VRU_track.object_states[-1].position[0] - VRU_track.object_states[ind].position[0], VRU_track.object_states[-1].position[1] - VRU_track.object_states[ind].position[1])
        diff = abs(heading - VRU_end_heading) % (2 * math.pi)
        diff = min(diff, 2 * math.pi - diff)
        if diff < 0.2:
            break
    VRU_crossing_states = VRU_track.object_states[ind:]
    AV_crossing_tracks = []
    for AV_track in AV_tracks:
        AV_start_heading, _ = cal_track_start_heading(AV_track)
        diff = abs(AV_start_heading - VRU_end_heading) % (2 * math.pi)
        diff = min(diff, 2 * math.pi - diff)
        if abs(diff - math.pi / 2) > 0.2:
            continue
        flag, _, _, _ = same_side([VRU_crossing_states[0].position, VRU_crossing_states[-1].position], AV_track.object_states[0].position, AV_track.object_states[-1].position)
        if not flag and AV_track.object_states[0].timestep <= VRU_crossing_states[0].timestep:
            AV_crossing_tracks.append(AV_track)
    start_dis = 10000
    AV_crossing_track = None
    for AV_track in AV_crossing_tracks:
        temp_start_dis = cal_dis_from_point_to_line(AV_track.object_states[0].position, VRU_crossing_states[0].position, VRU_crossing_states[-1].position)
        if AV_crossing_track is None:
            start_dis = temp_start_dis
            AV_crossing_track = AV_track
        if temp_start_dis < start_dis:
            start_dis = temp_start_dis
            AV_crossing_track = AV_track

    data = []
    if AV_crossing_track is not None:
        if VRU_crossing_states[0].timestep < AV_crossing_track.object_states[-1].timestep:
            VRU_start_ind = 0
            VRU_end_ind = min(VRU_crossing_states[-1].timestep, AV_crossing_track.object_states[-1].timestep) - VRU_crossing_states[0].timestep
            VRU_crossing_states = VRU_crossing_states[VRU_start_ind:VRU_end_ind]
            
            AV_start_ind = VRU_crossing_states[0].timestep - AV_crossing_track.object_states[0].timestep
            AV_end_ind = min(VRU_crossing_states[-1].timestep, AV_crossing_track.object_states[-1].timestep) - AV_crossing_track.object_states[0].timestep + 1
            AV_states = AV_crossing_track.object_states[AV_start_ind:AV_end_ind]

            length = len(VRU_crossing_states)
            if length > 1:
                for ind in range(1, length):
                    veh_lon_dis = cal_dis_from_point_to_line(AV_states[ind].position, VRU_crossing_states[0].position, VRU_crossing_states[-1].position)
                    veh_sp = cal_dis(AV_states[ind].velocity, [0, 0])
                    start_ind = max(ind - 5, 0)
                    end_ind = min(ind + 5, length, len(AV_states))
                    if end_ind > start_ind:
                        sp_list = [cal_dis(AV_states[i].velocity, [0, 0]) for i in range(start_ind, end_ind)]
                        t_list = [i * DELTA_T for i in range(start_ind, end_ind)]
                        veh_acc, _ = np.polyfit(t_list, sp_list, 1)
                        VRU_sp = cal_dis(VRU_crossing_states[ind].velocity, [0, 0])
                        data.append([veh_lon_dis, veh_sp, veh_acc, VRU_sp])
    return data


def process_folder(folder):
    folder_path = os.path.join(train_path, folder)
    avm = load_argoverse_map(folder_path)
    vector_straight_lane_segments, straight_lane_width = extract_straight_lane(avm.vector_lane_segments)
    vector_intersection_lane_segments = extract_intersection_lane(avm.vector_lane_segments, False)

    city_name, all_tracks = load_argoverse_tracks(folder_path)
    local_data = {
        'traj': [],
        'init_conditions': [],
    }

    for tracks in all_tracks:
        for one_track in tracks:
            flag_pedestrian_at_intersection, keys = pedestrian_crossing_exist(
                one_track, vector_straight_lane_segments, straight_lane_width, vector_intersection_lane_segments
            )
            if len(keys) > 0:
                filtered_segments = {key: vector_straight_lane_segments[key] for key in keys}
                straight_tracks = extract_straight_track_crossing_pedestrian_track(
                    tracks, one_track, filtered_segments, straight_lane_width
                )
                if straight_tracks:
                    traj = extract_crossing_data(one_track, straight_tracks)
                    if len(traj) > 0:
                        local_data['traj'].extend(traj)
                        local_data['init_conditions'].append(traj[0])

    return local_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='/path/to/Argoverse2/dataset')
    parser.add_argument('--save-path', type=str, default='output/data_process')
    args = parser.parse_args()

    train_path = os.path.join(args.dataset_path, 'train')
    save_path = os.path.join(args.save_path, 'vru')

    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(args.save_path, 'folder_list.txt'), 'r') as f:
        folder_list = [folder.strip() for folder in f.readlines()]

    all_data = []
    init_conditions = []
    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_folder, folder_list), total=len(folder_list), desc='Processing folders'):
            all_data.extend(result['traj'])
            init_conditions.extend(result['init_conditions'])

    save_csv(all_data, ['veh_lon_dis', 'veh_sp', 'veh_acc', 'ped_sp'], save_path, 'data', False)
    save_csv(init_conditions, ['veh_lon_dis', 'veh_sp', 'veh_acc', 'ped_sp'], save_path, 'init_conditions', False)

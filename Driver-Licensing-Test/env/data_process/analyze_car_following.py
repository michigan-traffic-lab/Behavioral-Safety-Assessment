import sys
import os

# Get the directory of the current script
dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(dir)

import argparse
import numpy as np
import math
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from env.data_process.extract_lanes import *

from utils import *
from settings import *


def grouping_tracks(tracks):
    grouped_tracks = []
    for track in tracks:
        flag = True
        for grouped_track in grouped_tracks:
            if abs(track[1] - grouped_track[0][1]) < 0.1 or abs(abs(track[1] - grouped_track[0][1]) - 2 * math.pi) < 0.1:
                rotated_track_start_position = rotate_point2_around_point1(grouped_track[0][0].object_states[0].position, track[0].object_states[0].position, grouped_track[0][1])
                if abs(rotated_track_start_position[1] - grouped_track[0][0].object_states[0].position[1]) < 2:
                    grouped_track.append(track)
                    flag = False
                    break
        if flag:
            grouped_tracks.append([track])
    return grouped_tracks

def extract_data(tracks):
    start_timestep = [track.object_states[0].timestep for track, _ in tracks]
    end_timestep = [track.object_states[-1].timestep for track, _ in tracks]
    start_time = min(start_timestep)
    end_time = max(end_timestep)

    rotate_center = tracks[0][0].object_states[0].position
    rotate_angle = tracks[0][1]

    data = []

    for t in range(start_time, end_time):
        positions = []
        for track, _ in tracks:
            if len(track.object_states) > 7:
                if track.object_states[3].timestep <= t <= track.object_states[-4].timestep:
                    ind = t - track.object_states[0].timestep
                    position = track.object_states[ind].position
                    velocity = cal_dis(track.object_states[ind].velocity, [0, 0])
                    x = [tmp_ind * DELTA_T for tmp_ind in range(ind-3, ind+4)]
                    y = [cal_dis(track.object_states[tmp_ind].velocity, [0, 0]) for tmp_ind in range(ind-3, ind+4)]
                    acceleration, _ = np.polyfit(x, y, 1)
                    positions.append([position, velocity, acceleration])
        if len(positions) > 1:
            rotated_positions = [[rotate_point2_around_point1(rotate_center, position, rotate_angle)[0], velocity, acceleration] for position, velocity, acceleration in positions]
            rotated_positions = np.array(rotated_positions)
            rotated_positions = rotated_positions[rotated_positions[:, 0].argsort()]
            for ind in range(len(rotated_positions) - 1):
                data.append([rotated_positions[ind + 1][0] - rotated_positions[ind][0], rotated_positions[ind][1], rotated_positions[ind][2], rotated_positions[ind + 1][1], rotated_positions[ind + 1][2]])
    return data


def process_folder(folder):
    folder_path = os.path.join(train_path, folder)
    _, all_tracks = load_argoverse_tracks(folder_path)

    straight_tracks = []
    for tracks in all_tracks:
        for one_track in tracks:
            flag, start_heading, end_heading, length = judge_straight_track(one_track)
            if flag:
                straight_tracks.append([one_track, start_heading])

    grouped_tracks = grouping_tracks(straight_tracks)
    return [item for group in grouped_tracks for item in extract_data(group)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='/path/to/Argoverse2/dataset')
    parser.add_argument('--save-path', type=str, default='output/data_process')
    args = parser.parse_args()

    train_path = os.path.join(args.dataset_path, 'train')
    save_path = os.path.join(args.save_path, 'car_following')
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(args.save_path, 'folder_list.txt'), 'r') as f:
        folder_list = [folder.strip() for folder in f.readlines()]

    # Run parallel processing
    all_data = []
    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_folder, folder_list), total=len(folder_list), desc='Processing folders'):
            all_data.extend(result)
    
    save_csv(all_data, ['dis', 'v_AV', 'a_AV', 'v_BV', 'a_BV'], save_path, 'data', False)

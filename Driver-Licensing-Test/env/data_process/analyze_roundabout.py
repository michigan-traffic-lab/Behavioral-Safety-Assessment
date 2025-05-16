import sys
import os

# Get the directory of the current script
dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(dir)

import pickle
import os
from rich.progress import track
import argparse

from env.data_process.roundabout.realistic_metric import *


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--dataset-path', type=str, default='/path/to/processed/rounD/dataset')
    argparse.add_argument('--save-path', type=str, default='output/data_process')
    args = argparse.parse_args()

    save_path = args.save_path + '/roundabout'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    path_to_traj_data = args.dataset_path + '/processed_data'

    drivable_map_dir = args.dataset_path + '/inference/rounD/drivablemap/rounD-drivablemap.jpg'
    sim_remove_vehicle_area_map = args.dataset_path + '/inference/rounD/ROIs-map/rounD-sim-remove-vehicle-area-map.jpg'

    # ROIs
    circle_map_dir = os.path.join(args.dataset_path + '/inference/rounD/ROIs-map/circle')
    entrance_map_dir = os.path.join(args.dataset_path + '/inference/rounD/ROIs-map/entrance')
    exit_map_dir = os.path.join(args.dataset_path + '/inference/rounD/ROIs-map/exit')
    crosswalk_map_dir = os.path.join(args.dataset_path + '/inference/rounD/ROIs-map/crosswalk')
    yielding_area_map_dir = os.path.join(args.dataset_path + '/inference/rounD/ROIs-map/yielding-area')
    at_circle_lane_map_dir = os.path.join(args.dataset_path + '/inference/rounD/ROIs-map/at-circle-lane')

    SimMetricsAnalyzer = RealisticMetrics(drivable_map_dir=drivable_map_dir, sim_remove_vehicle_area_map=sim_remove_vehicle_area_map,
                                          circle_map_dir=circle_map_dir, entrance_map_dir=entrance_map_dir,
                                          exit_map_dir=exit_map_dir, crosswalk_map_dir=crosswalk_map_dir, yielding_area_map_dir=yielding_area_map_dir,
                                          at_circle_lane_map_dir=at_circle_lane_map_dir,
                                          sim_resol=0.4)

    TIME_BUFF = []
    for i in track(range(89915), 'Loading trajectory data...'):
        vehicle_list = pickle.load(open(os.path.join(path_to_traj_data, str(i) + '.pickle'), "rb"))
        TIME_BUFF.append(vehicle_list)

    SimMetricsAnalyzer.construct_traj_data(TIME_BUFF)

    yielding_conflict_dist_and_v_dict = SimMetricsAnalyzer.yilding_distance_and_speed_analysis_parallel()

    # Save data
    save_csv(SimMetricsAnalyzer.acc, ['inner acc', 'outer acc'], save_path, 'data', False)

    save_csv(SimMetricsAnalyzer.dis_sp, ['inner dis', 'inner speed', 'outer dis', 'outer speed'], save_path, 'init_conditions', False)

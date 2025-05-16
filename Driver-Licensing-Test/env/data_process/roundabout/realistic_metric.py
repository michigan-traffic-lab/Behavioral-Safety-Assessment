"""
This class is to calculate realistic metrics to validate the performance of the proposed simulator.
"""
import os
import pandas as pd
import copy
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'
from itertools import combinations
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import math
from rich.progress import track
from tqdm import tqdm
from collections import defaultdict

from utils import *
from env.data_process.roundabout.road_matching import RoadMatcher
from env.data_process.roundabout.ROIs import ROIMatcher
from env.data_process.roundabout.trajectory_pool import TrajectoryPool


def extract_key_variables(track_id, path):
    delta_t = 0.4
    x = [vehicle.location.x for vehicle in path]
    y = [vehicle.location.y for vehicle in path]
    velocity = []
    acc = []
    all_dis = 0
    for ind in range(len(x) - 1):
        dis = math.sqrt((x[ind] - x[ind+1])**2 + (y[ind] - y[ind+1])**2)
        velocity.append(dis / delta_t)
        all_dis += dis
    ind = min(len(velocity), 5)
    start_velocity = sum(velocity[:ind]) / ind
    for ind in range(len(velocity)):
        start_ind = max(ind - 3, 0)
        end_ind = min(ind + 3, len(velocity))
        v_list = velocity[start_ind:end_ind]
        t_list = [t * delta_t for t in range(start_ind, end_ind)]
        a, _ = np.polyfit(t_list, v_list, 1)
        acc.append(a)
    return all_dis, start_velocity, acc


class RealisticMetrics(object):

    def __init__(self, drivable_map_dir=None, sim_remove_vehicle_area_map=None,
                 circle_map_dir=None, entrance_map_dir=None, exit_map_dir=None, crosswalk_map_dir=None, yielding_area_map_dir=None, at_circle_lane_map_dir=None,
                 sim_resol=0.4,
                 map_height=936, map_width=1678,
                 PET_configs=None):

        self.road_matcher = RoadMatcher(map_file_dir=drivable_map_dir, map_height=map_height, map_width=map_width)

        self.ROI_matcher = ROIMatcher(drivable_map_dir=drivable_map_dir, sim_remove_vehicle_area_map_dir=sim_remove_vehicle_area_map, circle_map_dir=circle_map_dir,
                                      entrance_map_dir=entrance_map_dir, exit_map_dir=exit_map_dir, crosswalk_map_dir=crosswalk_map_dir, yielding_area_map_dir=yielding_area_map_dir,
                                      at_circle_lane_map_dir=at_circle_lane_map_dir,
                                      map_height=map_height, map_width=map_width)

        self.traj_pool = None
        self.traj_df = None
        self.TIME_BUFF = None
        self.sim_resol = sim_resol  # simulation resolution in [s]
        self.PET_configs = PET_configs
        self.extracted_veh_pairs = []
        self.dis_sp = []
        self.acc = np.empty((0, 2))
        self.track_id = 0

    def time_buff_to_traj_pool(self, TIME_BUFF):
        traj_pool = TrajectoryPool(max_missing_age=float("inf"), road_matcher=self.road_matcher, ROI_matcher=self.ROI_matcher)
        for frame in track(TIME_BUFF, description='Time buffer to trajectory pool ...'):
            traj_pool.update(frame, ROI_matching=True)
        return traj_pool

    def construct_traj_data(self, TIME_BUFF):

        self.TIME_BUFF = TIME_BUFF

        # Construct traj pool
        self.traj_pool = self.time_buff_to_traj_pool(TIME_BUFF)
        self.traj_df = pd.DataFrame(columns=['vid', 'x', 'y', 'heading', 'region_position', 'yielding_area', 'at_circle_lane', 't', 'update', 'vehicle', 'dt', 'missing_days'])

        # # Construct traj df
        # for vid in track(self.traj_pool.vehicle_id_list(), description='Constructing trajectory data ...'):
        #     self.traj_df = self.traj_df.append(pd.DataFrame.from_dict(self.traj_pool.pool[vid]), ignore_index=True)

        traj_dataframes = []
        for vid in track(self.traj_pool.vehicle_id_list(), description='Constructing trajectory data ...'):
            traj_dataframes.append(pd.DataFrame.from_dict(self.traj_pool.pool[vid]))

        self.traj_df = pd.concat(traj_dataframes, ignore_index=True)

        # if 'crosswalk' in self.traj_df.region_position.unique() or 'off_road' in self.traj_df.region_position.unique():
        #     print(self.traj_df.region_position.unique())

    def OD_analysis_parallel(self):

        traj_num = 0  # number of vehicles
        complete_traj_num = 0  # number of vehicles complete the whole trip with entrance and exit

        OD_dict = defaultdict(lambda: defaultdict(int))
        grouped = self.traj_df.groupby('vid')

        for vid, v_traj in tqdm(grouped, desc="OD analysis"):
            traj_num += 1
            region_seq = v_traj['region_position'].tolist()

            # Skip trajectories that are too short
            if len(region_seq) < 2:
                continue

            v_entrance, v_exit = self._get_entrance_exit(region_seq)
            if v_entrance is not None and v_exit is not None:
                complete_traj_num += 1
                OD_dict[v_entrance][v_exit] += 1

        # readable_OD_dict = self._OD_dict_to_readable_dict(OD_dict)

        # sim_wall_time = (self.traj_df.t.max() - self.traj_df.t.min() + 1) * self.sim_resol

        # return OD_dict, readable_OD_dict, traj_num, complete_traj_num, sim_wall_time
    
    def OD_analysis(self):

        traj_num = 0  # number of vehicles
        complete_traj_num = 0  # number of vehicles complete the whole trip with entrance and exit

        OD_dict = {}
        for vid in track(self.traj_pool.vehicle_id_list(), description='OD analysis ...'):
            traj_num += 1

            v_traj = self.traj_df[self.traj_df['vid'] == vid]
            v_traj_region_position = v_traj.region_position.tolist()

            v_entrance, v_exit = self._get_entrance_exit(v_traj_region_position)
            if v_entrance is not None and v_exit is not None:
                complete_traj_num += 1

                if v_entrance not in OD_dict.keys():
                    OD_dict[v_entrance] = {v_exit: 1}
                else:
                    if v_exit not in OD_dict[v_entrance].keys():
                        OD_dict[v_entrance][v_exit] = 1
                    else:
                        OD_dict[v_entrance][v_exit] += 1

        readable_OD_dict = self._OD_dict_to_readable_dict(OD_dict)

        sim_wall_time = (self.traj_df.t.max() - self.traj_df.t.min() + 1) * self.sim_resol

        return OD_dict, readable_OD_dict, traj_num, complete_traj_num, sim_wall_time

    def _OD_dict_to_readable_dict(self, OD_dict):
        mapping_dict = {('entrance_n_1', 'exit_e'): ['entrance_N1', 'left'],
                        ('entrance_n_1', 'exit_s'): ['entrance_N1', 'through'],
                        ('entrance_n_1', 'exit_w'): ['entrance_N1', 'right'],
                        ('entrance_n_1', 'exit_n'): ['entrance_N1', 'U-turn'],
                        ('entrance_n_2', 'exit_e'): ['entrance_N2', 'left'],
                        ('entrance_n_2', 'exit_s'): ['entrance_N2', 'through'],
                        ('entrance_n_2', 'exit_w'): ['entrance_N2', 'right'],
                        ('entrance_n_2', 'exit_n'): ['entrance_N2', 'U-turn'],

                        ('entrance_e_1', 'exit_e'): ['entrance_E1', 'U-turn'],
                        ('entrance_e_1', 'exit_s'): ['entrance_E1', 'left'],
                        ('entrance_e_1', 'exit_w'): ['entrance_E1', 'through'],
                        ('entrance_e_1', 'exit_n'): ['entrance_E1', 'right'],
                        ('entrance_e_2', 'exit_e'): ['entrance_E2', 'U-turn'],
                        ('entrance_e_2', 'exit_s'): ['entrance_E2', 'left'],
                        ('entrance_e_2', 'exit_w'): ['entrance_E2', 'through'],
                        ('entrance_e_2', 'exit_n'): ['entrance_E2', 'right'],

                        ('entrance_s_1', 'exit_e'): ['entrance_S1', 'right'],
                        ('entrance_s_1', 'exit_s'): ['entrance_S1', 'U-turn'],
                        ('entrance_s_1', 'exit_w'): ['entrance_S1', 'left'],
                        ('entrance_s_1', 'exit_n'): ['entrance_S1', 'through'],
                        ('entrance_s_2', 'exit_e'): ['entrance_S2', 'right'],
                        ('entrance_s_2', 'exit_s'): ['entrance_S2', 'U-turn'],
                        ('entrance_s_2', 'exit_w'): ['entrance_S2', 'left'],
                        ('entrance_s_2', 'exit_n'): ['entrance_S2', 'through'],

                        ('entrance_w_1', 'exit_e'): ['entrance_W1', 'through'],
                        ('entrance_w_1', 'exit_s'): ['entrance_W1', 'right'],
                        ('entrance_w_1', 'exit_w'): ['entrance_W1', 'U-turn'],
                        ('entrance_w_1', 'exit_n'): ['entrance_W1', 'left'],
                        ('entrance_w_2', 'exit_e'): ['entrance_W2', 'through'],
                        ('entrance_w_2', 'exit_s'): ['entrance_W2', 'right'],
                        ('entrance_w_2', 'exit_w'): ['entrance_W2', 'U-turn'],
                        ('entrance_w_2', 'exit_n'): ['entrance_W2', 'left'],

                        ('entrance_n_rightturn', 'exit_n_rightturn'): ['N_dedicated_right_turn', 'N_dedicated_right_turn'],
                        ('entrance_n_rightturn', 'exit_e'): ['N_dedicated_right_turn', 'left'],
                        ('entrance_n_rightturn', 'exit_s'): ['N_dedicated_right_turn', 'through'],
                        ('entrance_n_rightturn', 'exit_w'): ['N_dedicated_right_turn', 'right'],
                        ('entrance_n_rightturn', 'exit_n'): ['N_dedicated_right_turn', 'U-turn'],

                        ('entrance_s_rightturn', 'exit_s_rightturn'): ['S_dedicated_right_turn', 'S_dedicated_right_turn'],
                        ('entrance_s_rightturn', 'exit_e'): ['S_dedicated_right_turn', 'right'],
                        ('entrance_s_rightturn', 'exit_s'): ['S_dedicated_right_turn', 'U-turn'],
                        ('entrance_s_rightturn', 'exit_w'): ['S_dedicated_right_turn', 'left'],
                        ('entrance_s_rightturn', 'exit_n'): ['S_dedicated_right_turn', 'through']
                        }

        base_dict_circle = {'right': 0, 'through': 0, 'left': 0, 'U-turn': 0, 'N_dedicated_right_turn': 0, 'S_dedicated_right_turn': 0}

        res_dict = {'entrance_N1': copy.deepcopy(base_dict_circle), 'entrance_N2': copy.deepcopy(base_dict_circle),
                    'entrance_E1': copy.deepcopy(base_dict_circle), 'entrance_E2': copy.deepcopy(base_dict_circle),
                    'entrance_S1': copy.deepcopy(base_dict_circle), 'entrance_S2': copy.deepcopy(base_dict_circle),
                    'entrance_W1': copy.deepcopy(base_dict_circle), 'entrance_W2': copy.deepcopy(base_dict_circle),
                    'N_dedicated_right_turn': copy.deepcopy(base_dict_circle), 'S_dedicated_right_turn': copy.deepcopy(base_dict_circle)}

        for v_entrance in OD_dict.keys():
            for v_exit in OD_dict[v_entrance].keys():
                if (v_entrance, v_exit) not in mapping_dict.keys():
                    print("{0} not in mapping dict!".format((v_entrance, v_exit)))
                    continue
                mapping_val = mapping_dict[(v_entrance, v_exit)]
                res_dict[mapping_val[0]][mapping_val[1]] += OD_dict[v_entrance][v_exit]

        return res_dict

    def _get_entrance_exit(self, region_position_list):
        entrance_list = [val for val in region_position_list if val.split("_")[0] == 'entrance']
        exit_list = [val for val in region_position_list if val.split("_")[0] == 'exit']

        v_entrance = entrance_list[-1] if len(entrance_list) > 0 else None
        v_exit = exit_list[-1] if len(exit_list) > 0 else None

        return v_entrance, v_exit

    def in_circle_traj_avg_speed_analysis_parallel(self):
        traj_speed_list = []
        circle_regions = {'circle_1_q', 'circle_2_q', 'circle_3_q', 'circle_4_q'}

        grouped = self.traj_df.groupby('vid')

        for vid, v_traj in tqdm(grouped, desc='In circle trajectory average speed analysis ...'):
            v_in_circle_traj = v_traj[v_traj['region_position'].isin(circle_regions)]

            if v_in_circle_traj.shape[0] * self.sim_resol <= 1:
                continue

            v_in_circle_traj = v_in_circle_traj.copy()
            v_in_circle_traj["dt"] = v_in_circle_traj["t"].diff()
            v_in_circle_traj["dx"] = v_in_circle_traj["x"].diff()
            v_in_circle_traj["dy"] = v_in_circle_traj["y"].diff()
            v_in_circle_traj["travel_distance"] = (v_in_circle_traj["dx"] ** 2 + v_in_circle_traj["dy"] ** 2) ** 0.5

            if not (v_in_circle_traj["dt"].dropna() > 0).all():
                continue  # skip corrupted or repeated timestamps

            travel_distance = v_in_circle_traj["travel_distance"].sum(skipna=True)
            travel_steps = v_in_circle_traj["dt"].sum(skipna=True)
            travel_time = travel_steps * self.sim_resol

            traj_speed = travel_distance / travel_time
            traj_speed_list.append(traj_speed)

        return traj_speed_list
    
    def in_circle_traj_avg_speed_analysis(self):
        traj_speed_list = []
        for vid in track(self.traj_pool.vehicle_id_list(), description='In circle trajectory average speed analysis ...'):
            v_traj = self.traj_df[self.traj_df['vid'] == vid]

            v_in_circle_traj = v_traj[v_traj['region_position'].isin(['circle_1_q', 'circle_2_q', 'circle_3_q', 'circle_4_q'])]
            if v_in_circle_traj.shape[0] * self.sim_resol <= 1:  # At least 1 second in the circle
                continue

            v_in_circle_traj.loc[:, "dt"] = v_in_circle_traj["t"].diff()
            v_in_circle_traj.loc[:, "dx"] = v_in_circle_traj["x"].diff()
            v_in_circle_traj.loc[:, "dy"] = v_in_circle_traj["y"].diff()
            v_in_circle_traj.loc[:, "travel_distance"] = (v_in_circle_traj["dx"] ** 2 + v_in_circle_traj["dy"] ** 2) ** 0.5

            assert (v_in_circle_traj.dt.dropna() > 0).all()

            travel_distance = v_in_circle_traj.travel_distance.sum(skipna=True)
            travel_steps = v_in_circle_traj.dt.sum(skipna=True)
            travel_time = travel_steps * self.sim_resol

            traj_speed = travel_distance / travel_time
            traj_speed_list.append(traj_speed)

        return traj_speed_list

    def in_circle_instant_speed_and_acceleration_analysis_parallel(self):
        instant_speed_list = []
        acceleration_list = []
        circle_regions = {'circle_1_q', 'circle_2_q', 'circle_3_q', 'circle_4_q'}

        grouped = self.traj_df.groupby('vid')

        for vid, v_traj in tqdm(grouped, desc='In circle instant speed and acceleration analysis ...'):
            v_in_circle_traj = v_traj[v_traj['region_position'].isin(circle_regions)]
            if v_in_circle_traj.shape[0] < 2:
                continue  # can't diff with <2 points

            v_in_circle_traj = v_in_circle_traj.copy()
            v_in_circle_traj["dt"] = v_in_circle_traj["t"].diff()
            v_in_circle_traj["dx"] = v_in_circle_traj["x"].diff()
            v_in_circle_traj["dy"] = v_in_circle_traj["y"].diff()
            v_in_circle_traj["travel_distance"] = (v_in_circle_traj["dx"] ** 2 + v_in_circle_traj["dy"] ** 2) ** 0.5

            if not (v_in_circle_traj["dt"].dropna() > 0).all():
                continue  # skip corrupted or repeated timestamps

            v_in_circle_traj["instant_speed"] = v_in_circle_traj["travel_distance"] / (v_in_circle_traj["dt"] * self.sim_resol)
            instant_speed_tmp = v_in_circle_traj["instant_speed"].dropna().tolist()
            instant_speed_list.extend(instant_speed_tmp)

            v_in_circle_traj["speed_change"] = v_in_circle_traj["instant_speed"].diff()
            v_in_circle_traj["acceleration"] = v_in_circle_traj["speed_change"] / (v_in_circle_traj["dt"] * self.sim_resol)
            acceleration_tmp = v_in_circle_traj["acceleration"].dropna().tolist()
            acceleration_list.extend(acceleration_tmp)

        return instant_speed_list, acceleration_list
    
    def in_circle_instant_speed_and_acceleration_analysis(self):
        instant_speed_list = []
        acceleration_list = []
        for vid in track(self.traj_pool.vehicle_id_list(), description='In circle instant speed and acceleration analysis ...'):
            v_traj = self.traj_df[self.traj_df['vid'] == vid]

            v_in_circle_traj = v_traj[v_traj['region_position'].isin(['circle_1_q', 'circle_2_q', 'circle_3_q', 'circle_4_q'])]
            # if v_in_circle_traj.shape[0] * self.sim_resol <= 1:  # At least 1 second in the circle
            #     continue

            v_in_circle_traj.loc[:, "dt"] = v_in_circle_traj["t"].diff()
            v_in_circle_traj.loc[:, "dx"] = v_in_circle_traj["x"].diff()
            v_in_circle_traj.loc[:, "dy"] = v_in_circle_traj["y"].diff()
            v_in_circle_traj.loc[:, "travel_distance"] = (v_in_circle_traj["dx"] ** 2 + v_in_circle_traj["dy"] ** 2) ** 0.5

            assert (v_in_circle_traj.dt.dropna() > 0).all()

            v_in_circle_traj.loc[:, "instant_speed"] = v_in_circle_traj['travel_distance'] / (v_in_circle_traj['dt'] * self.sim_resol)  # [m/s]
            # instant_speed_tmp = v_in_circle_traj[v_in_circle_traj['dt'] <= 2]['instant_speed'].tolist()
            instant_speed_tmp = v_in_circle_traj['instant_speed'].dropna().tolist()
            instant_speed_list += instant_speed_tmp

            v_in_circle_traj.loc[:, "speed_change"] = v_in_circle_traj["instant_speed"].diff()
            v_in_circle_traj.loc[:, "acceleration"] = v_in_circle_traj['speed_change'] / (v_in_circle_traj['dt'] * self.sim_resol)  # [m/s]
            acceleration_tmp = v_in_circle_traj['acceleration'].dropna().tolist()
            acceleration_list += acceleration_tmp

        return instant_speed_list, acceleration_list

    def all_positions_instant_speed_and_acceleration_analysis(self):
        instant_speed_list = []
        acceleration_list = []
        for vid in track(self.traj_pool.vehicle_id_list(), description='All positions instant speed and acceleration analysis ...'):
            v_traj = self.traj_df[self.traj_df['vid'] == vid]

            if v_traj.shape[0] * self.sim_resol <= 1:  # At least 1 second in the simulation
                continue

            v_traj.loc[:, "dt"] = v_traj["t"].diff()
            v_traj.loc[:, "dx"] = v_traj["x"].diff()
            v_traj.loc[:, "dy"] = v_traj["y"].diff()
            v_traj.loc[:, "travel_distance"] = (v_traj["dx"] ** 2 + v_traj["dy"] ** 2) ** 0.5

            try:
                assert (v_traj.dt.dropna() > 0).all()
            except:
                continue
                # raise ValueError('d')

            v_traj.loc[:, "instant_speed"] = v_traj['travel_distance'] / (v_traj['dt'] * self.sim_resol)  # [m/s]
            # instant_speed_tmp = v_traj[v_traj['dt'] <= 2]['instant_speed'].tolist()
            instant_speed_tmp = v_traj['instant_speed'].dropna().tolist()
            instant_speed_list += instant_speed_tmp

            v_traj.loc[:, "speed_change"] = v_traj["instant_speed"].diff()
            v_traj.loc[:, "acceleration"] = v_traj['speed_change'] / (v_traj['dt'] * self.sim_resol)  # [m/s]
            acceleration_tmp = v_traj['acceleration'].dropna().tolist()
            acceleration_list += acceleration_tmp

        return instant_speed_list, acceleration_list


    def yielding_analysis(self, yielding_speed_thres=2.2352):
        """Yielding probability with and without circulating vehicles.

        Parameters
        ----------
        yielding_speed_thres: [m/s]

        Returns
        -------

        """
        yielding_conflicting_quadrant_mapping = {'yielding_n': 'circle_2_q', 'yielding_e': 'circle_3_q',
                                                 'yielding_s': 'circle_4_q', 'yielding_w': 'circle_1_q'}

        res_dict = {'num_yield_with_conflict': 0, 'num_yield_no_conflict': 0,
                    'num_not_yield_with_conflict': 0, 'num_not_yield_no_conflict': 0}

        for vid in track(self.traj_pool.vehicle_id_list(), description='Yielding analysis ...'):
            v_traj = self.traj_df[self.traj_df['vid'] == vid]
            v_in_yielding_area = v_traj[v_traj['yielding_area'].isin(['yielding_n', 'yielding_e', 'yielding_s', 'yielding_w'])]
            v_yielding_location_list = v_in_yielding_area.yielding_area.unique().tolist()

            if v_in_yielding_area.shape[0] <= 1 or len(v_yielding_location_list) != 1:  # At least 1 time step in the yielding area
                continue

            v_yielding_location = v_yielding_location_list[0]
            conflict_circle_quadrant = yielding_conflicting_quadrant_mapping[v_yielding_location]

            v_in_yielding_area.loc[:, "dt"] = v_in_yielding_area["t"].diff()
            assert (v_in_yielding_area.dt.dropna() > 0).all()
            v_in_yielding_area.loc[:, "dx"] = v_in_yielding_area["x"].diff()
            v_in_yielding_area.loc[:, "dy"] = v_in_yielding_area["y"].diff()
            v_in_yielding_area.loc[:, "travel_distance"] = (v_in_yielding_area["dx"] ** 2 + v_in_yielding_area["dy"] ** 2) ** 0.5
            v_in_yielding_area.loc[:, 'speed'] = v_in_yielding_area['travel_distance']/(v_in_yielding_area['dt'] * self.sim_resol)

            # Whether the SV has yielding behavior
            yield_flag = (v_in_yielding_area['speed'] < yielding_speed_thres).any()

            # Whether there is other vehicles in the left quadrant circle.
            time_interval = v_in_yielding_area['t'].tolist()
            other_v_during_the_interval = self.traj_df[(self.traj_df['t'].isin(time_interval)) & (self.traj_df['vid'] != vid)]
            exist_v_in_conflict_quadrant = (other_v_during_the_interval['region_position'] == conflict_circle_quadrant).any()

            if not yield_flag:
                if exist_v_in_conflict_quadrant:
                    res_dict['num_not_yield_with_conflict'] += 1
                else:
                    res_dict['num_not_yield_no_conflict'] += 1
            else:
                if exist_v_in_conflict_quadrant:
                    res_dict['num_yield_with_conflict'] += 1
                else:
                    res_dict['num_yield_no_conflict'] += 1

        return res_dict

    def yielding_analysis_every_moment(self, yielding_speed_thres=2.2352):
        """Yielding probability with and without circulating vehicles.
        In this yielding definition, each time moment is counted that whether the vehicle is yielding (i.e., speed < yielding speed threshold)

        :param yielding_speed_thres: [m/s]
        :return:
        """
        yielding_conflicting_quadrant_mapping = {'yielding_n': 'circle_2_q', 'yielding_e': 'circle_3_q',
                                                 'yielding_s': 'circle_4_q', 'yielding_w': 'circle_1_q'}

        res_dict = {'num_yield_with_conflict': 0, 'num_yield_no_conflict': 0,
                    'num_not_yield_with_conflict': 0, 'num_not_yield_no_conflict': 0}

        for vid in track(self.traj_pool.vehicle_id_list(), description='Yielding analysis for every moment ...'):
            v_traj = self.traj_df[self.traj_df['vid'] == vid]
            v_in_yielding_area = v_traj[v_traj['yielding_area'].isin(['yielding_n', 'yielding_e', 'yielding_s', 'yielding_w'])]
            v_yielding_location_list = v_in_yielding_area.yielding_area.unique().tolist()

            if v_in_yielding_area.shape[0] <= 1 or len(v_yielding_location_list) != 1:  # At least 1 time step in the yielding area
                continue

            v_yielding_location = v_yielding_location_list[0]
            conflict_circle_quadrant = yielding_conflicting_quadrant_mapping[v_yielding_location]

            v_in_yielding_area.loc[:, "dt"] = v_in_yielding_area["t"].diff()
            assert (v_in_yielding_area.dt.dropna() > 0).all()
            v_in_yielding_area.loc[:, "dx"] = v_in_yielding_area["x"].diff()
            v_in_yielding_area.loc[:, "dy"] = v_in_yielding_area["y"].diff()
            v_in_yielding_area.loc[:, "travel_distance"] = (v_in_yielding_area["dx"] ** 2 + v_in_yielding_area["dy"] ** 2) ** 0.5
            v_in_yielding_area.loc[:, 'speed'] = v_in_yielding_area['travel_distance'] / (v_in_yielding_area['dt'] * self.sim_resol)

            for t in v_in_yielding_area['t'].tolist():
                ego_state_this_step = v_in_yielding_area[v_in_yielding_area['t'] == t]
                ego_speed, ego_x, ego_y = ego_state_this_step.speed.item(), ego_state_this_step.x.item(), ego_state_this_step.y.item()
                if not pd.notna(ego_speed):
                    continue

                # Whether the SV is yielding at the current step
                yield_flag = ego_speed < yielding_speed_thres

                other_v_in_conflict_quadrant = self.traj_df[(self.traj_df['t'] == t) & (self.traj_df['vid'] != vid) & (self.traj_df['region_position'] == conflict_circle_quadrant)]
                exist_v_in_conflict_quadrant = other_v_in_conflict_quadrant.shape[0] > 0

                if not yield_flag:
                    if exist_v_in_conflict_quadrant:
                        res_dict['num_not_yield_with_conflict'] += 1
                    else:
                        res_dict['num_not_yield_no_conflict'] += 1
                else:
                    if exist_v_in_conflict_quadrant:
                        res_dict['num_yield_with_conflict'] += 1
                    else:
                        res_dict['num_yield_no_conflict'] += 1
        return res_dict

    def yilding_distance_and_speed_analysis_parallel(self, yielding_speed_thres=2.2352):
        """Analyze closest conflict vehicle's distance and speed for yielding vehicles."""

        yielding_conflict_dist_and_v_dict = {
            "yield_dist_and_v_list": [],
            "not_yield_dist_and_v_list": []
        }

        quadrant_map = {
            'yielding_n': 'circle_2_q',
            'yielding_e': 'circle_3_q',
            'yielding_s': 'circle_4_q',
            'yielding_w': 'circle_1_q'
        }

        grouped = self.traj_df.groupby('vid')

        for vid, v_traj in tqdm(grouped, desc='Analyzing data...'):
            v_yield = v_traj[v_traj['yielding_area'].isin(quadrant_map.keys())]
            unique_areas = v_yield['yielding_area'].unique()

            if len(v_yield) <= 1 or len(unique_areas) != 1:
                continue

            quadrant = quadrant_map[unique_areas[0]]
            v_yield = v_yield.copy()
            v_yield['dt'] = v_yield['t'].diff()
            v_yield['dx'] = v_yield['x'].diff()
            v_yield['dy'] = v_yield['y'].diff()
            v_yield['travel_distance'] = (v_yield['dx']**2 + v_yield['dy']**2)**0.5
            v_yield['speed'] = v_yield['travel_distance'] / (v_yield['dt'] * self.sim_resol)

            if not (v_yield['dt'].dropna() > 0).all():
                continue

            for t, row in v_yield.iterrows():
                ego_t = row['t']
                ego_speed = row['speed']
                ego_x, ego_y = row['x'], row['y']

                if not pd.notna(ego_speed):
                    continue

                others = self.traj_df.query(
                    "t == @ego_t and vid != @vid and region_position == @quadrant"
                ).copy()

                if others.empty:
                    continue

                others['Euclidean_dist'] = ((others['x'] - ego_x) ** 2 + (others['y'] - ego_y) ** 2) ** 0.5
                closest = others.loc[others['Euclidean_dist'].idxmin()]
                conflict_vid = closest['vid']
                conflict_traj = self.traj_df.query("vid == @conflict_vid and t <= @ego_t")

                if len(conflict_traj) <= 1:
                    continue

                last_two = conflict_traj.iloc[-2:]
                time_diff = last_two.iloc[1]['t'] - last_two.iloc[0]['t']
                if time_diff > 2:
                    continue

                dist = ((last_two.iloc[1]['x'] - last_two.iloc[0]['x']) ** 2 + (last_two.iloc[1]['y'] - last_two.iloc[0]['y']) ** 2) ** 0.5
                conflict_speed = dist / (time_diff * self.sim_resol)

                yield_flag = ego_speed < yielding_speed_thres
                target_list = 'yield_dist_and_v_list' if yield_flag else 'not_yield_dist_and_v_list'
                yielding_conflict_dist_and_v_dict[target_list].append([closest['Euclidean_dist'], conflict_speed])

                self.extract_roundabout_data(conflict_vid, vid)
                self.track_id += 1
                break  # Only analyze first valid step per vid

        return yielding_conflict_dist_and_v_dict
    
    def yilding_distance_and_speed_analysis(self, yielding_speed_thres=2.2352):
        """The Euclidean distance and the speed of the closest vehicle in the circle with the ego-vehicle

        Parameters
        ----------
        yielding_speed_thres = 2.2352

        Returns
        -------

        """
        yielding_conflict_dist_and_v_dict = {"yield_dist_and_v_list": [], "not_yield_dist_and_v_list": []}  # [[dist, v], ...], unit: [m, m/s].

        yielding_conflicting_quadrant_mapping = {'yielding_n': 'circle_2_q', 'yielding_e': 'circle_3_q',
                                                 'yielding_s': 'circle_4_q', 'yielding_w': 'circle_1_q'}

        for vid in track(self.traj_pool.vehicle_id_list(), description='Analyzing data...'):
            v_traj = self.traj_df[self.traj_df['vid'] == vid]
            v_in_yielding_area = v_traj[v_traj['yielding_area'].isin(['yielding_n', 'yielding_e', 'yielding_s', 'yielding_w'])]
            v_yielding_location_list = v_in_yielding_area.yielding_area.unique().tolist()

            if v_in_yielding_area.shape[0] <= 1 or len(v_yielding_location_list) != 1:  # At least 1 time step in the yielding area
                continue

            v_yielding_location = v_yielding_location_list[0]
            conflict_circle_quadrant = yielding_conflicting_quadrant_mapping[v_yielding_location]

            v_in_yielding_area.loc[:, "dt"] = v_in_yielding_area["t"].diff().astype(float)
            assert (v_in_yielding_area.dt.dropna() > 0).all()
            v_in_yielding_area.loc[:, "dx"] = v_in_yielding_area["x"].diff()
            v_in_yielding_area.loc[:, "dy"] = v_in_yielding_area["y"].diff()
            v_in_yielding_area.loc[:, "travel_distance"] = (v_in_yielding_area["dx"] ** 2 + v_in_yielding_area["dy"] ** 2) ** 0.5
            v_in_yielding_area.loc[:, 'speed'] = v_in_yielding_area['travel_distance']/(v_in_yielding_area['dt'] * self.sim_resol)

            for t in v_in_yielding_area['t'].tolist():
                ego_state_this_step = v_in_yielding_area[v_in_yielding_area['t'] == t]
                ego_speed, ego_x, ego_y = ego_state_this_step.speed.item(), ego_state_this_step.x.item(), ego_state_this_step.y.item()
                if not pd.notna(ego_speed):
                    continue

                other_v_in_conflict_quadrant = self.traj_df[(self.traj_df['t'] == t) & (self.traj_df['vid'] != vid) & (self.traj_df['region_position'] == conflict_circle_quadrant)]
                if other_v_in_conflict_quadrant.shape[0] == 0:  # No other vehicles in the conflict quadrant
                    continue
                other_v_in_conflict_quadrant['Euclidean_dist'] = ((other_v_in_conflict_quadrant['x'] - ego_x) ** 2 + (other_v_in_conflict_quadrant['y'] - ego_y) ** 2) ** 0.5
                closest_other_v = other_v_in_conflict_quadrant.loc[other_v_in_conflict_quadrant.Euclidean_dist.idxmin()]

                # Calculate closest other vehicle speed
                conflict_v_id = closest_other_v.vid
                conflict_v_prev_traj = self.traj_df[(self.traj_df['vid'] == conflict_v_id) & (self.traj_df['t'] <= t)]
                if conflict_v_prev_traj.shape[0] <= 1:
                    continue
                travel_dist = ((conflict_v_prev_traj.iloc[-1]['x'] - conflict_v_prev_traj.iloc[-2]['x']) ** 2 + (conflict_v_prev_traj.iloc[-1]['y'] - conflict_v_prev_traj.iloc[-2]['y']) ** 2) ** 0.5
                travel_time = conflict_v_prev_traj.iloc[-1]['t'] - conflict_v_prev_traj.iloc[-2]['t']
                if travel_time > 2:
                    continue
                conflict_v_speed = travel_dist / (travel_time * self.sim_resol)
                conflict_v_dist = closest_other_v.Euclidean_dist

                # Whether the SV is yielding at the current step
                yield_flag = ego_state_this_step['speed'].item() < yielding_speed_thres
                if yield_flag:
                    yielding_conflict_dist_and_v_dict['yield_dist_and_v_list'].append([conflict_v_dist, conflict_v_speed])
                else:
                    yielding_conflict_dist_and_v_dict['not_yield_dist_and_v_list'].append([conflict_v_dist, conflict_v_speed])

                self.extract_roundabout_data(conflict_v_id, vid)
                break

        return yielding_conflict_dist_and_v_dict

    def extract_roundabout_data(self, inner_id, outer_id):
        if [inner_id, outer_id] not in self.extracted_veh_pairs:
            inner_path = []
            outer_path = []
            for vehicles in self.TIME_BUFF:
                inner_flag = False
                outer_flag = False
                inner_vehicle = None
                outer_vehicle = None
                for vehicle in vehicles:
                    if inner_id == vehicle.id:
                        inner_flag = True
                        inner_vehicle = copy.deepcopy(vehicle)
                    if outer_id == vehicle.id:
                        outer_flag = True
                        outer_vehicle = copy.deepcopy(vehicle)
                if inner_flag and outer_flag:
                    inner_path.append(inner_vehicle)
                    outer_path.append(outer_vehicle)
            if len(inner_path) > 5 and len(outer_path) > 5:
                inner_conflict_point = None
                outer_conflict_point = None
                min_dis = 10000
                for inner_veh in inner_path:
                    for outer_veh in outer_path:
                        dis = math.sqrt((inner_veh.location.x - outer_veh.location.x)**2 + (inner_veh.location.y - outer_veh.location.y)**2)
                        if dis < min_dis:
                            min_dis = copy.deepcopy(dis)
                            inner_conflict_point = copy.deepcopy(inner_veh)
                            outer_conflict_point = copy.deepcopy(outer_veh)
                if min_dis < 1.5:
                    # plt.plot([vehicle.location.x for vehicle in inner_path], [vehicle.location.y for vehicle in inner_path], 'r')
                    # plt.plot([vehicle.location.x for vehicle in outer_path], [vehicle.location.y for vehicle in outer_path], 'g')
                    # plt.axis('equal')
                    # plt.show()
                    inner_dis, inner_sp, inner_acc = extract_key_variables(self.track_id, inner_path)
                    outer_dis, outer_sp, outer_acc = extract_key_variables(self.track_id, outer_path)
                    self.acc = np.vstack((self.acc, np.stack([inner_acc, outer_acc], axis=1)))
                    self.dis_sp.append([inner_dis, inner_sp, outer_dis, outer_sp])
                    self.extracted_veh_pairs.append([inner_id, outer_id])
                else:
                    outer_point_1 = [outer_path[-2].location.x, outer_path[-2].location.y]
                    outer_point_2 = [outer_path[-1].location.x, outer_path[-1].location.y]
                    flag = False
                    for ind in range(len(inner_path)-1):
                        inner_point_1 = [inner_path[ind].location.x, inner_path[ind].location.y]
                        inner_point_2 = [inner_path[ind+1].location.x, inner_path[ind+1].location.y]
                        if not determine_two_points_on_the_same_side(inner_point_1, inner_point_2, outer_point_1, outer_point_2):
                            flag = True
                            break
                    inner_point_1 = [inner_path[-2].location.x, inner_path[-2].location.y]
                    inner_point_2 = [inner_path[-1].location.x, inner_path[-1].location.y]
                    flag = False
                    for ind in range(len(outer_path)-1):
                        outer_point_1 = [outer_path[ind].location.x, outer_path[ind].location.y]
                        outer_point_2 = [outer_path[ind+1].location.x, outer_path[ind+1].location.y]
                        if not determine_two_points_on_the_same_side(outer_point_1, outer_point_2, inner_point_1, inner_point_2):
                            if determine_point_ahead_of_points(outer_point_1, inner_point_1, inner_point_2):
                                flag = True
                                break
                    if flag:
                        inner_dis, inner_sp, inner_acc = extract_key_variables(self.track_id, inner_path)
                        outer_dis, outer_sp, outer_acc = extract_key_variables(self.track_id, outer_path)
                        self.acc = np.vstack((self.acc, np.stack([inner_acc, outer_acc], axis=1)))
                        self.dis_sp.append([inner_dis, inner_sp, outer_dis, outer_sp])
                        self.extracted_veh_pairs.append([inner_id, outer_id])

        return 0

    def range_analysis(self):
        """
        Distance between vehicles in entrance region.
        Returns
        -------

        """
        range_list = []
        for t_step in self.traj_df.t.unique().tolist():
            traj_df_at_t_step = self.traj_df[self.traj_df['t'] == t_step]

            for entrance_pos in ['entrance_n_1', 'entrance_n_2',
                                 'entrance_e_1', 'entrance_e_2',
                                 'entrance_s_1', 'entrance_s_2',
                                 'entrance_w_1', 'entrance_w_2']:
                v_in_entrance_pos = traj_df_at_t_step[traj_df_at_t_step['region_position'] == entrance_pos]

                if v_in_entrance_pos.shape[0] <= 1:
                    continue

                num_v = v_in_entrance_pos.shape[0]

                # All pairwise distances
                dist_tmp = []
                for idx_pair in combinations(range(v_in_entrance_pos.shape[0]), r=2):
                    v1 = v_in_entrance_pos.iloc[idx_pair[0]]
                    v2 = v_in_entrance_pos.iloc[idx_pair[1]]

                    dist = np.linalg.norm([v1.x - v2.x, v1.y - v2.y])
                    dist_tmp.append(dist)

                # The first k is the range
                dist_tmp.sort()
                range_list += dist_tmp[:num_v-1]

        return range_list

    def _verify_three_circle_approximation(self, traj_df_at_t_step, radius):
        """
        This function is to plot the vehicle shape and three circle approximations.
        """
        from shapely.geometry import Polygon, Point
        import matplotlib.pyplot as plt
        v = traj_df_at_t_step.iloc[0]
        polygon = v['vehicle'].poly_box
        circle_c = Point(v['center_circle_x'], v['center_circle_y']).buffer(radius)
        circle_f = Point(v['front_circle_x'], v['front_circle_y']).buffer(radius)
        circle_r = Point(v['rear_circle_x'], v['rear_circle_y']).buffer(radius)
        plt.plot(*polygon.exterior.xy)
        plt.plot(*circle_c.exterior.xy, label='c')
        plt.plot(*circle_f.exterior.xy, label='f')
        plt.plot(*circle_r.exterior.xy, label='r')
        plt.plot([v['center_circle_x'] + dist * np.cos(np.radians(v['heading'])) for dist in np.linspace(0., 5., num=100)],
                 [v['center_circle_y'] + dist * np.sin(np.radians(v['heading'])) for dist in np.linspace(0., 5., num=100)], label='heading')
        plt.legend()
        plt.axis('equal')
        plt.show()

    def distance_analysis(self, mode='single_circle', only_in_roundabout_circle=False):
        """
        mode:
            - 'single_circle': calculate the Euclidean distance of vehicle center.
            - 'three_circle': approximate each vehicle using three circles and calculate the closest distance between vehicles' circle center.

        Calculate the Euclidean distance between any pair of vehicles
        """
        if mode not in ['single_circle', 'three_circle']:
            raise ValueError("{0} not supported for distance analysis, choose from [center_distance, three_circle].".format(mode))

        distance_list = []

        if mode == 'single_circle':
            for t_step in self.traj_df.t.unique().tolist():
                traj_df_at_t_step = self.traj_df[self.traj_df['t'] == t_step]
                if only_in_roundabout_circle:
                    traj_df_at_t_step = traj_df_at_t_step[traj_df_at_t_step['region_position'].isin(['circle_1_q', 'circle_2_q', 'circle_3_q', 'circle_4_q'])]
                pos_list = [[val[0], val[1]] for val in zip(traj_df_at_t_step.x.tolist(), traj_df_at_t_step.y.tolist())]
                distance_list_at_t_step = list(pdist(pos_list, metric='euclidean'))  # pairwise distance of all vehicles.
                distance_list += distance_list_at_t_step

        if mode == "three_circle":
            radius, center_point_distance = 1.0, 2.7  # the radius of each circle, the distance between the front and rear circles.
            for t_step in self.traj_df.t.unique().tolist():
                traj_df_at_t_step = self.traj_df[self.traj_df['t'] == t_step]
                if only_in_roundabout_circle:
                    traj_df_at_t_step = traj_df_at_t_step[traj_df_at_t_step['region_position'].isin(['circle_1_q', 'circle_2_q', 'circle_3_q', 'circle_4_q'])]
                traj_df_at_t_step["center_circle_x"] = traj_df_at_t_step["x"]
                traj_df_at_t_step["center_circle_y"] = traj_df_at_t_step["y"]
                traj_df_at_t_step["front_circle_x"] = traj_df_at_t_step["x"] + (center_point_distance / 2) * np.cos(np.radians(traj_df_at_t_step["heading"]))
                traj_df_at_t_step["front_circle_y"] = traj_df_at_t_step["y"] + (center_point_distance / 2) * np.sin(np.radians(traj_df_at_t_step["heading"]))
                traj_df_at_t_step["rear_circle_x"] = traj_df_at_t_step["x"] - (center_point_distance / 2) * np.cos(np.radians(traj_df_at_t_step["heading"]))
                traj_df_at_t_step["rear_circle_y"] = traj_df_at_t_step["y"] - (center_point_distance / 2) * np.sin(np.radians(traj_df_at_t_step["heading"]))

                # Loop through all vehicles and calculate distance with other vehicles.
                for row_idx in range(traj_df_at_t_step.shape[0] - 1):
                    dis_v = []

                    v_info = traj_df_at_t_step.iloc[row_idx]
                    other_vs = traj_df_at_t_step.iloc[row_idx+1:]

                    for ego_v_circle_pos in ['center_circle', 'front_circle', 'rear_circle']:
                        x_name, y_name = '_'.join([ego_v_circle_pos, 'x']), '_'.join([ego_v_circle_pos, 'y'])
                        v_pos = np.array([[v_info[x_name], v_info[y_name]]])
                        for other_v_circle_pos in ['center_circle', 'front_circle', 'rear_circle']:
                            x_name, y_name = '_'.join([other_v_circle_pos, 'x']), '_'.join([other_v_circle_pos, 'y'])
                            pos_array = np.array([[val[0], val[1]] for val in zip(other_vs[x_name].tolist(), other_vs[y_name].tolist())])
                            if pos_array.shape[0] == 0:
                                continue
                            dis = np.linalg.norm(v_pos - pos_array, axis=1)
                            dis_v.append(list(dis))

                    dis_v = np.array(dis_v)
                    distance_the_vehicle = list(dis_v.min(axis=0))  # The minimum distance with each other vehicle
                    distance_list += distance_the_vehicle

        return distance_list

    def density_analysis(self, moving_window_wall_time=10):
        """
        Average number of vehicles within the map within a moving time window (e.g., 5s)

        Returns
        -------

        """
        num_steps = int(moving_window_wall_time / self.sim_resol)
        max_t_step = self.traj_df.t.max()
        num_of_vehs_list = []
        for t_step in self.traj_df.t.unique().tolist():

            if t_step + num_steps >= max_t_step:
                continue

            num_of_vehs_in_moving_window = []
            for t_in_window in range(t_step, t_step + num_steps):
                veh_num = (self.traj_df[self.traj_df['t'] == t_in_window]).shape[0]
                num_of_vehs_in_moving_window.append(veh_num)

            avg_num_of_vehs = np.mean(num_of_vehs_in_moving_window)
            num_of_vehs_list.append(avg_num_of_vehs)

        return num_of_vehs_list

    def collision_analysis(self, TIME_BUFF, extra_buffer=False):
        total_steps, collision_steps = 0, 0
        for vehicle_list in TIME_BUFF:
            total_steps += len(vehicle_list)
            for vehicle_pair in combinations(vehicle_list, r=2):
                v1, v2 = vehicle_pair[0], vehicle_pair[1]
                if extra_buffer:
                    v1_poly, v2_poly = v1.safe_poly_box, v2.safe_poly_box
                else:
                    v1_poly, v2_poly = v1.poly_box, v2.poly_box

                if v1_poly.intersects(v2_poly):
                    collision_steps += 2
                    break
        collision_rate = collision_steps / total_steps

        return total_steps, collision_steps, collision_rate

    def offroad_analysis(self):
        total_steps, off_road_steps = 0, 0
        for vid in self.traj_pool.vehicle_id_list():
            v_traj = self.traj_df[self.traj_df['vid'] == vid]

            buff_x_flatten, buff_y_flatten = v_traj.x.tolist(), v_traj.y.tolist()

            if len(buff_x_flatten) == 0:  # No vehicle in this frame
                continue

            pxl_pts_all = self.road_matcher._world2pxl(np.array([buff_x_flatten, buff_y_flatten]).T)

            if pxl_pts_all.size == 2:  # Only 1 vehicle in the sim, the shape will become (2,)
                pxl_pts_all = pxl_pts_all.reshape(-1, 2)
            x0_list, y0_list = pxl_pts_all[:, 0].tolist(), pxl_pts_all[:, 1].tolist()

            pxl_color_val = self.road_matcher.road_map[y0_list, x0_list]
            on_the_road = pxl_color_val > 128.

            all_steps_this_vehicle = len(pxl_color_val)
            on_the_road_steps_this_vehicle = np.sum(on_the_road)
            offroad_steps_this_vehicle = all_steps_this_vehicle - on_the_road_steps_this_vehicle

            total_steps += all_steps_this_vehicle
            off_road_steps += offroad_steps_this_vehicle

        orr = off_road_steps / total_steps  # offroad rate

        return total_steps, off_road_steps, orr

    def near_miss_detection_acceleration_based(self):
        save_t_list = []  # the time step that have near-miss happens
        detect_v_list = []

        # 1. first check whether the vehicle has large deceleration, 2. then check whether it is very close to other vehicles within a time window.
        deceleration_thershold = -8  # unit: m/s^2.
        multiple_low_acc_number = 2  # consecutive low acc situation within 2s.
        distance_threshold = 4  # unit: m. Euclidean distance of mass center of two vehicles.

        for vid in self.traj_pool.vehicle_id_list():

            v_traj = self.traj_df[self.traj_df['vid'] == vid]
            if v_traj.shape[0] * self.sim_resol <= 1:  # At least 1 second
                continue

            # step 1: check deceleration
            v_traj.loc[:, "dt"] = v_traj["t"].diff()
            v_traj.loc[:, "dx"] = v_traj["x"].diff()
            v_traj.loc[:, "dy"] = v_traj["y"].diff()
            v_traj.loc[:, "travel_distance"] = (v_traj["dx"] ** 2 + v_traj["dy"] ** 2) ** 0.5

            assert (v_traj.dt.dropna() > 0).all()

            v_traj.loc[:, "instant_speed"] = v_traj['travel_distance'] / (v_traj['dt'] * self.sim_resol)  # [m/s]

            v_traj.loc[:, "speed_change"] = v_traj["instant_speed"].diff()
            v_traj.loc[:, "acceleration"] = v_traj['speed_change'] / (v_traj['dt'] * self.sim_resol)  # [m/s]

            if not (v_traj["acceleration"] < deceleration_thershold).any():
                continue

            # step 2: check distance with other vehicles when the deceleration is very large.
            # time step that the deceleration is harder than the threshold
            time_list = v_traj.loc[v_traj['acceleration'] < deceleration_thershold, 't'].tolist()
            for t in time_list:
                time_interval = [t + val for val in range(-2, 3)]  # 1s before and after the large deceleration
                other_traj = self.traj_df[(self.traj_df['t'].isin(time_interval)) & (self.traj_df['vid'] != vid)]
                v_pos = np.array([[v_traj.loc[v_traj['t'] == t, 'x'].item(), v_traj.loc[v_traj['t'] == t, 'y'].item()]])
                pos_array = np.array([[val[0], val[1]] for val in zip(other_traj.x.tolist(), other_traj.y.tolist())])
                if pos_array.shape[0] == 0:
                    continue
                dis = np.linalg.norm(v_pos - pos_array, axis=1)
                distance_with_other_v_flag = (dis < distance_threshold).any()

                time_interval = [t + val for val in range(-5, 1)]  # 2s before the small deceleration
                self_traj = v_traj[v_traj['t'].isin(time_interval)]
                multiple_low_acc_flag = ((self_traj["acceleration"] < deceleration_thershold).sum() >= multiple_low_acc_number)

                if distance_with_other_v_flag and multiple_low_acc_flag:
                    save_t_list.append(t)
                    detect_v_list.append(vid)

        return save_t_list, detect_v_list

    def near_miss_detection_speed_based(self):
        save_t_list = []  # the time step that have near-miss happens
        detect_v_list = []

        # 1. first check whether the vehicle has small velocity, 2. then check whether deceleration, distance, etc., satisfy requirements within a time window.
        instant_speed_thershold = 2  # unit: m/s
        multiple_low_speed_number = 2  # consecutive low speed situation within 2s.
        distance_threshold = 3.5  # unit: m. Euclidean distance of mass center of two vehicles.
        travel_distance_threshold = 8  # unit: m. Euclidean distance that the vehicle needs to travel.
        deceleration_thershold = 0  # unit: m/s^2.

        for vid in self.traj_pool.vehicle_id_list():

            v_traj = self.traj_df[self.traj_df['vid'] == vid]
            if v_traj.shape[0] * self.sim_resol <= 1:  # At least 1 second
                continue

            # step 1: check deceleration
            v_traj.loc[:, "dt"] = v_traj["t"].diff()
            v_traj.loc[:, "dx"] = v_traj["x"].diff()
            v_traj.loc[:, "dy"] = v_traj["y"].diff()
            v_traj.loc[:, "travel_distance"] = (v_traj["dx"] ** 2 + v_traj["dy"] ** 2) ** 0.5

            assert (v_traj.dt.dropna() > 0).all()

            v_traj.loc[:, "instant_speed"] = v_traj['travel_distance'] / (v_traj['dt'] * self.sim_resol)  # [m/s]

            v_traj.loc[:, "speed_change"] = v_traj["instant_speed"].diff()
            v_traj.loc[:, "acceleration"] = v_traj['speed_change'] / (v_traj['dt'] * self.sim_resol)  # [m/s]

            # if not (v_traj["acceleration"] < deceleration_thershold).any():
            #     continue
            if not (v_traj["instant_speed"] < instant_speed_thershold).any():
                continue

            # step 2: check whether deceleration, distance, etc., satisfy requirements within a time window.
            # time step that the speed is smaller than the threshold
            time_list = v_traj.loc[v_traj['instant_speed'] < instant_speed_thershold, 't'].tolist()
            for t in time_list:
                time_interval = [t + val for val in range(-5, 1)]  # 2s before the small speed
                self_traj = v_traj[v_traj['t'].isin(time_interval)]
                travel_distance = self_traj.travel_distance.sum()
                travel_distance_flag = travel_distance > travel_distance_threshold
                deceleration_flag = (self_traj['acceleration'] < deceleration_thershold).any()

                multiple_low_speed_flag = ((self_traj["instant_speed"] < instant_speed_thershold).sum() >= multiple_low_speed_number)

                time_interval = [t + val for val in range(-2, 3)]  # 1s before and after the large deceleration
                other_traj = self.traj_df[(self.traj_df['t'].isin(time_interval)) & (self.traj_df['vid'] != vid)]
                v_pos = np.array([[v_traj.loc[v_traj['t'] == t, 'x'].item(), v_traj.loc[v_traj['t'] == t, 'y'].item()]])
                pos_array = np.array([[val[0], val[1]] for val in zip(other_traj.x.tolist(), other_traj.y.tolist())])
                if pos_array.shape[0] == 0:
                    continue
                dis = np.linalg.norm(v_pos - pos_array, axis=1)
                distance_flag = (dis < distance_threshold).any()

                if travel_distance_flag and deceleration_flag and distance_flag and multiple_low_speed_flag:
                    save_t_list.append(t)
                    detect_v_list.append(vid)

        return save_t_list, detect_v_list

    def PET_analysis(self, background_map=None):

        # occupancy_res = [ndarray, ndarray,...] each ndarray in it is the occupancy results of each position at the time step i.
        # each ndarray is height_n * width_n, where each cell is the vehicle id that occupied the position at that time step.
        occupancy_res = []
        for t in track(range(len(self.TIME_BUFF)), description='PET analysis ...'):
            occupancy_at_t_step = np.zeros((self.PET_configs['height_n'], self.PET_configs['width_n']), dtype=object)
            occupancy_at_t_step[:, :] = 'empty'  # initialize the occupancy ndarray where no vehicles any position.

            vehicle_list_at_t_step = self.TIME_BUFF[t]
            for v in vehicle_list_at_t_step:
                # Check all 5 points of the vehicle (center, each corner point of the bounding box)
                v_center_pos = [v.location.x, v.location.y]
                v_pt1_pos, v_pt2_pos, v_pt3_pos, v_pt4_pos = v.realworld_4_vertices
                v_all_pos = [list(v_center_pos), list(v_pt1_pos), list(v_pt2_pos), list(v_pt3_pos), list(v_pt4_pos)]
                v_all_pos = self.road_matcher._world2pxl(v_all_pos)  # Transform to pxl coordinates
                for pos in v_all_pos:
                    # Check whether the position is within the grid area
                    if not ((pos[1] >= self.PET_configs['height_end']) or (pos[1] <= self.PET_configs['height_start']) \
                            or (pos[0] <= self.PET_configs['width_start']) or (pos[0] >= self.PET_configs['width_end'])):
                        height_idx, width_idx = int(divmod((pos[1] - self.PET_configs['height_start']), self.PET_configs['height_res'])[0]), \
                                                int(divmod((pos[0] - self.PET_configs['width_start']), self.PET_configs['width_res'])[0])

                        if occupancy_at_t_step[height_idx, width_idx] == 'empty':
                            occupancy_at_t_step[height_idx, width_idx] = v.id
                        else:
                            # If there are multiple vehicles occupying a same location (the grid might be too sparse), randomly determine which vehicle is occupying.
                            if np.random.uniform() > 0.5:
                                occupancy_at_t_step[height_idx, width_idx] = v.id

            occupancy_res.append(occupancy_at_t_step)
            # self._verify_and_visualize_PET_occupancy_map(occupancy_at_t_step, vehicle_list_at_t_step, background_map)
        occupancy_res = np.stack(occupancy_res)  # time * height_n * width_n

        # Loop over all positions to calculate the PET on it along the time
        PET_list = []
        for height_idx in range(occupancy_res.shape[1]):
            for width_idx in range(occupancy_res.shape[2]):
                occupancy_at_this_position = occupancy_res[:, height_idx, width_idx]
                PET_at_this_position = []
                last_occupied_t, last_occupied_vid = None, 'empty'
                for t_idx in range(occupancy_res.shape[0]):
                    current_occupied_t, current_occupied_vid = t_idx, occupancy_at_this_position[t_idx]
                    if current_occupied_vid == 'empty':  # No vehicle on this position at this moment, nothing needs to do.
                        continue
                    if last_occupied_vid != 'empty' and last_occupied_vid != current_occupied_vid:  # There is a new vehicle occupy this position.
                        PET = (current_occupied_t - last_occupied_t) * self.sim_resol  # The PET
                        PET_at_this_position.append(PET)
                        last_occupied_t, last_occupied_vid = current_occupied_t, current_occupied_vid
                    if last_occupied_vid == 'empty' or last_occupied_vid == current_occupied_vid:  # There is a vehicle first occupy this position.
                        last_occupied_t, last_occupied_vid = current_occupied_t, current_occupied_vid
                PET_list += PET_at_this_position

        return PET_list

    def _verify_and_visualize_PET_occupancy_map(self, occupancy_at_t_step, vehicle_list, background_map):
        import cv2
        import matplotlib.pyplot as plt

        map_test = self.PET_configs['basemap_img'].copy()
        for height_idx in range(occupancy_at_t_step.shape[0]):
            for width_idx in range(occupancy_at_t_step.shape[1]):
                if occupancy_at_t_step[height_idx, width_idx] != 'empty':
                    width_lb, width_ub = int(self.PET_configs['width_start'] + width_idx * self.PET_configs['width_res']), \
                                         int(self.PET_configs['width_start'] + (width_idx + 1) * self.PET_configs['width_res'])
                    height_lb, height_ub = int(self.PET_configs['height_start'] + height_idx * self.PET_configs['height_res']), \
                                           int(self.PET_configs['height_start'] + (height_idx + 1) * self.PET_configs['height_res'])
                    map_test[height_lb:height_ub, width_lb:width_ub, :] = (255, 255, 255)
        # Plot the occupany map
        plt.figure(figsize=(16, 9))
        plt.imshow(map_test)
        plt.show()

        # Plot the vehicle on the road fig
        vis = background_map.render(vehicle_list, with_traj=True, linewidth=6)
        img = vis[:, :, ::-1]
        # img = cv2.resize(img, (768, int(768 * background_map.h / background_map.w)))  # resize when needed
        cv2.imshow('vis', img)  # rgb-->bgr
        cv2.waitKey(0)

    @staticmethod
    def _verify_and_visualize_grid_map_for_PET_analysis(basemap_img, width_start, width_end, width_n, width_res,
                                                        height_start, height_end, height_n, height_res):
        """
        This function is to used to determine and visualize the grid mesh settings.
        The transparent color blocks are grid area.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(16, 9))

        for height_idx in range(height_n):
            for width_idx in range(width_n):
                width_lb, width_ub = int(width_start + width_idx * width_res), int(width_start + (width_idx + 1) * width_res)
                height_lb, height_ub = int(height_start + height_idx * height_res), int(height_start + (height_idx + 1) * height_res)
                basemap_img[height_lb:height_ub, width_lb:width_ub, :] += (0.3 * np.random.uniform(0, 255, (1, 3))).astype('uint8')


        plt.imshow(basemap_img)
        plt.show()


if __name__ == '__main__':
    import pickle
    from basemap import Basemap
    import cv2
    import matplotlib.pyplot as plt

    path_to_traj_data = r'./test_dataset'

    drivable_map_dir = r'../road_matching/drivablemap/rounD-drivablemap.jpg'
    sim_remove_vehicle_area_map = r'../road_matching/ROIs-map/rounD-sim-remove-vehicle-area-map.jpg'

    # ROIs
    circle_map_dir = os.path.join(r'../road_matching/ROIs-map/region-matcher-maps/circle')
    entrance_map_dir = os.path.join(r'../road_matching/ROIs-map/region-matcher-maps/entrance')
    exit_map_dir = os.path.join(r'../road_matching/ROIs-map/region-matcher-maps/exit')
    crosswalk_map_dir = os.path.join(r'../road_matching/ROIs-map/region-matcher-maps/crosswalk')
    yielding_area_map_dir = os.path.join(r'../road_matching/ROIs-map/region-matcher-maps/yielding-area')

    SimMetricsAnalyzer = RealisticMetrics(drivable_map_dir=drivable_map_dir, sim_remove_vehicle_area_map=sim_remove_vehicle_area_map,
                                          circle_map_dir=circle_map_dir, entrance_map_dir=entrance_map_dir, exit_map_dir=exit_map_dir, crosswalk_map_dir=crosswalk_map_dir, yielding_area_map_dir=yielding_area_map_dir, sim_resol=0.4)

    TIME_BUFF = []
    traj_dirs = os.listdir(os.path.join(path_to_traj_data))
    for i in range(0, len(traj_dirs)):
        vehicle_list = pickle.load(open(os.path.join(path_to_traj_data, traj_dirs[i]), "rb"))
        TIME_BUFF.append(vehicle_list)

    SimMetricsAnalyzer.construct_traj_data(TIME_BUFF)

    # OD analysis
    OD_dict, readable_OD_dict, traj_num, complete_traj_num = SimMetricsAnalyzer.OD_analysis()
    print(OD_dict)
    print(readable_OD_dict)
    print("==== traj_num: {0}, complete_traj_num: {1}".format(traj_num, complete_traj_num))
    print()

    # in circle trajectory average speed analysis
    traj_speed_list = SimMetricsAnalyzer.in_circle_traj_avg_speed_analysis()
    print(traj_speed_list)
    print('==== Mean traj speed: {0} m/s, {1} mph'.format(np.mean(traj_speed_list), np.mean(traj_speed_list) * 2.23694))
    print()
    # plt.hist(traj_speed_list)
    # plt.show()

    # in circle instant speed analysis
    instant_speed_list, acceleration_list = SimMetricsAnalyzer.in_circle_instant_speed_and_acceleration_analysis()
    print(instant_speed_list)
    # plt.hist(instant_speed_list)
    # plt.show()
    print()

    # density distribution analysis
    num_of_vehs_list = SimMetricsAnalyzer.density_analysis(moving_window_wall_time=10)
    print(num_of_vehs_list)
    # plt.hist(num_of_vehs_list)
    # plt.show()

    # background_map = Basemap(map_file_dir=r'../basemap/rounD-official-map.svg', map_height=624, map_width=1119)
    # video_writer = cv2.VideoWriter(r'demo.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 2, (1119, 624))
    # visualize_time_buff(TIME_BUFF, background_map)

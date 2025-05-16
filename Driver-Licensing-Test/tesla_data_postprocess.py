import os
import pandas as pd
import utm
import matplotlib.pyplot as plt
import math
from rich.progress import track
import copy

from utils import utm_to_sumo_coordinate, get_sec, cal_dis, save_csv, project_point_on_line


if __name__ == '__main__':
    test_name = 'Tesla'
    path = 'output/' + test_name

    ori_data_path = path + '/original_test_data'
    data_path = path + '/test_data/test_round_1'
    os.makedirs(data_path, exist_ok=True)

    folders = os.listdir(ori_data_path)
    print(folders)

    traj_keys = ['timestamp', 'AV x', 'AV y', 'AV sp', 'AV lon sp', 'AV lat sp', 'AV acc', 'AV lon acc', 'AV lat acc', 'AV heading',
                'challenger x', 'challenger y', 'challenger sp', 'challenger lon sp', 'challenger lat sp', 'challenger acc', 'challenger lon acc', 'challenger lat acc', 'challenger heading']

    for folder in folders:
        test_record_file = ori_data_path + '/' + folder + '/' + folder + '_test_record.xlsx'
        ori_test_record = pd.read_excel(test_record_file)
        ori_test_record_keys = ori_test_record.columns
        ori_case_id_ind = ori_test_record_keys.get_loc('Case id')

        test_record = []
        for line in ori_test_record.values:
            if line[-1] in ['Yes', 'yes', 'Y', 'y']:
                test_record.append(line)

        ori_traj_files = os.listdir(ori_data_path + '/' + folder + '/data')
        ori_traj_files.sort()
        record_keys = []
        record = []
        for line in track(test_record, description=folder):
            try:
                ori_case_id = str(int(line[ori_case_id_ind]))
            except:
                ori_case_id = line[ori_case_id_ind]
            collision_flag = line[-3]
            if collision_flag in ['Yes', 'yes', 'Y', 'y']:
                collision_flag = True
            else:
                collision_flag = False
            scenario = line[0]
            traj_path = data_path + '/' + scenario
            test_time = line[1]
            if int(test_time) > 1:
                traj_path += str(test_time)
            if not os.path.exists(traj_path):
                os.makedirs(traj_path)
            risk_level_ind = ori_test_record_keys.get_loc('risk level')
            risk_level = line[risk_level_ind]
            start_time_ind = ori_test_record_keys.get_loc('start time')

            if len(record_keys) == 0:
                record_keys = ['case id', 'risk level']
                for key in list(ori_test_record_keys[risk_level_ind + 1: start_time_ind]):
                    record_keys.append(key)
                record_keys += ['init timestamp', 'dis err', 'sp err', 'AV init sp', 'collision']

            if ori_case_id.isdigit():
                case_id = int(ori_case_id)
            else:
                case_id = int(ori_case_id[:-1])
            value = [case_id, risk_level]
            for tmp in list(line[risk_level_ind + 1: start_time_ind]):
                value.append(tmp)
            for ori_traj_file in ori_traj_files:
                if '_' + ori_case_id + '.' in ori_traj_file and scenario in ori_traj_file:
                    ori_traj_file_path = ori_data_path + '/' + folder + '/data/' + ori_traj_file
                    with open(ori_traj_file_path, 'r') as f:
                        ori_traj_data = f.readlines()
                    for data in ori_traj_data:
                        if 'Test point data' in data:
                            start_ind = ori_traj_data.index(data) + 3
                            break
                    keys = ori_traj_data[start_ind - 1].split('\t')

                    traj_data = []
                    AV_x_list = []
                    AV_y_list = []
                    AV_sp_list = []
                    AV_heading_list = []
                    challenger_x_list = []
                    challenger_y_list = []
                    challenger_sp_list = []
                    challenger_heading_list = []
                    start_flag = False
                    _challenger_sp = 0
                    _AV_sp = 0
                    time_to_meeting_pos_list = []
                    for line in ori_traj_data[start_ind:]:
                        data = line.split('\t')
                        timestamp = get_sec(data[1].replace(',', '.'))
                        challenger_lat = float(data[15].replace(',', '.'))
                        challenger_long = float(data[16].replace(',', '.'))
                        challenger_utm_x, challenger_utm_y, _, _ = utm.from_latlon(challenger_lat, challenger_long)
                        challenger_x, challenger_y = utm_to_sumo_coordinate([challenger_utm_x, challenger_utm_y])
                        challenger_sp = float(data[6].replace(',', '.')) / 3.6
                        challenger_lon_sp = float(data[9].replace(',', '.')) / 3.6
                        challenger_lat_sp = float(data[10].replace(',', '.')) / 3.6
                        challenger_lon_acc = float(data[32].replace(',', '.'))
                        challenger_lat_acc = float(data[33].replace(',', '.'))
                        if len(traj_data) == 0:
                            challenger_acc = 0
                        else:
                            challenger_acc = (challenger_sp - _challenger_sp) / (timestamp - traj_data[-1][0])
                        challenger_heading = (-float(data[24].replace(',', '.'))) % 360
                        if scenario not in ['vru_at_crosswalk', 'vru_without_crosswalk']:
                            challenger_x = challenger_x - 0.3 * math.cos(math.radians(90 - challenger_heading))
                            challenger_y = challenger_y - 0.3 * math.sin(math.radians(90 - challenger_heading))
                        _challenger_sp = challenger_sp

                        AV_lat = float(data[82].replace(',', '.'))
                        AV_long = float(data[83].replace(',', '.'))
                        AV_utm_x, AV_utm_y, _, _ = utm.from_latlon(AV_lat, AV_long)
                        AV_x, AV_y = utm_to_sumo_coordinate([AV_utm_x, AV_utm_y])
                        AV_sp = float(data[76].replace(',', '.')) / 3.6
                        AV_lon_sp = float(data[79].replace(',', '.')) / 3.6
                        AV_lat_sp = float(data[80].replace(',', '.')) / 3.6
                        AV_lon_acc = float(data[96].replace(',', '.'))
                        AV_lat_acc = float(data[97].replace(',', '.'))
                        if len(traj_data) == 0:
                            AV_acc = 0
                        else:
                            AV_acc = (AV_sp - _AV_sp) / (timestamp - traj_data[-1][0])
                        AV_heading = (-float(data[91].replace(',', '.'))) % 360
                        _AV_sp = AV_sp

                        AV_x_list.append(AV_x)
                        AV_y_list.append(AV_y)
                        AV_sp_list.append(AV_sp)
                        AV_heading_list.append(AV_heading)
                        challenger_x_list.append(challenger_x)
                        challenger_y_list.append(challenger_y)
                        challenger_sp_list.append(challenger_sp)
                        challenger_heading_list.append(challenger_heading)

                        traj_data.append(
                            [timestamp, AV_x, AV_y, AV_sp, AV_lon_sp, AV_lat_sp, AV_acc, AV_lon_acc, AV_lat_acc, AV_heading,
                             challenger_x, challenger_y, challenger_sp, challenger_lon_sp, challenger_lat_sp, challenger_acc, challenger_lon_acc, challenger_lat_acc, challenger_heading]
                        )

                        time_to_meeting_pos = float(data[38].replace(',', '.'))
                        time_to_meeting_pos_list.append(time_to_meeting_pos)
                        if (time_to_meeting_pos < 0 or 'car_following' in scenario) and not start_flag:
                            start_timestamp = copy.deepcopy(timestamp)
                            value.append(timestamp)
                            start_flag = True
                            dis_err = 0
                            for key in record_keys:
                                if 'dis' in key and 'err' not in key:
                                    expected_dis_ind = record_keys.index(key)
                                    projected_x, projected_y = project_point_on_line([challenger_x, challenger_y], [AV_x, AV_y], 90 - AV_heading)
                                    dis_err = cal_dis([projected_x, projected_y], [AV_x, AV_y]) - float(value[expected_dis_ind])
                                    break
                            value.append(dis_err)
                            sp_err = 0
                            for key in record_keys:
                                if 'sp' in key and 'err' not in key:
                                    expected_sp_ind = record_keys.index(key)
                                    if scenario in ['lane_change', 'lane_departure_same']:
                                        sp_err = abs(AV_sp) - abs(challenger_sp) - float(value[expected_sp_ind])
                                    elif scenario in ['lane_departure_opposite']:
                                        sp_err = abs(AV_sp) + abs(challenger_sp) - float(value[expected_sp_ind])
                                    elif scenario in ['vru_without_crosswalk']:
                                        ind = ori_traj_data.index(line)
                                        line = ori_traj_data[ind + 80].split('\t')
                                        vru_sp = float(line[6].replace(',', '.')) / 3.6
                                        sp_err = abs(vru_sp) - float(value[expected_sp_ind])
                                    else:
                                        sp_err = abs(challenger_sp) - float(value[expected_sp_ind])
                                    break
                            value += [sp_err, AV_sp, collision_flag]
                            record.append(value)
                    if collision_flag:
                        traj_data.append([0 for _ in range(len(traj_keys))])
                    
                    save_csv(traj_data, traj_keys, traj_path, str(case_id), False)
        sorted_record = sorted(record, key=lambda x: x[0])
        save_csv(sorted_record, record_keys, traj_path, 'record', False)

import os
import re
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from pathlib import Path
import json

from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap


class VehicleInfo:
    def __init__(self, length, width, v_max, v_min, a_max, a_min) -> None:
        self.length = length
        self.width = width
        self.v_max = v_max
        self.v_min = v_min
        self.a_max = a_max
        self.a_min = a_min

def dis_from_point_to_point(point1, point2):
    dis = 0
    for x, y in zip(point1, point2):
        dis += (x - y) ** 2
    return math.sqrt(dis)


def dis_from_ind_to_ind(route, start_ind, end_ind):
    dis = 0
    if start_ind < end_ind:
        for i in range(start_ind, end_ind):
            dis += dis_from_point_to_point(route[i][:2], route[i+1][:2])
    return dis


def check_in_route(route, position, previous_NPN, thr):
    start_ind = max(0, previous_NPN - 50)
    end_ind = min(previous_NPN + 50, len(route))
    min_dis = 1000000
    NPN = 0
    for ind in range(start_ind, end_ind):
        dis = dis_from_point_to_point(route[ind][:2], position)
        if dis < min_dis:
            min_dis = dis
            NPN = ind
    if min_dis <= thr:
        return NPN, True
    else:
        return NPN, False


def check_in_scenario(start_ind, end_ind, in_route, NPN):
    if in_route:
        if start_ind <= NPN <= end_ind:
            return True
    return False


class vehicleState:
    timestamp = .0
    x = .0
    y = .0
    speed = .0
    acceleration = .0
    heading = .0


def save_csv(data, key, path, name, index_flag):
    if len(key) <= len(data[0]):
        csvfile = pd.DataFrame(columns=key, data=data)
        if path == None:
            csvfile.to_csv(name + '.csv', index=index_flag)
        else:
            csvfile.to_csv(path + '/' + name + '.csv', index=index_flag)


def check_file_exists(name, dir):
    flag = False
    regular_name = ''
    for n in name:
        regular_name = regular_name + n + '*'
    for files in os.listdir(dir):
        if re.search(regular_name, files):
            flag = True
            break
    return flag


def check_point_near_to_point(point1, point2, check_range):
    if abs(point1[0] - point2[0]) < check_range and abs(point1[1] - point2[1]) < check_range:
        return True
    else:
        return False


def haversine(coord1, coord2):
    R = 6372800  # Earth radius in meters
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    
    phi1, phi2 = math.radians(lat1), math.radians(lat2) 
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + \
        math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))


def cal_dis(point1, point2):
    dis = 0
    for x, y in zip(point1, point2):
        dis += (x - y) ** 2
    return math.sqrt(dis)


def distancPoint2Line(x1, y1, x2, y2, x, y):  # a signed distance
    A = y2-y1
    B = x1-x2
    C = (x2-x1)*y1-(y2-y1)*x1

    distance = abs(A*x+B*y+C)/math.sqrt(A*A+B*B)
    L = (y-y1)*(x2-x1) - (y2-y1)*(x-x1)

    if L < 0:
        distance = -distance

    return distance


def is_number(s):
    try:
        temp = float(s)
        return temp, True
    except ValueError:
        pass

    try:
        import unicodedata
        temp = unicodedata.numeric(s)
        return temp, True
    except (TypeError, ValueError):
        pass

    return s, False


def get_scenario_folder_name(scenario):
    if scenario == 'Cut-In':
        return 'cut_in'
    elif scenario == 'Car Following':
        return 'car_following'
    elif scenario == 'VRU Crossing the Street without Crosswalk':
        return 'vru_without_crosswalk'
    elif scenario == 'VRU Crossing the Street at the Crosswalk':
        return 'vru_at_crosswalk'
    elif scenario == 'Lane Departure (same direction)':
        return 'lane_departure_same'
    elif scenario == 'Lane Departure (opposite direction)':
        return 'lane_departure_opposite'
    elif scenario == 'Left Turn (AV goes straight)':
        return 'left_turn_straight'
    elif scenario == 'Left Turn (AV turns left)':
        return 'left_turn_turn'
    elif scenario == 'Right Turn (AV goes straight)':
        return 'right_turn_straight'
    elif scenario == 'Right Turn (AV turns right)':
        return 'right_turn_turn'
    elif scenario == 'AV Merging into the Roundabout':
        return 'roundabout_av_outside'
    elif scenario == 'BV Merging into the Roundabout':
        return 'roundabout_av_inside'
    elif scenario == 'Vehicle Encroachment':
        return 'vehicle_encroachment'
    elif scenario == 'Traffic Signal':
        return 'traffic_signal'
    elif scenario == 'Merge':
        return 'merge'
    else:
        return 'others'


def read_csv(path):
    f = open(path, 'rb').read()
    origin_data = f.decode().split('\n')
    data = [[] for _ in range(len(origin_data) - 2)]

    name = origin_data[0].split(',')
    if name[-1][-1:] == '\r':
        name[-1] = name[-1][:-1]

    for i in range(1, len(origin_data) - 1):
        flag = False
        index = 0
        for ch in origin_data[i]:
            if ch == '"':
                flag = not flag
                origin_data[i] = origin_data[i][:index] + origin_data[i][(index + 1):]
                index -= 1
            if flag:
                if ch == ',':
                    origin_data[i] = origin_data[i][:index] + origin_data[i][(index + 1):]
                    index -= 1
            index += 1
        temp = origin_data[i].split(',')
        for j in range(len(temp)):
            num, flag = is_number(temp[j])
            if num != '':
                if flag:
                    data[i - 1].append(num)
                else:
                    data[i - 1].append(temp[j])
    return name, data


def haversine(coord1, coord2):
    R = 6372800  # Earth radius in meters
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    
    phi1, phi2 = math.radians(lat1), math.radians(lat2) 
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + \
        math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))


def rotate_point2_around_point1(point1, point2, angle):
    # rotate point2 around point1 clockwise
    x1, y1 = point1[:2]
    x2, y2 = point2[:2]

    x = x1 + math.cos(angle) * (x2 - x1) + math.sin(angle) * (y2 - y1)
    y = y1 - math.sin(angle) * (x2 - x1) + math.cos(angle) * (y2 - y1)
    return [x, y]


def ratio(x, bound1, bound2):
    # output the same ratio in bound2 as that of x in bound1
    y = (x - bound1[0]) / (bound1[1] - bound1[0]) * (bound2[1] - bound2[0]) + bound2[0]
    return y


def intersect_y(y_val, y_curve, x_curve):
    # find the intersection of a curve and a horizontal line
    index = (np.abs(y_curve - y_val)).argmin()
    if index == len(y_curve) - 1:
        ind_up = index
        ind_lo = index - 1
    elif index == 0:
        ind_up = index + 1
        ind_lo = index
    elif np.abs(y_curve[index+1] - y_val) > np.abs(y_curve[index-1] - y_val):
        ind_up = index
        ind_lo = index - 1
    else:
        ind_up = index + 1
        ind_lo = index
    y_interp = np.interp(y_val, [y_curve[ind_lo],y_curve[ind_up]], [x_curve[ind_lo],x_curve[ind_up]])
    return y_interp


def intersect_x(x_val, x_curve, y_curve):
    # find the intersection of a curve and a vertical line
    index = (np.abs(x_curve - x_val)).argmin()
    if index == len(x_curve) - 1:
        ind_up = index
        ind_lo = index - 1
    elif index == 0:
        ind_up = index + 1
        ind_lo = index
    elif np.abs(x_curve[index+1] - x_val) > np.abs(x_curve[index-1] - x_val):
        ind_up = index
        ind_lo = index - 1
    else:
        ind_up = index + 1
        ind_lo = index
    x_interp = np.interp(x_val, [x_curve[ind_lo],x_curve[ind_up]], [y_curve[ind_lo],y_curve[ind_up]])
    return x_interp


def cal_dis_along_path(start_ind, end_ind, path):
    if start_ind > end_ind:
        print('start_ind must be lower than end_ind!')
        return 0
    elif start_ind == end_ind:
        return 0
    else:
        if len(path) <= end_ind:
            print('end_ind is greater than path length!')
            return 0
        else:
            dis = 0
            for i in range(start_ind, end_ind):
                dis += cal_dis(path[i][:2], path[i+1][:2])
            return dis


def cal_min_dis_point_to_rectangle(rectangle, point):
    # rectangle: [center point x, center point y, length, width, heading]
    # point: [x, y]
    center_point_x, center_point_y, length, width, heading = rectangle
    rotated_point = rotate_point2_around_point1([center_point_x, center_point_y], point, heading)
    if rotated_point[0] > center_point_x + length / 2:
        if rotated_point[1] > center_point_y + width / 2:
            return 'outside', cal_dis(rotated_point, [center_point_x + length / 2, center_point_y + width / 2])
        elif center_point_y - width / 2 <= rotated_point[1] <= center_point_y + width / 2:
            return 'outside', rotated_point[0] - center_point_x - length / 2
        elif rotated_point[1] < center_point_y - width / 2:
            return 'outside', cal_dis(rotated_point, [center_point_x + length / 2, center_point_y - width / 2])
    elif rotated_point[0] == center_point_x + length / 2:
        if rotated_point[1] > center_point_y + width / 2:
            return 'outside', rotated_point[1] - center_point_y - width / 2
        elif center_point_y - width / 2 <= rotated_point[1] <= center_point_y + width / 2:
            return 'overlap', 0
        elif rotated_point[1] < center_point_y - width / 2:
            return 'outside', center_point_y - width / 2 - rotated_point[1]
    elif center_point_x - length / 2 < rotated_point[0] < center_point_x + length / 2:
        if rotated_point[1] > center_point_y + width / 2:
            return 'outside', rotated_point[1] - center_point_y - width / 2
        elif rotated_point[1] == center_point_y + width / 2:
            return 'overlap', 0
        elif center_point_y - width / 2 < rotated_point[1] < center_point_y + width / 2:
            return 'inside', -min(center_point_x + length / 2 - rotated_point[0], rotated_point[0] - center_point_x + length / 2,
                                  center_point_y + width / 2 - rotated_point[1], rotated_point[1] - center_point_y + width / 2)
        elif rotated_point[1] == center_point_y - width / 2:
            return 'overlap', 0
        elif rotated_point[1] < center_point_y - width / 2:
            return 'outside', center_point_y - width / 2 - rotated_point[1]
    elif rotated_point[0] == center_point_x - length / 2:
        if rotated_point[1] > center_point_y + width / 2:
            return 'outside', rotated_point[1] - center_point_y - width / 2
        elif center_point_y - width / 2 <= rotated_point[1] <= center_point_y + width / 2:
            return 'overlap', 0
        elif rotated_point[1] < center_point_y - width / 2:
            return 'outside', center_point_y - width / 2 - rotated_point[1]
    elif rotated_point[0] < center_point_x - width / 2:
        if rotated_point[1] > center_point_y + width / 2:
            return 'outside', cal_dis(rotated_point, [center_point_x - length / 2, center_point_y + width / 2])
        elif center_point_y - width / 2 <= rotated_point[1] <= center_point_y + width / 2:
            return 'outside', center_point_x - length / 2 - rotated_point[0]
        elif rotated_point[1] < center_point_y - width / 2:
            return 'outside', cal_dis(rotated_point, [center_point_x - length / 2, center_point_y - width / 2])
    return 'err', 0


def cal_corners(rect):
    cx, cy, width, height, angle = rect
    # Calculate the coordinates of the four corners of the rectangle
    w_half, h_half = width / 2, height / 2
    angle_cos = math.cos(angle)
    angle_sin = math.sin(angle)
    corners = []

    for dx, dy in [(-w_half, -h_half), (w_half, -h_half), (w_half, h_half), (-w_half, h_half)]:
        x = cx + dx * angle_cos - dy * angle_sin
        y = cy + dx * angle_sin + dy * angle_cos
        corners.append((x, y))

    return corners


def cal_min_dis_of_rectangles(rectangle_1, rectangle_2):
    # rectangle: [center point x, center point y, length, width, heading]
    # heading: x - 0, y - pi / 2
    if check_overlap_of_rectangles(rectangle_1, rectangle_2):
        return 0
    else:
        corners_1 = cal_corners(rectangle_1)
        corners_2 = cal_corners(rectangle_2)
        dis = []
        for corner in corners_1:
            dis.append(cal_min_dis_point_to_rectangle(rectangle_2, corner)[1])
        for corner in corners_2:
            dis.append(cal_min_dis_point_to_rectangle(rectangle_1, corner)[1])
        return min(dis)


def cal_2d_TTC(rectangle_1, speed_1, rectangle_2, speed_2):
    # rectangle: [center point x, center point y, length, width, heading]
    # heading: x - 0, y - pi / 2
    # speed direction is the same as heading
    flag = False
    ttc = math.inf
    x1, y1, length1, width1, heading1 = rectangle_1
    x2, y2, length2, width2, heading2 = rectangle_2
    rotated_x1, rotated_y1 = rotate_point2_around_point1([x2, y2], [x1, y1], heading2)

    rotated_rectangle_1 = [rotated_x1 - x2, rotated_y1 - y2, length1, width1, heading1 - heading2]
    cross_point = [-(rotated_y1 - y2) / math.tan(rotated_rectangle_1[-1]) + (rotated_x1 - x2), 0]
    if cross_point[0] > 0 and ((rotated_y1 > 0 and math.sin(rotated_rectangle_1[-1]) < 0) or (rotated_y1 < 0 and math.sin(rotated_rectangle_1[-1]) > 0)):
        rotated_left_rear_corner1, rotated_left_front_corner1, rotated_right_front_corner1, rotated_right_rear_corner1 = rbbox_to_corners(rotated_rectangle_1)
        cross_point1 = [(width2 / 2 - rotated_left_rear_corner1[1]) / math.tan(rotated_rectangle_1[-1]) + rotated_left_rear_corner1[0], width2 / 2]
        cross_point2 = [(-width2 / 2 - rotated_left_rear_corner1[1]) / math.tan(rotated_rectangle_1[-1]) + rotated_left_rear_corner1[0], width2 / 2]
        cross_point3 = [(width2 / 2 - rotated_right_front_corner1[1]) / math.tan(rotated_rectangle_1[-1]) + rotated_right_front_corner1[0], width2 / 2]
        cross_point4 = [(-width2 / 2 - rotated_right_front_corner1[1]) / math.tan(rotated_rectangle_1[-1]) + rotated_right_front_corner1[0], width2 / 2]
        dis = [cal_dis(cross_point1, rotated_left_front_corner1), cal_dis(cross_point2, rotated_left_rear_corner1),
            cal_dis(cross_point3, rotated_right_front_corner1), cal_dis(cross_point4, rotated_right_rear_corner1)]
        min_t1 = min(dis) / speed_1
        max_t1 = max(dis) / speed_1

        rotated_x2, rotated_y2 = rotate_point2_around_point1([x1, y1], [x2, y2], heading1)
        rotated_rectangle_2 = [rotated_x2 - x1, rotated_y2 - y1, length2, width2, heading2 - heading1]
        rotated_left_rear_corner1, rotated_left_front_corner1, rotated_right_front_corner1, rotated_right_rear_corner1 = rbbox_to_corners(rotated_rectangle_2)
        cross_point1 = [(width1 / 2 - rotated_left_rear_corner1[1]) / math.tan(rotated_rectangle_1[-1]) + rotated_left_rear_corner1[0], width1 / 2]
        cross_point2 = [(-width1 / 2 - rotated_left_rear_corner1[1]) / math.tan(rotated_rectangle_1[-1]) + rotated_left_rear_corner1[0], width1 / 2]
        cross_point3 = [(width1 / 2 - rotated_right_front_corner1[1]) / math.tan(rotated_rectangle_1[-1]) + rotated_right_front_corner1[0], width1 / 2]
        cross_point4 = [(-width1 / 2 - rotated_right_front_corner1[1]) / math.tan(rotated_rectangle_1[-1]) + rotated_right_front_corner1[0], width1 / 2]
        dis = [cal_dis(cross_point1, rotated_left_front_corner1), cal_dis(cross_point2, rotated_left_rear_corner1),
            cal_dis(cross_point3, rotated_right_front_corner1), cal_dis(cross_point4, rotated_right_rear_corner1)]
        min_t2 = min(dis) / speed_2
        max_t2 = max(dis) / speed_2
        if min_t1 >= max_t2 or min_t2 >= max_t1:
            return flag, ttc
        else:
            t = max(min_t1, min_t2)
            while t < min(max_t1, max_t2):
                moved_rotated_rectangle_2 = [rotated_x2 - x1 + speed_2 * t * math.cos(heading2 - heading1), rotated_y2 - y1 + speed_2 * t * math.sin(heading2 - heading1), length2, width2, heading2 - heading1]
                moved_rotated_rectangle_1 = [speed_1 * t, 0, length1, width1, 0]
                if check_overlap_of_rectangles(moved_rotated_rectangle_1, moved_rotated_rectangle_2):
                    ttc = t
                    return True, ttc
                t += 0.1
            return False, ttc
    else:
        return flag, ttc


def rbbox_to_corners(rbbox):
    # generate clockwise corners and rotate it counterclockwise
    # rbbox: [center x, center y, length, width, angle]
    cx, cy, y_d, x_d, angle = rbbox[:5]
    a_cos = math.cos(-angle)
    a_sin = math.sin(-angle)
    corners_x = [-y_d / 2, y_d / 2, y_d / 2, -y_d / 2]
    corners_y = [x_d / 2, x_d / 2, -x_d / 2, -x_d / 2]
    corners = []
    for i in range(4):
        corners.append([a_cos * corners_x[i] + a_sin * corners_y[i] + cx, -a_sin * corners_x[i] + a_cos * corners_y[i] + cy])
    return corners


def check_overlap_of_rectangles(rbbox1, rbbox2):
    corners1 = rbbox_to_corners(rbbox1)
    corners2 = rbbox_to_corners(rbbox2)
    corners1 = np.asarray(corners1)
    corners2 = np.asarray(corners2)
    corners1 = Polygon(corners1[:8].reshape((4, 2)))
    corners2 = Polygon(corners2[:8].reshape((4, 2)))
    if not corners1.is_valid or not corners2.is_valid:
        return 0
    inter = Polygon(corners1).intersection(Polygon(corners2)).area
    if inter > 0:
        return True
    else:
        return False


def cal_TTC_simple(rectangle_1, speed_1, rectangle_2, speed_2):
    # rectangle: [center point x, center point y, length, width, heading]
    # heading: x - 0, y - pi / 2
    # speed direction is the same as heading
    flag = False
    ttc = math.inf
    x1, y1, length1, width1, heading1 = rectangle_1
    x2, y2, length2, width2, heading2 = rectangle_2
    rotated_x1, rotated_y1 = rotate_point2_around_point1([x2, y2], [x1, y1], heading2)

    rotated_rectangle_1 = [rotated_x1 - x2, rotated_y1 - y2, length1, width1, heading1 - heading2]
    cross_point = [-(rotated_y1 - y2) / math.tan(rotated_rectangle_1[-1]) + (rotated_x1 - x2), 0]
    if cross_point[0] > 0 and ((rotated_y1 > 0 and math.sin(rotated_rectangle_1[-1]) < 0) or (rotated_y1 < 0 and math.sin(rotated_rectangle_1[-1]) > 0)):
        rotated_left_rear_corner1, rotated_left_front_corner1, rotated_right_front_corner1, rotated_right_rear_corner1 = rbbox_to_corners(rotated_rectangle_1)
        cross_point1 = [(width2 / 2 - rotated_left_rear_corner1[1]) / math.tan(rotated_rectangle_1[-1]) + rotated_left_rear_corner1[0], width2 / 2]
        cross_point2 = [(-width2 / 2 - rotated_left_rear_corner1[1]) / math.tan(rotated_rectangle_1[-1]) + rotated_left_rear_corner1[0], width2 / 2]
        cross_point3 = [(width2 / 2 - rotated_right_front_corner1[1]) / math.tan(rotated_rectangle_1[-1]) + rotated_right_front_corner1[0], width2 / 2]
        cross_point4 = [(-width2 / 2 - rotated_right_front_corner1[1]) / math.tan(rotated_rectangle_1[-1]) + rotated_right_front_corner1[0], width2 / 2]
        dis = [cal_dis(cross_point1, rotated_left_front_corner1), cal_dis(cross_point2, rotated_left_rear_corner1),
            cal_dis(cross_point3, rotated_right_front_corner1), cal_dis(cross_point4, rotated_right_rear_corner1)]
        min_t1 = min(dis) / speed_1
        max_t1 = max(dis) / speed_1
        return True, min_t1
    else:
        return flag, ttc


def cal_route_conflict_ind(route_1, length_1, width_1, route_2, length_2, width_2):
    dis = np.zeros((len(route_1), len(route_2)))
    heading_1 = [cal_heading(route_1[i:i+2]) for i in range(len(route_1) - 1)]
    heading_1 += [heading_1[-1]]
    heading_2 = [cal_heading(route_2[i:i+2]) for i in range(len(route_2) - 1)]
    heading_2 += [heading_2[-1]]
    for i in range(len(route_1)):
        for j in range(len(route_2)):
            dis[i, j] = cal_min_dis_of_rectangles([route_1[i][0], route_1[i][1], length_1, width_1, heading_1[i]], [route_2[j][0], route_2[j][1], length_2, width_2, heading_2[j]])
            # dis[i, j] = cal_dis(route_1[i, :2], route_2[j, :2])
    min_dis_1 = [min(dis[i, :]) for i in range(len(route_1))]
    min_dis_2 = [min(dis[:, j]) for j in range(len(route_2))]
    ind_1, ind_2 = -1, -1
    for i in range(len(min_dis_1)):
        if min_dis_1[i] <= 0:
            ind_1 = i
            break
    for j in range(len(min_dis_2)):
        if min_dis_2[j] <= 0:
            ind_2 = j
            break
    return ind_1, ind_2


def cal_route_segment_length(route):
    dis = []
    for i in range(len(route) - 1):
        dis.append(cal_dis(route[i][:2], route[i+1][:2]))
    return dis


def cal_TTC_with_route(rectangle_1, speed_1, route_1, ind_1, seg_len_1, rectangle_2, speed_2, route_2, ind_2, seg_len_2, NNPN_1=-1, NNPN_2=-1):
    # rectangle: [center point x, center point y, length, width, heading]
    # heading: x - 0, y - pi / 2
    # speed direction is the same as heading
    flag = False
    ttc = math.inf
    if ind_1 != -1 and ind_2 != -1:
        NNPN_1 = find_next_route_points(route_1, rectangle_1[:2], NNPN_1)
        NNPN_2 = find_next_route_points(route_2, rectangle_2[:2], NNPN_2)
        if speed_1 == 0:
            time_to_collision_area_1 = math.inf
        else:
            time_to_collision_area_1 = cal_dis_on_route(route_1, rectangle_1[:2], NNPN_1, ind_1, seg_len_1) / speed_1
        if speed_2 == 0:
            time_to_collision_area_2 = math.inf
        else:
            time_to_collision_area_2 = cal_dis_on_route(route_2, rectangle_2[:2], NNPN_2, ind_2, seg_len_2) / speed_2
        time_to_collision_area = max(time_to_collision_area_1, time_to_collision_area_2)
        if time_to_collision_area_1 == time_to_collision_area_2 == 0:
            tmp_rec_1, NNPN_1_ = rectangle_1, NNPN_1
            tmp_rec_2, NNPN_2_ = rectangle_2, NNPN_2
        elif time_to_collision_area == time_to_collision_area_1:
            tmp_rec_1, NNPN_1_ = [route_1[ind_1][0], route_1[ind_1][1], rectangle_1[2], rectangle_1[3], (90 - route_1[ind_1][2]) / 180 * math.pi], ind_1
            tmp_rec_2, NNPN_2_ = cal_position_on_route(rectangle_2, time_to_collision_area, speed_2, route_2, NNPN_2, seg_len_2)
        elif time_to_collision_area == time_to_collision_area_2:
            tmp_rec_1, NNPN_1_ = cal_position_on_route(rectangle_1, time_to_collision_area, speed_1, route_1, NNPN_1, seg_len_1)
            tmp_rec_2, NNPN_2_ = [route_2[ind_2][0], route_2[ind_2][1], rectangle_2[2], rectangle_2[3], (90 - route_2[ind_2][2]) / 180 * math.pi], ind_2
        if NNPN_1_ == -1 or NNPN_2_ == -1:
            return flag, ttc

        t = 0
        delta_t = 0.1
        while not flag and 0 <= NNPN_1_ < len(seg_len_1) and 0 <= NNPN_2_ < len(seg_len_2):
            t += delta_t
            tmp_rec_1, NNPN_1_ = cal_position_on_route(tmp_rec_1, delta_t, speed_1, route_1, NNPN_1_, seg_len_1)
            tmp_rec_2, NNPN_2_ = cal_position_on_route(tmp_rec_2, delta_t, speed_2, route_2, NNPN_2_, seg_len_2)
            flag = check_overlap_of_rectangles(tmp_rec_1, tmp_rec_2)
            heading_1 = tmp_rec_1[-1] % (2 * math.pi)
            heading_2 = tmp_rec_2[-1] % (2 * math.pi)
            if abs(heading_1 - heading_2) < 0.1:
                break
        heading_1 = tmp_rec_1[-1] % (2 * math.pi)
        heading_2 = tmp_rec_2[-1] % (2 * math.pi)
        if flag:
            ttc = t + time_to_collision_area
        if not flag and abs(heading_1 - heading_2) < 0.1:
            rotated_pos_2 = rotate_point2_around_point1(tmp_rec_1[:2], tmp_rec_2[:2], tmp_rec_1[-1])
            if rotated_pos_2[0] > tmp_rec_1[0] and speed_1 > speed_2:
                flag = True
                ttc = t + time_to_collision_area + (rotated_pos_2[0] - tmp_rec_1[0] - tmp_rec_1[2] / 2 - tmp_rec_2[2] / 2) / (speed_1 - speed_2)
            if rotated_pos_2[0] < tmp_rec_1[0] and speed_1 < speed_2:
                flag = True
                ttc = t + time_to_collision_area + (tmp_rec_1[0] - rotated_pos_2[0] - tmp_rec_1[2] / 2 - tmp_rec_2[2] / 2) / (speed_2 - speed_1)
        if ttc == math.inf:
            tmp = 0
        if ttc < 0:
            ttc = math.inf
        return flag, ttc
    else:
        return flag, ttc


def cal_dis_on_route(route, position, NNPN, target_ind, seg_len):
    if NNPN >= target_ind:
        return 0
    else:
        dis = cal_dis(position, route[NNPN][:2])
        for i in range(NNPN, target_ind):
            dis += seg_len[i]
        return dis


def cal_position_on_route(rectangle, t, sp, route, NNPN, seg_len):
    if NNPN >= len(seg_len):
        return rectangle, -1
    dis2NNPN = cal_dis(rectangle[:2], route[NNPN][:2])
    while dis2NNPN <= 0:
        NNPN += 1
        if NNPN >= len(seg_len):
            return rectangle, -1
        dis2NNPN = cal_dis(rectangle[:2], route[NNPN][:2])
    dis = sp * t
    length = rectangle[2]
    width = rectangle[3]
    if dis <= dis2NNPN:
        x = dis / dis2NNPN * (route[NNPN][0] - rectangle[0]) + rectangle[0]
        y = dis / dis2NNPN * (route[NNPN][1] - rectangle[1]) + rectangle[1]
        heading = dis / dis2NNPN * ((90 - route[NNPN][2]) / 180 * math.pi - rectangle[4]) + rectangle[4]
        return [x, y, length, width, heading], NNPN
    else:
        dis -= dis2NNPN
        NNPN_ = NNPN
        while dis > seg_len[NNPN_]:
            dis -= seg_len[NNPN_]
            NNPN_ += 1
            if NNPN_ >= len(seg_len):
                x = dis * math.cos((90 - route[NNPN_][2]) / 180 * math.pi) + route[NNPN_][0]
                y = dis * math.sin((90 - route[NNPN_][2]) / 180 * math.pi) + route[NNPN_][1]
                heading = (90 - route[NNPN_][2]) / 180 * math.pi
                return [x, y, length, width, heading], -1  # obj is out of the route
        x = dis / seg_len[NNPN_] * (route[NNPN_+1][0] - route[NNPN_][0]) + route[NNPN_][0]
        y = dis / seg_len[NNPN_] * (route[NNPN_+1][1] - route[NNPN_][1]) + route[NNPN_][1]
        if route[NNPN_][2] - route[NNPN_+1][2] > 180:
            heading_ = route[NNPN_+1][2] + 360
        elif route[NNPN_][2] - route[NNPN_+1][2] < -180:
            heading_ = route[NNPN_+1][2] - 360
        else:
            heading_ = route[NNPN_+1][2]
        heading = dis / seg_len[NNPN_] * ((90 - heading_) / 180 * math.pi - (90 - route[NNPN_][2]) / 180 * math.pi) + (90 - route[NNPN_][2]) / 180 * math.pi
        return [x, y, length, width, heading], NNPN_ + 1


def find_next_route_points(route, position, NNPN=-1):
    # NNPN: nearest next point number
    dis = []
    if NNPN == -1:
        for n in range(len(route)):
            dis.append(cal_dis(route[n][:2], position))
    else:
        for n in range(NNPN, NNPN+100):
            dis.append(cal_dis(route[n%len(route)][:2], position))
    duo_dis = []
    for i in range(len(dis) - 1):
        duo_dis.append(dis[i] + dis[i+1])

    if len(duo_dis) == 0:
        return len(route) - 1
    else:
        if NNPN == -1:
            ind = duo_dis.index(min(duo_dis))
            return ind
        else:
            ind = duo_dis.index(min(duo_dis))
            return NNPN + ind


def angle_from_vec1_to_vec2(vec1, vec2):
    # Calculate the angle of each vector using atan2
    angle_vec1 = np.arctan2(vec1[1], vec1[0])
    angle_vec2 = np.arctan2(vec2[1], vec2[0])
    
    # Calculate the angle from vec1 to vec2
    angle = angle_vec2 - angle_vec1
    
    # Normalize the angle to be within -pi and pi
    angle = np.arctan2(np.sin(angle), np.cos(angle))

    return angle


def str_replacer(s, newstring, index, nofail=False):
    # raise an error if index is outside of the string
    if not nofail and index not in range(len(s)):
        raise ValueError("index outside given string")

    # if not erroring, but the index is still not in the correct range..
    if index < 0:  # add it to the beginning
        return newstring + s
    if index > len(s):  # add it to the end
        return s + newstring

    # insert the new string between "slices" of the original
    return s[:index] + newstring + s[index + 1:]


def utm_to_sumo_coordinate(utm_coordinate):
    """
    Convert the UTM coordinate to the SUMO coordinate. the input will be a list of [x, y].
    """
    return [float(utm_coordinate[0]-277600+102.89), float(utm_coordinate[1]-4686800+281.25)]


def sumo_to_utm_coordinate(sumo_coordinate):
    """
    Convert the SUMO coordinate to the UTM coordinate. the input will be a list of [x, y].
    """
    return [float(sumo_coordinate[0]+277600-102.89), float(sumo_coordinate[1]+4686800-281.25)]


def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def distance_from_point_to_line_with_angle(point0, point1, theta):
    """Calculate the perpendicular distance from point0 to a line given point1 on the line and the angle."""
    x0, y0 = point0
    x1, y1 = point1
    # Convert angle in degrees to slope
    m = math.tan(math.radians(theta))
    
    # Convert point-slope form of line to standard form ax + by + c = 0
    # y - y1 = m(x - x1)
    # y - mx = y1 - mx1
    a = -m
    b = 1
    c = -y1 + m * x1
    
    # Calculate the distance using the standard distance formula
    numerator = abs(a * x0 + b * y0 + c)
    denominator = math.sqrt(a**2 + b**2)
    return numerator / denominator


def curvature(points):
    # Define the points as a NumPy array
    x, y = points[:,0], points[:,1]

    # Calculate the first and second derivatives of the curve
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Calculate the curvature of the curve
    curvature = -(ddx*dy - dx*ddy) / np.power(dx**2 + dy**2, 1.5)

    return np.mean(curvature)


def cal_heading(points):
    # Calculate the heading of the line defined by the points
    k, b = np.polyfit([point[0] for point in points],
                      [point[1] for point in points], 1)
    k_, b_ = np.polyfit([ind for ind in range(len(points))], [point[0] for point in points], 1)
    if k_ > 0:
        heading = np.arctan(k)
    else:
        heading = np.arctan(k) + math.pi
    heading = heading % (2 * math.pi)
    return heading


def project_point_on_line(point0, point1, heading):
    # point0: point to be projected
    # point1: point on the line
    # heading: heading of the line
    # Convert heading to radians
    px, py = point0
    lx, ly = point1
    theta = math.radians(heading)
    
    # Calculate direction vector
    dx = math.cos(theta)
    dy = math.sin(theta)
    
    # Vector from line point to point to be projected
    vx = px - lx
    vy = py - ly
    
    # Dot product of direction vector and vector vx, vy
    t = vx * dx + vy * dy
    
    # Calculate projection point
    proj_x = lx + t * dx
    proj_y = ly + t * dy
    
    return proj_x, proj_y


def are_points_on_same_side(point1, point2, point3, point4):
    # Calculate the determinant for each point
    # point1 and point2 are the points that define the line
    # point3 and point4 are the points to check
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    x4, y4 = point4
    D1 = (x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)
    D2 = (x4 - x1) * (y2 - y1) - (y4 - y1) * (x2 - x1)

    # Check if the determinants have the same sign
    return D1 * D2 > 0


def rotation(x, y, x0, y0, angle):
    # Rotate point (x, y) around point (x0, y0) by angle counterclockwise
    x_rot =  (x - x0) * math.cos(angle) + (y - y0) * math.sin(angle) + x0
    y_rot = -(x - x0) * math.sin(angle) + (y - y0) * math.cos(angle) + y0
    return x_rot, y_rot


def ratio(x, bound1, bound2):
    # output the same ratio in bound2 as that of x in bound1
    y = (x - bound1[0]) / (bound1[1] - bound1[0]) * (bound2[1] - bound2[0]) + bound2[0]
    return y


def fig_format_setting(fontsize, scenario=None, tight_layout=True, constrained_layout=False):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times", "DejaVu Serif", "Georgia"]
    plt.rcParams['axes.labelsize'] = fontsize
    plt.rcParams['xtick.labelsize'] = fontsize
    plt.rcParams['ytick.labelsize'] = fontsize

    fig, ax = plt.subplots(constrained_layout=constrained_layout)

    ax.tick_params(axis='y', labelsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)

    if scenario != 'Traffic Signal':
        ylim = ax.get_ylim()
        if ylim[1] - ylim[0] < 0.2:
            c = sum(ylim) / 2
            c = round(c, 1)
            lower_limit = max(0, c - 0.1)
            ax.set_ylim([lower_limit, lower_limit + 0.2])
    if tight_layout:
        plt.tight_layout()
    return fig, ax


def load_argoverse_tracks(scenario_dir):
    # load tracks
    argoverse_scenario_dir = Path(scenario_dir)
    all_scenario_files = sorted(argoverse_scenario_dir.rglob('*.parquet'))
    tracks = []
    for scenario_path in all_scenario_files:
        scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)  # refer to README.md in av2.datasets.motion_forecasting.data_schema to learn the format
        cand_tracks = []
        for one_track in scenario.tracks:
            if cal_dis(one_track.object_states[0].position, one_track.object_states[-1].position) < 5:
                continue
            cand_tracks.append(one_track)
        tracks.append(cand_tracks)
    return scenario.city_name, tracks


def load_argoverse_map(scenario_dir):
    argoverse_scenario_dir = Path(scenario_dir)
    avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath=argoverse_scenario_dir, build_raster=False)  # refer to av2.map.map_api.ArgoverseStaticMap to learn the format
    return avm


def cal_angle(dx, dy):
    angle_rad = math.atan2(dy, dx) % (2 * math.pi)
    return angle_rad


def collinear(point1, point2, point3, threshold):
    # whether point1 is on the line crossing point 2 and 3
    x1, y1 = point1[:2]
    x2, y2 = point2[:2]
    x3, y3 = point3[:2]
    return abs((y1 - y2) * (x1 - x3) - (y1 - y3) * (x1 - x2)) <= abs(threshold * (x1 - x3) * (x1 - x2))


def cal_dis_from_point_to_line(point1, point2, point3):
    # from point1 to the line go through point2 and point3
    x1, y1 = point1[:2]
    x2, y2 = point2[:2]
    x3, y3 = point3[:2]
    return abs((x3 - x2) * (y2 - y1) - (x2 - x1) * (y3 - y2)) / math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)


def determine_point_between_lines(point0, point1, point2, point3, point4):
    # determine if point0 is between the lines that go through point1 & 2 and point3 & 4
    x0, y0 = point0[:2]
    x1, y1 = point1[:2]
    x2, y2 = point2[:2]
    x3, y3 = point3[:2]
    x4, y4 = point4[:2]
    if y2 > y1 and y4 < y3:
        x3, y3 = point4[:2]
        x4, y4 = point3[:2]
    temp1 = (y2 - y1) * (x0 - x1) - (y0 - y1) * (x2 - x1)
    temp2 = (y4 - y3) * (x0 - x3) - (y0 - y3) * (x4 - x3)
    if temp1 * temp2 < 0:
        return True
    else:
        return False


def same_side(points, point1, point2):
    # determine if all points are on the same side of the line going through point1 and point2
    set1 = []
    set2 = []
    set3 = []
    x1, y1 = point1[:2]
    x2, y2 = point2[:2]
    for point in points:
        if len(set1) == 0:
            if collinear(point, point1, point2, 0):
                set3.append(point)
            else:
                set1.append(point)
        else:
            x, y = point[:2]
            x_, y_ = set1[0]
            if ((y1-y2)*(x-x1)+(x2-x1)*(y-y1))*((y1-y2)*(x_-x1)+(x2-x1)*(y_-y1))>0:
                set1.append(point)
            elif ((y1-y2)*(x-x1)+(x2-x1)*(y-y1))*((y1-y2)*(x_-x1)+(x2-x1)*(y_-y1))<0:
                set2.append(point)
            else:
                set3.append(point)
    if len(set2) == 0 and len(set3) == 0:
        return True, set1, set2, set3
    else:
        return False, set1, set2, set3


def line_intersection(point1, point2, point3, point4):
    # find the intersection point of the lines that go through point 1 & 2 and point 3 & 4
    x1, y1 = point1[:2]
    x2, y2 = point2[:2]
    x3, y3 = point3[:2]
    x4, y4 = point4[:2]
    xdiff = (x1 - x2, x3 - x4)
    ydiff = (y1 - y2, y3 - y4)

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(point1, point2), det(point3, point4))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]


def judge_straight_track(track):
    start_position = track.object_states[0].position
    end_position = track.object_states[-1].position
    length = cal_dis(start_position, end_position)
    if length < 3:
        angle = cal_angle(end_position[0] - start_position[0], end_position[1] - start_position[1])
        return False, angle, angle, length
    else:
        for state in track.object_states:
            position_ = state.position
            dis = cal_dis(start_position, position_)
            if dis > 1:
                start_angle = cal_angle(position_[0] - start_position[0], position_[1] - start_position[1])
                break
        for state in reversed(track.object_states):
            _position = state.position
            dis = cal_dis(end_position, _position)
            if dis > 1:
                end_angle = cal_angle(end_position[0] - _position[0], end_position[1] - _position[1])
        if abs(start_angle - end_angle) < 0.1 or abs(abs(start_angle - end_angle) - 2 * math.pi) < 0.1:
            return True, start_angle, end_angle, length
        else:
            return False, start_angle, end_angle, length


def check_point_inside_polygon(point, polygon_point_list):
    # notice the polygon point order!!!!
    point = Point(point[0], point[1])
    polygon = Polygon(polygon_point_list)
    return polygon.contains(point)


def cal_track_start_speed(track):
    x_list = [state.position[0] for state in track.object_states[:5]]
    y_list = [state.position[1] for state in track.object_states[:5]]
    t_list = [0.1 * i for i in range(5)]
    vx, _ = np.polyfit(t_list, x_list, 1)
    vy, _ = np.polyfit(t_list, y_list, 1)
    return math.sqrt(vx ** 2 + vy ** 2)


def determine_two_points_on_the_same_side(point1, point2, point3, point4):
    # determine if point1 and point2 on the same side of the line going through point3 and point4
    x1, y1 = point1[:2]
    x2, y2 = point2[:2]
    x3, y3 = point3[:2]
    x4, y4 = point4[:2]
    temp1 = (y4 - y3) * (x1 - x3) - (y1 - y3) * (x4 - x3)
    temp2 = (y4 - y3) * (x2 - x3) - (y2 - y3) * (x4 - x3)
    if temp1 * temp2 > 0:
        return True
    else:
        return False


def determine_point_ahead_of_points(point1, point2, point3):
    # determine if point1 is ahead of the vector point2->point3
    x1, y1 = point1[:2]
    x2, y2 = rotate_point2_around_point1(point3, point2, math.pi / 2)
    x3, y3 = point3[:2]
    temp1 = (y3 - y2) * (x1 - x2) - (y1 - y2) * (x3 - x2)
    if temp1 > 0:
        return False
    else:
        return True


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def parse_config(path_to_json=r'./config.json'):
    with open(path_to_json) as f:
      data = json.load(f)
    return Struct(**data)


def plot_acc_distribution(
        data,
        max_value,
        fontsize,
        colors,
        alphas,
        folder,
        file_name,
        xlabel=r'Acceleration ($m/s^2$)',
        reverse=False
    ):
    ind1 = int(0.01 * len(data))
    ind2 = int(0.1 * len(data))
    ind3 = int(0.8 * len(data))

    fig_format_setting(fontsize)
    plt.hist([data])
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    plt.clf()
    if not reverse:
        x_bound = [xlim[1], max_value, data[ind1], data[ind2], data[ind3], xlim[0]]
    else:
        x_bound = [xlim[0], max_value, data[ind1], data[ind2], data[ind3], xlim[1]]
    print(x_bound)
    y = [ylim[0], ylim[0], ylim[1], ylim[1]]
    for ind in range(len(x_bound) - 1):
        x = [x_bound[ind], x_bound[ind + 1], x_bound[ind + 1], x_bound[ind]]
        plt.fill(x, y, facecolor=colors[ind], alpha=alphas[ind])
    plt.hist([data])
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.savefig(folder + '/' + file_name + '.svg', format='svg', bbox_inches='tight')
    plt.close()
    return data[ind1], data[ind2], data[ind3]


def plot_init_condition_distribution(
        data,
        x_range,
        y_range,
        x_res,
        y_res,
        fontsize=20,
        xlabel='',
        ylabel='',
        folder='.',
        file_name='initial_condition_distribution'
    ):
    points = np.array(data)

    # Extract x and y coordinates
    x = points[:, 0]
    y = points[:, 1]

    # Define the heatmap grid
    x_bins = np.arange(x_range[0], x_range[1], x_res)  # Bin edges for x
    y_bins = np.arange(y_range[0], y_range[1], y_res)  # Bin edges for y

    # Create a 2D histogram
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])

    # Plot the heatmap
    fig_format_setting(fontsize)

    plt.imshow(
        heatmap.T, origin='lower', cmap='turbo', 
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        interpolation='bicubic'  # Use interpolation for smoothness
    )

    cbar = plt.colorbar()
    cbar.set_label('Frequency', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.axis('auto')
    plt.tight_layout()
    plt.savefig(folder + '/' + file_name + '.svg', format='svg', bbox_inches='tight')
    plt.close()

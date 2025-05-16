import sys
import os

# Get the directory of the current script
dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(dir)

from utils import *
from settings import *


def extract_straight_lane(vector_lane_segments, flag_bike_lane_included=False):
    vector_straight_lane_segments = {}
    lane_width = {}
    for key, value in vector_lane_segments.items():
        if value.lane_type == 'VEHICLE' or flag_bike_lane_included:
            left_lane_marking = value.left_lane_boundary.xyz
            start_heading = cal_lane_start_heading(left_lane_marking)
            flag = True
            for ind in range(1, len(left_lane_marking)):
                heading = cal_angle(left_lane_marking[ind, 0] - left_lane_marking[ind - 1, 0], left_lane_marking[ind, 1] - left_lane_marking[ind - 1, 1])
                if abs(start_heading - heading) > straight_heaading_threshold:
                    flag = False
                    break
            if flag:
                vector_straight_lane_segments[key] = value
                straight_lane_marking = value.right_lane_boundary.xyz
                lane_width[key] = cal_dis_from_point_to_line(straight_lane_marking[0], left_lane_marking[0], left_lane_marking[-1])
    return vector_straight_lane_segments, lane_width


def extract_intersection_lane(vector_lane_segments, flag_bike_lane_included=False):
    vector_intersection_lane_segments = {}
    for key, value in vector_lane_segments.items():
        if value.is_intersection:
            vector_intersection_lane_segments[key] = value
    return vector_intersection_lane_segments


def extract_right_turn_lane(vector_lane_segments, flag_bike_lane_included=False):
    vector_right_turn_lane_segments = {}
    lane_width = []
    for key, value in vector_lane_segments.items():
        if (value.is_intersection and value.lane_type == 'VEHICLE') or flag_bike_lane_included:
            left_lane_marking = value.left_lane_boundary.xyz
            start_heading = cal_lane_start_heading(left_lane_marking)
            end_heading = cal_lane_end_heading(left_lane_marking)
            if abs(start_heading - math.pi / 2 - end_heading) < turning_heading_threshold or abs(start_heading + math.pi * 3 / 2 - end_heading) < turning_heading_threshold:
                vector_right_turn_lane_segments[key] = value
                right_lane_marking = value.right_lane_boundary.xyz
                lane_width.append(cal_dis(right_lane_marking[0], left_lane_marking[0]))
    return vector_right_turn_lane_segments, lane_width


def extract_straight_lane_intersecting_turning_lane(vector_lane_segments, vector_turning_lane_segments):
    vector_straight_lane_intersecting_turning_lane_segments = {}
    for key, value in vector_lane_segments.items():
        if value.lane_type == 'VEHICLE':
            turning_keys = list(vector_turning_lane_segments.keys())
            if not key in turning_keys:
                for _, turning_value in vector_turning_lane_segments.items():
                    if key in turning_value.predecessors or key in turning_value.successors:
                        vector_straight_lane_intersecting_turning_lane_segments[key] = value
                    else:
                        if collinear(value.right_lane_boundary.xyz[0], value.right_lane_boundary.xyz[-1], turning_value.right_lane_boundary.xyz[-1], 0.1) or collinear(value.right_lane_boundary.xyz[0], value.right_lane_boundary.xyz[-1], turning_value.right_lane_boundary.xyz[0], 0.1):
                            vector_straight_lane_intersecting_turning_lane_segments[key] = value
    return vector_straight_lane_intersecting_turning_lane_segments


def cal_lane_start_heading(lane_marking):
    return cal_angle(lane_marking[1, 0] - lane_marking[0, 0], lane_marking[1, 1] - lane_marking[0, 1])


def cal_lane_end_heading(lane_marking):
    return cal_angle(lane_marking[-1, 0] - lane_marking[-2, 0], lane_marking[-1, 1] - lane_marking[-2, 1])


def group_lanes(vector_lane_segments, lane_width):
    grouped_lanes = []
    grouped_lane_widths = []
    grouped_keys = []
    cand_ids = list(vector_lane_segments.keys())
    while len(cand_ids) > 0:
        id = cand_ids[0]
        ids = [id]
        cand_ids.remove(id)
        predecessors = vector_lane_segments[id].predecessors
        successors = vector_lane_segments[id].successors
        find_predecessor_ids(vector_lane_segments, id, ids, predecessors, cand_ids)
        find_successor_ids(vector_lane_segments, id, ids, successors, cand_ids)
        grouped_lanes.append({id: vector_lane_segments[id] for id in ids})
        grouped_lane_widths.append({id: lane_width[id] for id in ids})
    return grouped_lanes, grouped_lane_widths


def find_predecessor_ids(vector_lane_segments, id, ids, predecessors, cand_ids):
    for predecessor in predecessors:
        if predecessor in cand_ids:
            heading_1 = cal_lane_end_heading(vector_lane_segments[predecessor].left_lane_boundary.xyz)
            heading_2 = cal_lane_start_heading(vector_lane_segments[id].left_lane_boundary.xyz)
            if abs(heading_1 - heading_2) < straight_heaading_threshold:
                ids.append(predecessor)
                cand_ids.remove(predecessor)
                predecessors.remove(predecessor)
                predecessors += vector_lane_segments[predecessor].predecessors
                find_predecessor_ids(vector_lane_segments, id, ids, predecessors, cand_ids)


def find_successor_ids(vector_lane_segments, id, ids, successors, cand_ids):
    for successor in successors:
        if successor in cand_ids:
            heading_1 = cal_lane_start_heading(vector_lane_segments[successor].left_lane_boundary.xyz)
            heading_2 = cal_lane_end_heading(vector_lane_segments[id].left_lane_boundary.xyz)
            if abs(heading_1 - heading_2) < straight_heaading_threshold:
                ids.append(successor)
                cand_ids.remove(successor)
                successors.remove(successor)
                successors += vector_lane_segments[successor].successors
                find_successor_ids(vector_lane_segments, id, ids, successors, cand_ids)

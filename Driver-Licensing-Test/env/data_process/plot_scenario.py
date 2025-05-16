from pathlib import Path
from rich.progress import track
import matplotlib.pyplot as plt
from typing import List
import numpy as np
import math

from env.data_process.extract_tracks import *
from utils import *
from settings import *


def extract_intersection_lane(vector_lane_segments, flag=True):
    vector_intersection_lane_segments = {}
    for key, value in vector_lane_segments.items():
        if value.is_intersection and value.lane_type == "VEHICLE":
            vector_intersection_lane_segments[key] = value
            if flag:
                for ind in value.successors:
                    try:
                        vector_intersection_lane_segments[ind] = vector_lane_segments[ind]
                    except Exception as e:
                        pass
                for ind in value.predecessors:
                    try:
                        vector_intersection_lane_segments[ind] = vector_lane_segments[ind]
                    except Exception as e:
                        pass
    return vector_intersection_lane_segments

def plot_map(lane_segment, color='k', rotate_center=None, rotate_angle=0, show_lane_id=False):
    if rotate_center is None:
        left_lane_marking = np.transpose(lane_segment.left_lane_boundary.xyz)
        right_lane_marking = np.transpose(lane_segment.right_lane_boundary.xyz)
    else:
        left_lane_marking = np.transpose([rotate_point2_around_point1(rotate_center, point, rotate_angle) for point in lane_segment.left_lane_boundary.xyz])
        right_lane_marking = np.transpose([rotate_point2_around_point1(rotate_center, point, rotate_angle) for point in lane_segment.right_lane_boundary.xyz])
    plt.plot(left_lane_marking[0], left_lane_marking[1], color)
    plt.plot(right_lane_marking[0], right_lane_marking[1], color)
    if show_lane_id:
        plt.text(right_lane_marking[0][0], right_lane_marking[1][0], str(lane_segment.id))

def plot_track(track, color='g', rotate_center=None, rotate_angle=0, arrow=True):
    positions = []
    for state in track.object_states:
        if state.observed or True:
            if rotate_center is None:
                positions.append(state.position)
            else:
                positions.append(rotate_point2_around_point1(rotate_center, state.position, rotate_angle))
    positions = np.transpose(positions)
    start_heading, p_start = cal_track_start_heading(track)
    start_heading = start_heading - rotate_angle
    
    end_heading, p_end = cal_track_end_heading(track)
    end_heading = end_heading - rotate_angle

    if rotate_center is not None:
        p_start = rotate_point2_around_point1(rotate_center, p_start, rotate_angle)
        p_end = rotate_point2_around_point1(rotate_center, p_end, rotate_angle)

    arrow_len = 5
    plt.plot(positions[0], positions[1], color)
    if arrow:
        plt.plot(positions[0, 0], positions[1, 0], 'ro')
        plt.plot(p_start[0], p_start[1], 'ro')
        plt.arrow(positions[0, 0], positions[1, 0], arrow_len * math.cos(start_heading), arrow_len * math.sin(start_heading), head_width=2, color='r')
        plt.plot(positions[0, -1], positions[1, -1], 'bo')
        plt.plot(p_end[0], p_end[1], 'bo')
        plt.arrow(positions[0, -1], positions[1, -1], arrow_len * math.cos(end_heading), arrow_len * math.sin(end_heading), head_width=2, color='b')
    plt.axis('equal')

import math

from extract_lanes import cal_lane_start_heading
from utils import *
from settings import *


def extract_straight_tracks(all_tracks, grouped_straight_lane_segments, grouped_lane_width):
    straight_tracks = []
    for tracks in all_tracks:
        for track in tracks:
            for vector_straight_lane_segments, lane_width in zip(grouped_straight_lane_segments, grouped_lane_width):
                if straight_track_exist(track, vector_straight_lane_segments, lane_width):
                    straight_tracks.append(track)
                    break
    return straight_tracks


def straight_track_exist(track, vector_straight_lane_segments, lane_width):
    start_heading, _ = cal_track_start_heading(track)
    end_heading, _ = cal_track_end_heading(track)
    diff = abs(start_heading - end_heading) % (2 * math.pi)
    diff = min(diff, 2 * math.pi - diff)
    if diff < 30 * math.pi / 180:
        return True
    return False


def check_track_in_lane(track, lane_segments, lane_width):
    flag = True
    for state in track.object_states:
        position = state.position
        for key, value in lane_segments.items():
            dis2left = cal_dis_from_point_to_line(position, value.left_lane_boundary.xyz[0], value.left_lane_boundary.xyz[-1])
            dis2right = cal_dis_from_point_to_line(position, value.right_lane_boundary.xyz[0], value.right_lane_boundary.xyz[-1])
            if dis2left > lane_width[key] or dis2right > lane_width[key]:
                flag = False
                break
        if not flag:
            break
    return flag


def find_departure_point(track, end_lane_segments):
    for state in track.object_states:
        position = state.position
        for lane_segment in end_lane_segments:
            for key, value in lane_segment.items():
                if check_point_inside_polygon(position, value.left_lane_marking.polyline[:, :2].tolist() + value.right_lane_marking.polyline[::-1, :2].tolist()):
                    return state


def cal_track_start_heading(track):
    p1x, p1y = track.object_states[0].position[:2]
    angle = 0
    for state in track.object_states:
        p2x, p2y = state.position[:2]
        dis = cal_dis([p1x, p1y], [p2x, p2y])
        if dis > 3:
            angle = cal_angle(p2x - p1x, p2y - p1y)
            break
    return angle, [p2x, p2y]


def cal_track_end_heading(track):
    p2x, p2y = track.object_states[-1].position[:2]
    angle = 0
    for state in reversed(track.object_states):
        p1x, p1y = state.position[:2]
        dis = cal_dis([p1x, p1y], [p2x, p2y])
        if dis > 3:
            angle = cal_angle(p2x - p1x, p2y - p1y)
            break
    return angle, [p1x, p1y]

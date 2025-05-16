import numpy as np
import math

from utils import *


class FlexibleConflictPointPlanner:
    def __init__(self, AV_route, BV_route, scenario_info, init_condition, lane_width):
        self.ratio = init_condition["ratio"]
        self.triggerring_relative_dis = init_condition["dis"]
        self.triggerring_relative_sp = init_condition["relative sp"]
        self.direction = init_condition["direction"]
        AV_start_ind = scenario_info["AV"]["start ind"]
        AV_end_ind = scenario_info["AV"]["end ind"]
        AV_k, AV_b = np.polyfit(
            [point[0] for point in AV_route[AV_start_ind:AV_end_ind]],
            [point[1] for point in AV_route[AV_start_ind:AV_end_ind]],
            1,
        )
        BV_start_ind = scenario_info["challenger"]["start ind"]
        BV_end_ind = scenario_info["challenger"]["end ind"]
        start_b = (
            BV_route[BV_start_ind + 100][1] - AV_k * BV_route[BV_start_ind + 100][0]
        )

        k = AV_k
        mid_k = k
        mid_b = start_b + (AV_b - start_b) * self.ratio
        self.mid_line_point_1 = [
            BV_route[BV_start_ind][0],
            mid_k * BV_route[BV_start_ind][0] + mid_b,
        ]
        self.mid_line_point_2 = [
            BV_route[BV_end_ind][0],
            mid_k * BV_route[BV_end_ind][0] + mid_b,
        ]
        if self.direction == "same":
            self.angle = math.atan(1 / k)
        elif self.direction == "opposite":
            self.angle = math.atan(1 / k) + math.pi

        self.BV_route = BV_route
        self.BV_start_ind = BV_start_ind
        self.BV_end_ind = BV_end_ind
        self.NPN = 0

        self.beyond_flag = False  # if the distance between BV and AV is long enough
        self.triggerred_flag = False  # if the BV could start to change its lane
        self.generate_lc_route = False  # if the lane chagne route has been generated
        self.lc_complete_flag = False  # if the BV finish lane-change process
        self.complete_flag = False  # if the scenario is completed
        self.generate_after_lc_route = (
            False  # if the after lane change route has been generated
        )
        self.lc_complete_point = [0, 0]
        self.AV_hist = []

        self.lane_width = lane_width
        self.forward_time = 0.7
        self.lc_time = 3

        self.desired_speed = 0
        self.desired_position = [0, 0]
        self.desired_heading = 0
        self.time_step = 0.1

        self.dis_err = 0
        self.sp_err = 0
        self.init_AV_sp = 0
        self.init_timestamp = 0

    def planning(self, AV_state, BV_state):
        self._manage_state(AV_state, BV_state)
        self._set_desired_speed(AV_state, BV_state)
        self._generate_route(BV_state)
        self._set_desired_position(BV_state)
        command = {
            "type": "SetSumoTransform",
            "position": self.desired_position,
            "velocity": self.desired_speed,
            "angle": self.desired_heading,
        }
        return command

    def _manage_state(self, AV_state, BV_state):
        dis = cal_dis([AV_state.x, AV_state.y], [BV_state.x, BV_state.y])
        if len(self.AV_hist) > 10:
            self.AV_hist.pop(0)
        self.AV_hist.append([AV_state.timestamp, AV_state.speed, dis])
        if not self.beyond_flag:
            rotated_BV_position = rotate_point2_around_point1(
                [AV_state.x, AV_state.y],
                [BV_state.x, BV_state.y],
                (90 - AV_state.heading) / 180 * math.pi,
            )
            longitudinal_dis = rotated_BV_position[0] - AV_state.x
            forward_relative_longitudinal_dis = (
                self.triggerring_relative_dis
                + self.forward_time * self.triggerring_relative_sp
            )
            if (
                longitudinal_dis > forward_relative_longitudinal_dis
                and AV_state.speed > 4
            ):
                k, _ = np.polyfit(
                    [line[0] for line in self.AV_hist],
                    [line[1] for line in self.AV_hist],
                    1,
                )
                if abs(k) < 0.2:
                    self.beyond_flag = True
        if self.beyond_flag and not self.triggerred_flag:
            rotated_BV_position = rotate_point2_around_point1(
                [AV_state.x, AV_state.y],
                [BV_state.x, BV_state.y],
                (90 - AV_state.heading) / 180 * math.pi,
            )
            longitudinal_dis = rotated_BV_position[0] - AV_state.x
            horizontal_dis = abs(rotated_BV_position[1] - AV_state.y)
            if (
                longitudinal_dis < self.triggerring_relative_dis + 0.25
                and abs(horizontal_dis - self.lane_width) < 0.3
            ):
                self.dis_err = longitudinal_dis - self.triggerring_relative_dis
                if self.direction == "same":
                    self.sp_err = (
                        AV_state.speed - BV_state.speed - self.triggerring_relative_sp
                    )
                elif self.direction == "opposite":
                    self.sp_err = (
                        AV_state.speed + BV_state.speed - self.triggerring_relative_sp
                    )
                self.init_AV_sp = AV_state.speed
                self.init_timestamp = AV_state.timestamp
                self.triggerred_flag = True
        if self.beyond_flag and self.triggerred_flag and not self.lc_complete_flag:
            dis = distancPoint2Line(
                self.mid_line_point_1[0],
                self.mid_line_point_1[1],
                self.mid_line_point_2[0],
                self.mid_line_point_2[1],
                BV_state.x,
                BV_state.y,
            )
            if self.triggerred_flag and abs(dis) < 0.2:
                self.lc_complete_flag = True
                self.lc_complete_point = [BV_state.x, BV_state.y]
        if (
            self.beyond_flag
            and self.triggerred_flag
            and self.lc_complete_flag
            and not self.complete_flag
        ):
            rotated_BV_position = rotate_point2_around_point1(
                self.lc_complete_point, [BV_state.x, BV_state.y], -self.angle
            )
            dis = abs(rotated_BV_position[1] - self.lc_complete_point[1])
            if dis > 15:
                self.complete_flag = True

    def _set_desired_speed(self, AV_state, BV_state):
        if (
            not self.beyond_flag
            and not self.triggerred_flag
            and not self.lc_complete_flag
            and not self.complete_flag
        ):
            if self.direction == "same":
                if AV_state.speed > 0:
                    rotated_BV_position = rotate_point2_around_point1(
                        [AV_state.x, AV_state.y],
                        [BV_state.x, BV_state.y],
                        (90 - AV_state.heading) / 180 * math.pi,
                    )
                    longitudinal_dis = rotated_BV_position[0] - AV_state.x
                    forward_relative_longitudinal_dis = (
                        self.triggerring_relative_dis
                        + self.forward_time * self.triggerring_relative_sp
                    )
                    if longitudinal_dis > forward_relative_longitudinal_dis:
                        self.desired_speed = AV_state.speed
                    else:
                        self.desired_speed = AV_state.speed + min(
                            self.triggerring_relative_dis / 6, 10
                        )
                else:
                    self.desired_speed = 0
            elif self.direction == "opposite":
                self.desired_speed = 0
        if (
            self.beyond_flag
            and not self.triggerred_flag
            and not self.lc_complete_flag
            and not self.complete_flag
        ):
            if self.direction == "same":
                self.desired_speed = max(
                    AV_state.speed - self.triggerring_relative_sp, 1
                )
            elif self.direction == "opposite":
                self.desired_speed = self.triggerring_relative_sp - AV_state.speed
        if (
            self.beyond_flag
            and self.triggerred_flag
            and not self.lc_complete_flag
            and not self.complete_flag
        ):
            # once the lane change is started, BV speed is constant
            pass
        if (
            self.beyond_flag
            and self.triggerred_flag
            and self.lc_complete_flag
            and not self.complete_flag
        ):
            # when the lane change is just finished, BV will keep the speed for a while
            pass
        if (
            self.beyond_flag
            and self.triggerred_flag
            and self.lc_complete_flag
            and self.complete_flag
        ):
            self.desired_speed = 10

    def _generate_route(self, BV_state):
        self.NPN, _ = check_in_route(
            self.BV_route, [BV_state.x, BV_state.y], self.NPN, self.lane_width
        )
        if not self.triggerred_flag:
            self._generate_straight_route()
        if self.triggerred_flag and not self.generate_lc_route:
            self.generate_lc_route = True
            self._generate_sin_route(
                [BV_state.x, BV_state.y],
                self.ratio * self.lane_width / 2,
                self.lc_time * max(self.desired_speed, 1),
            )
            # since the route is changed, NPN should be recalculated
            self.NPN, _ = check_in_route(
                self.BV_route, [BV_state.x, BV_state.y], self.NPN, self.lane_width
            )
        if self.lc_complete_flag and not self.complete_flag:
            self._generate_straight_route()
        if self.complete_flag and not self.generate_after_lc_route:
            self.generate_after_lc_route = True
            if self.direction == "same" and BV_state.scenario == "Cut-In":
                amplitude = (1 - self.ratio) * self.lane_width / 2
            else:
                amplitude = -self.ratio * self.lane_width / 2
            self._generate_sin_route(
                [BV_state.x, BV_state.y],
                amplitude,
                self.lc_time * max(self.desired_speed, 1),
            )
            # since the route is changed, NPN should be recalculated
            self.NPN, _ = check_in_route(
                self.BV_route, [BV_state.x, BV_state.y], self.NPN, self.lane_width
            )

    def _generate_straight_route(self):
        if not self.triggerred_flag:
            ref_point_1 = self.BV_route[self.BV_start_ind][:2]
            ref_point_2 = self.BV_route[self.BV_start_ind + 10][:2]
        elif self.triggerred_flag and self.lc_complete_flag and not self.complete_flag:
            ref_point_1 = self.mid_line_point_1
            ref_point_2 = self.mid_line_point_2
        elif self.complete_flag:
            ref_point_1 = self.BV_route[self.BV_end_ind][:2]
            ref_point_2 = self.BV_route[self.BV_end_ind - 10][:2]
        for i in range(self.NPN, self.NPN + 20):
            dis = distancPoint2Line(
                ref_point_1[0],
                ref_point_1[1],
                ref_point_2[0],
                ref_point_2[1],
                self.BV_route[i][0],
                self.BV_route[i][1],
            )
            if abs(dis) > 0.2:
                heading = self.BV_route[i][2] / 180 * math.pi
                self.BV_route[i][0] += dis * math.cos(heading)
                self.BV_route[i][1] += -dis * math.sin(heading)

    def _generate_sin_route(self, start_point, amplitude, wavelength):
        # generate half-sin route
        ind = self.NPN
        x = []
        y = []
        while ind < len(self.BV_route):
            rotated_route_point = rotate_point2_around_point1(
                start_point, [self.BV_route[ind][0], self.BV_route[ind][1]], -self.angle
            )
            dx = rotated_route_point[1] - start_point[1]
            if dx > 0:
                dy = (
                    amplitude * math.sin(math.pi * (dx / wavelength - 1 / 2))
                    + amplitude
                )
                k = (
                    amplitude
                    * math.cos(math.pi * (dx / wavelength - 1 / 2))
                    * math.pi
                    / wavelength
                )
                heading = math.atan(k)
                adjusted_rotated_route_point = [
                    start_point[0] - dy,
                    rotated_route_point[1],
                ]
                self.BV_route[ind][:2] = rotate_point2_around_point1(
                    start_point, adjusted_rotated_route_point, self.angle
                )
                heading = (self.angle - heading) / math.pi * 180
                if heading < 0:
                    heading += 360
                self.BV_route[ind][2] = heading
                x.append(dx)
                y.append(dy)
                if 2 * abs(amplitude) - abs(dy) < 0.1:
                    break
            ind += 1

    def _set_desired_position(self, BV_state):
        NPN = self.NPN
        if NPN < len(self.BV_route) - 1:
            distanceToLaneCenter = distancPoint2Line(
                self.BV_route[NPN][0],
                self.BV_route[NPN][1],
                self.BV_route[NPN + 1][0],
                self.BV_route[NPN + 1][1],
                BV_state.x,
                BV_state.y,
            )
        else:
            distanceToLaneCenter = distancPoint2Line(
                self.BV_route[NPN][0],
                self.BV_route[NPN][1],
                self.BV_route[NPN - 1][0],
                self.BV_route[NPN - 1][1],
                BV_state.x,
                BV_state.y,
            )
        NNPN = find_next_route_points(self.BV_route, [BV_state.x, BV_state.y], NPN)
        point1 = self.BV_route[NNPN - 1][:2]
        heading1 = self.BV_route[NNPN - 1][2] / 180 * math.pi
        dis1 = cal_dis(point1, [BV_state.x, BV_state.y])
        point2 = self.BV_route[NNPN][:2]
        heading2 = self.BV_route[NNPN][2] / 180 * math.pi
        dis2 = cal_dis(point2, [BV_state.x, BV_state.y])
        if abs(heading1 - heading2) > math.pi:
            if heading1 > heading2:
                heading1 -= 2 * math.pi
            else:
                heading2 -= 2 * math.pi
        heading = heading1 * dis2 / (dis1 + dis2) + heading2 * dis1 / (dis1 + dis2)
        x = (
            BV_state.x
            + self.desired_speed * self.time_step * math.sin(heading)
            + distanceToLaneCenter * math.cos(heading)
        )
        y = (
            BV_state.y
            + self.desired_speed * self.time_step * math.cos(heading)
            - distanceToLaneCenter * math.sin(heading)
        )
        self.desired_position = [x, y]
        self.desired_heading = heading / math.pi * 180

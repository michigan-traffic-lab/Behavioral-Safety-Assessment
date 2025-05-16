from utils import *


class FixedConflictPointPlanner():
    def __init__(self, AV_route, BV_route, scenario_info, init_condition):
        self.AV_route = AV_route
        self.AV_start_ind = scenario_info['AV']['start ind']
        self.AV_end_ind = scenario_info['AV']['end ind']
        self.AV_conflict_ind = scenario_info['AV']['conflict ind']

        self.BV_route = BV_route
        self.BV_start_ind = scenario_info['challenger']['start ind']
        self.BV_end_ind = scenario_info['challenger']['end ind']
        self.BV_conflict_ind = scenario_info['challenger']['conflict ind']
        self.BV_init_ind = scenario_info['challenger']['init ind']

        self.init_dis = init_condition['dis']
        self.init_sp = init_condition['sp']

        self.init_flag = False
        self.start_flag = False
        self.complete_flag = False

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
        self._set_desired_speed_patch(AV_state, BV_state)
        self._set_desired_position(BV_state)
        command = {
            "type": "SetSumoTransform",
            "position": self.desired_position,
            "velocity": self.desired_speed,
            "angle": self.desired_heading
        }
        return command

    def _manage_state(self, AV_state, BV_state):
        if not self.init_flag and not self.start_flag and not self.complete_flag:
            if AV_state.NPN > self.AV_start_ind and AV_state.NPN < self.AV_conflict_ind and BV_state.NPN < self.BV_init_ind:
                self.init_flag = True
        elif self.init_flag and not self.start_flag and not self.complete_flag:
            if AV_state.scenario == 'Merge':
                dis_err = self.AV_route[self.AV_conflict_ind][1] - AV_state.y - self.init_dis
            else:
                dis_err = dis_from_ind_to_ind(self.AV_route, AV_state.NPN, self.AV_conflict_ind) - self.init_dis
            if BV_state.NPN >= self.BV_init_ind and dis_err < 0.5:
                if AV_state.scenario == 'Merge':
                    self.dis_err = self.AV_route[self.AV_conflict_ind][1] - AV_state.y - self.init_dis
                else:
                    self.dis_err = dis_from_ind_to_ind(self.AV_route, AV_state.NPN, self.AV_conflict_ind) - self.init_dis
                self.sp_err = BV_state.speed - self.init_sp
                self.init_AV_sp = AV_state.speed
                self.init_timestamp = AV_state.timestamp
                self.start_flag = True
        elif self.init_flag and self.start_flag and not self.complete_flag:
            if (AV_state.NPN > self.AV_conflict_ind and BV_state.NPN > self.BV_conflict_ind) or BV_state.speed == 0:
                self.complete_flag = True

    def _set_desired_speed(self, AV_state, BV_state):
        if not self.init_flag and not self.start_flag and not self.complete_flag:
            self.desired_speed = 0
        elif self.init_flag and not self.start_flag and not self.complete_flag:
            if AV_state.scenario == 'Merge':
                AV_dis = self.AV_route[self.AV_conflict_ind][1] - AV_state.y
            else:
                if AV_state.NPN + 1 >= self.AV_conflict_ind:
                    AV_dis = cal_dis([AV_state.x, AV_state.y], self.AV_route[self.AV_conflict_ind][:2])
                else:
                    AV_dis = dis_from_ind_to_ind(self.AV_route, AV_state.NPN + 1, self.AV_conflict_ind) + cal_dis([AV_state.x, AV_state.y], self.AV_route[AV_state.NPN + 1][:2])
            AV_delta_dis = AV_dis - self.init_dis
            tf = AV_delta_dis / max(AV_state.speed, 0.01)  # - 0.3
            if self.BV_init_ind >= BV_state.NPN >= self.BV_init_ind - 1:
                BV_dis = cal_dis([BV_state.x, BV_state.y], self.BV_route[self.BV_init_ind + 1][:2]) - cal_dis(self.BV_route[self.BV_init_ind][:2], self.BV_route[self.BV_init_ind + 1][:2])
            elif BV_state.NPN >= self.BV_init_ind + 1:
                BV_dis = - dis_from_ind_to_ind(self.BV_route, BV_state.NPN - 1, self.BV_init_ind) - cal_dis([BV_state.x, BV_state.y], self.BV_route[BV_state.NPN - 1][:2])
            else:
                BV_dis = dis_from_ind_to_ind(self.BV_route, BV_state.NPN + 1, self.BV_init_ind) + cal_dis([BV_state.x, BV_state.y], self.BV_route[BV_state.NPN + 1][:2])
            vf = self.init_sp
            vi = BV_state.speed
            A = [[tf ** 2 / 2, tf], [tf ** 3 / 6, tf ** 2 / 2]]
            B = [vf - vi, BV_dis - tf * vi]
            k_aa = (A[0][1] * B[1] - A[1][1] * B[0]) / (A[0][1] * A[1][0] - A[0][0] * A[1][1])
            b_aa = (A[0][0] * B[1] - A[1][0] * B[0]) / (A[0][0] * A[1][1] - A[0][1] * A[1][0])
            if b_aa < -6:
                b_aa = -6
            self.desired_speed = BV_state.speed + self.time_step * b_aa
            if tf > 100:
                self.desired_speed = 0
            if self.desired_speed > 10:
                self.desired_speed = 10
            elif self.desired_speed < 0:
                self.desired_speed = 0
            if AV_state.speed < 0.1 and BV_state.NPN > self.BV_init_ind - 5:
                self.desired_speed = self.init_sp  # if AV yield to BV before AV reaches the initial position, then BV speed is set to the initial speed
        elif self.init_flag and self.start_flag and not self.complete_flag:
            self.desired_speed = self.init_sp
        elif self.init_flag and self.start_flag and self.complete_flag:
            self.desired_speed = self.init_sp

    def _set_desired_position(self, BV_state):
        NPN = BV_state.NPN
        if NPN < len(self.BV_route) - 1:
            distanceToLaneCenter = distancPoint2Line(self.BV_route[NPN][0], self.BV_route[NPN][1], self.BV_route[NPN+1][0], self.BV_route[NPN+1][1], BV_state.x, BV_state.y)
        else:
            distanceToLaneCenter = distancPoint2Line(self.BV_route[NPN][0], self.BV_route[NPN][1], self.BV_route[NPN-1][0], self.BV_route[NPN-1][1], BV_state.x, BV_state.y)
        if BV_state.speed == 0:
            if BV_state.timestamp == 0:
                self.desired_heading = self.BV_route[NPN][2]
            else:
                self.desired_heading = BV_state.heading
        else:
            NNPN = find_next_route_points(self.BV_route, [BV_state.x, BV_state.y], NPN)
            point1 = self.BV_route[NNPN-1][:2]
            heading1 = self.BV_route[NNPN-1][2]
            dis1 = cal_dis(point1, [BV_state.x, BV_state.y])
            point2 = self.BV_route[NNPN][:2]
            heading2 = self.BV_route[NNPN][2]
            dis2 = cal_dis(point2, [BV_state.x, BV_state.y])
            if abs(heading1 - heading2) > 180:
                if heading1 > heading2:
                    heading1 -= 2 * 180
                else:
                    heading2 -= 2 * 180
            self.desired_heading = heading1 * dis2 / (dis1 + dis2) + heading2 * dis1 / (dis1 + dis2)
        self.desired_heading = self.BV_route[NPN][2]
        delta_x = BV_state.speed * self.time_step * math.sin(self.desired_heading / 180 * math.pi) + distanceToLaneCenter * math.cos(self.desired_heading / 180 * math.pi)
        delta_y = BV_state.speed * self.time_step * math.cos(self.desired_heading / 180 * math.pi) - distanceToLaneCenter * math.sin(self.desired_heading / 180 * math.pi)
        delta_d = math.sqrt(delta_x ** 2 + delta_y ** 2)
        if delta_d > 0:
            x = BV_state.x + delta_x * BV_state.speed * self.time_step / delta_d
            y = BV_state.y + delta_y * BV_state.speed * self.time_step / delta_d
        else:
            x, y = BV_state.x, BV_state.y
        self.desired_position = [x, y]

    def _set_desired_speed_patch(self, AV_state, BV_state):
        if self.init_flag and self.start_flag:
            if AV_state.scenario == 'Right Turn (AV turns right)':
                if AV_state.y > 209 and AV_state.y > BV_state.y:
                    self.desired_speed = max(BV_state.speed - self.time_step * 7.06, 0)
            elif AV_state.scenario == 'Merge':
                if AV_state.x - BV_state.x < 2 and 0 < AV_state.y - BV_state.y < 10:
                    self.desired_speed = max(BV_state.speed - self.time_step * 7.06, AV_state.speed)

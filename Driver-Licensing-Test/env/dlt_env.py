from terasim.envs.template import EnvTemplate

import time
from bs4 import BeautifulSoup
import traci

from env.state_manager import *
from traffic_signal.traffic_signal_decision_model import TrafficSignalDecisionModel
from settings import *
from utils import check_overlap_of_rectangles


class DLTEnv(EnvTemplate):
    """This Env provides a basic Env implementation.

    Env developers can derived from this class or build their own implementations directly on BaseEnv
    """

    def __init__(
        self, vehicle_factory, info_extractor, init_conditions, case_num, settings
    ):
        super().__init__(vehicle_factory, info_extractor)

        self.test_name = settings.test_name
        self.record = {scenario: [] for scenario in settings.scenario_list}
        self.record_scenario = None
        self.scenario_list = settings.scenario_list
        self.next_scenario = None
        self.case_num = case_num
        self.test_data_path = settings.test_data_path

        self.AV_route = settings.AV_route
        self.AV_sumo_route_name = settings.AV_sumo_route_name
        self.AV_start_lane_id = settings.AV_start_lane_id
        self.AV_length = settings.AV_length
        self.AV_width = settings.AV_width

        self.BV_num = settings.BV_num
        self.BV_routes = settings.BV_routes
        self.BV_start_lane_id = settings.BV_start_lane_id
        self.BV_scenarios = settings.BV_scenarios
        self.BV_init_position = settings.BV_init_position
        self.BV_length = settings.BV_length
        self.BV_width = settings.BV_width

        self.VRU_num = settings.VRU_num
        self.VRU_start_lane_id = settings.VRU_start_lane_id
        self.VRU_scenarios = settings.VRU_scenarios
        self.VRU_length = settings.VRU_length
        self.VRU_width = settings.VRU_width

        self.traffic_signal_num = settings.traffic_signal_num
        self.traffic_signal_scenarios = settings.traffic_signal_scenarios
        self.traffic_signal_list = {}

        self.risk_level = init_conditions["risk_level"]
        self.init_conditions = init_conditions["case_init_conditions"]
        self.original_init_conditions = init_conditions["original_case_init_conditions"]
        self.collision = {scenario: False for scenario in settings.scenario_list}

        self.AV_route_point = []

        self.state_manager = StateManager(settings)

        self.settings = settings

    def on_start(self, ctx):
        # your initialization (vehicle position, etc.) IDM_waymo_motion
        self.add_vehicle(
            veh_id="CAV",
            route_id=self.AV_sumo_route_name,
            lane="first",
            lane_id=self.AV_start_lane_id,
            position=0,
            speed=0,
            type_id="AV",
        )

        # set traffic signal initial state
        with open(self.settings.sumo_map_file, "r") as f:
            data = f.read()
        Bs_data = BeautifulSoup(data, "xml")
        for node in Bs_data.find_all("tlLogic"):
            traffic_signal_state_len = len(
                traci.trafficlight.getRedYellowGreenState(node["id"])
            )
            traffic_signal_state = ""
            for _ in range(traffic_signal_state_len):
                traffic_signal_state += "O"
            traci.trafficlight.setRedYellowGreenState(node["id"], traffic_signal_state)

    def on_step(self, ctx):
        # Make decisions and execute commands
        self._add_vehicle()
        self._del_vehicle()
        control_cmds, infos = self.make_decisions()
        self.execute_control_commands(control_cmds)
        self._save_test_data()
        time.sleep(0.1)

        # Simulation stop check
        return self.should_continue_simulation()

    def on_stop(self, ctx):
        pass

    def make_decisions(self):
        """Make decisions for all vehicles."""
        all_vehicle_observation = {
            veh.id: veh.observation["ego"]['data'] if 'data' in veh.observation["ego"] else veh.observation["ego"]
            for veh in self.vehicle_list
        }
        for veh in self.vehicle_list:
            if 'timestamp' in veh.observation["ego"]:
                current_time = veh.observation["ego"]['timestamp']
            else:
                current_time = traci.simulation.getCurrentTime() / 1000.0 - 0.1  # observe the last step
            break

        # manage the state of all vehicles
        self.state_manager.update_agent_state(current_time, all_vehicle_observation)
        self.state_manager.update_scenario_state()

        # collision check
        if (
            self._collision_detect()
            and self.state_manager.AV_state.scenario is not None
        ):
            self.collision[self.state_manager.AV_state.scenario] = True

        for veh_id in list(self.vehicle_list.keys()):
            if "BV" in veh_id:
                # generate BV cmd
                i = int(veh_id[-1]) - 1
                self.vehicle_list[veh_id].decision_model.update(
                    self.state_manager.AV_state, self.state_manager.BV_states[i]
                )
                self.state_manager.update_agent_routes(
                    i, self.vehicle_list[veh_id].decision_model.route
                )
            if "VRU" in veh_id:
                # generate VRU cmd
                i = int(veh_id[-1]) - 1
                self.vehicle_list[veh_id].decision_model.update(
                    self.state_manager.AV_state, self.state_manager.VRU_states[i]
                )

        # traffic signal control
        for traffic_signal_decision_model in list(self.traffic_signal_list.values()):
            traffic_signal_decision_model.update(self.state_manager.AV_state)
            traffic_signal_decision_model.derive_control_command_from_observation(
                self.state_manager.AV_state
            )

        # by default, all vehicles in the vehicle list will make decisions
        control_command_and_info = {
            veh.id: veh.make_decision() for veh in self.vehicle_list
        }
        control_command_dict = {
            veh_id: command_and_info[0]
            for veh_id, command_and_info in control_command_and_info.items()
        }
        info_dict = {
            veh_id: command_and_info[1]
            for veh_id, command_and_info in control_command_and_info.items()
        }
        return control_command_dict, info_dict

    def execute_control_commands(self, control_commands: dict):
        """Execute the control commands of all vehicles."""
        for veh_id, command in control_commands.items():
            self.vehicle_list[veh_id].apply_control(command)

    def should_continue_simulation(self):
        # By default, the simulation will stop when all vehicles leave the network
        if self.simulator.get_vehicle_min_expected_number() == 0:
            return False

        # You can also define your own termination condition, e.g., when the AV reaches the destination,
        # when collisions between AV and BV happen, etc.
        if self.state_manager.AV_state.complete or (self.state_manager.AV_state.scenario and self.collision[self.state_manager.AV_state.scenario]):
            return False  # dict(reason = "All Vehicles Left", info = {})  # All Vehicles Left

        # Otherwise return True to continue simulation
        return True

    def _add_vehicle(self):
        # update current scenario
        if self.state_manager.scenario_states["start"]:
            current_scenario = None
            if len(self.scenario_list) > 0:
                self.next_scenario = self.scenario_list[0]
        else:
            current_scenario = self.state_manager.AV_state.scenario
            if current_scenario is not None:
                ind = self.scenario_list.index(current_scenario)
                if ind == len(self.scenario_list) - 1:
                    self.next_scenario = None
                else:
                    self.next_scenario = self.scenario_list[ind + 1]

        # add BV
        if self.BV_num > 0:
            for BV_scenario_list, ind in zip(
                self.state_manager.BV_scenario_lists,
                list(range(len(self.state_manager.BV_scenario_lists))),
            ):
                if self.next_scenario == BV_scenario_list[0] and not "BV " + str(
                    ind + 1
                ) in list(self.vehicle_list.keys()):
                    position = self.BV_init_position[ind]
                    self.add_vehicle(
                        veh_id="BV " + str(ind + 1),
                        route_id="BV_route_" + str(ind + 1),
                        lane="first",
                        lane_id=self.BV_start_lane_id[ind],
                        position=position,
                        speed=0,
                        type_id="BV",
                    )
                    conditions = {}
                    for scenario in list(self.init_conditions.keys()):
                        if scenario in self.BV_scenarios[ind]:
                            conditions[scenario] = self.init_conditions[scenario]
                    self.vehicle_list["BV " + str(ind + 1)].decision_model.init(
                        ind + 1, conditions, self.settings
                    )

        # add VRU
        if self.VRU_num > 0:
            for VRU_scenario_list, ind in zip(
                self.state_manager.VRU_scenario_lists,
                list(range(len(self.state_manager.VRU_scenario_lists))),
            ):
                if self.next_scenario == VRU_scenario_list[0] and not "VRU " + str(
                    ind + 1
                ) in list(self.vehicle_list.keys()):
                    # traci.person.add('VRU ' + str(ind+1), 'VRU_route_' + str(ind+1))
                    # traci.person.setColor('VRU ' + str(ind+1), (255, 150, 150, 255))
                    # traci.person.moveToXY(pedID, "", 0, x, y)
                    self.add_vehicle(
                        veh_id="VRU " + str(ind + 1),
                        route_id="VRU_route_" + str(ind + 1),
                        lane="first",
                        lane_id=self.VRU_start_lane_id[ind],
                        position=0,
                        speed=0,
                        type_id="VRU",
                    )
                    conditions = {}
                    for scenario in list(self.init_conditions.keys()):
                        if scenario in self.VRU_scenarios[ind]:
                            conditions[scenario] = self.init_conditions[scenario]
                    self.vehicle_list["VRU " + str(ind + 1)].decision_model.init(
                        ind + 1, conditions, self.settings
                    )

        # add traffic signal
        if self.traffic_signal_num > 0:
            for traffic_signal_scenario_list, ind in zip(
                self.state_manager.traffic_signal_scenario_lists,
                list(range(len(self.state_manager.traffic_signal_scenario_lists))),
            ):
                if self.next_scenario == traffic_signal_scenario_list[0]:
                    conditions = {}
                    for scenario in list(self.init_conditions.keys()):
                        if scenario in self.traffic_signal_scenarios[ind]:
                            conditions[scenario] = self.init_conditions[scenario]
                    self.traffic_signal_list["Traffic Signal " + str(ind + 1)] = (
                        TrafficSignalDecisionModel()
                    )
                    self.traffic_signal_list["Traffic Signal " + str(ind + 1)].init(
                        ind + 1, conditions, self.settings
                    )

    def _del_vehicle(self):
        # delete BV or VRU
        for veh_id in list(self.vehicle_list.keys()):
            if veh_id != "CAV":
                last_scenario = self.vehicle_list[veh_id].decision_model.scenario_list[
                    -1
                ]
                if self.state_manager.scenario_states[last_scenario].complete:
                    self.remove_vehicle(veh_id)

    def _collision_detect(self):
        AV_cx = self.state_manager.AV_state.x - self.AV_length / 2 * math.sin(self.state_manager.AV_state.heading / 180 * math.pi)
        AV_cy = self.state_manager.AV_state.y - self.AV_length / 2 * math.cos(self.state_manager.AV_state.heading / 180 * math.pi)
        AV_rectangle = [AV_cx, AV_cy, self.AV_length, self.AV_width, (90 - self.state_manager.AV_state.heading) / 180 * math.pi]
        for i, BV_state in enumerate(self.state_manager.BV_states):
            BV_cx = BV_state.x - self.BV_length[i] / 2 * math.sin(BV_state.heading / 180 * math.pi)
            BV_cy = BV_state.y - self.BV_length[i] / 2 * math.cos(BV_state.heading / 180 * math.pi)
            BV_rectangle = [BV_cx, BV_cy, self.BV_length[i], self.BV_width[i], (90 - BV_state.heading) / 180 * math.pi]
            if check_overlap_of_rectangles(AV_rectangle, BV_rectangle):
                return True
        for i, VRU_state in enumerate(self.state_manager.VRU_states):
            VRU_cx = VRU_state.x - self.VRU_length[i] / 2 * math.sin(VRU_state.heading / 180 * math.pi)
            VRU_cy = VRU_state.y - self.VRU_length[i] / 2 * math.cos(VRU_state.heading / 180 * math.pi)
            VRU_rectangle = [VRU_cx, VRU_cy, self.VRU_length[i], self.VRU_width[i], (90 - VRU_state.heading) / 180 * math.pi]
            if check_overlap_of_rectangles(AV_rectangle, VRU_rectangle):
                return True
        return False

    def _save_test_data(self):
        if self.state_manager.AV_state_hist[-1].scenario is not None:
            for BV_state_hist in self.state_manager.BV_states_hist:
                if BV_state_hist[-1].scenario is None:
                    continue
                if (
                    self.state_manager.AV_state_hist[-1].scenario
                    != BV_state_hist[-1].scenario
                ):
                    continue
                if self.collision[BV_state_hist[-1].scenario]:
                    continue
                if self.state_manager.scenario_states[
                    BV_state_hist[-1].scenario
                ].complete:
                    continue
                self.record[self.state_manager.AV_state_hist[-1].scenario].append(
                    [
                        self.state_manager.AV_state_hist[0].timestamp,
                        self.state_manager.AV_state_hist[0].x,
                        self.state_manager.AV_state_hist[0].y,
                        self.state_manager.AV_state_hist[0].speed,
                        self.state_manager.AV_state_hist[0].lon_speed,
                        self.state_manager.AV_state_hist[0].lat_speed,
                        self.state_manager.AV_state_hist[0].acceleration,
                        self.state_manager.AV_state_hist[0].lon_acceleration,
                        self.state_manager.AV_state_hist[0].lat_acceleration,
                        self.state_manager.AV_state_hist[0].heading,
                        BV_state_hist[0].x,
                        BV_state_hist[0].y,
                        BV_state_hist[0].speed,
                        BV_state_hist[0].lon_speed,
                        BV_state_hist[0].lat_speed,
                        BV_state_hist[0].acceleration,
                        BV_state_hist[0].lon_acceleration,
                        BV_state_hist[0].lat_acceleration,
                        BV_state_hist[0].heading,
                    ]
                )
                self.record_scenario = self.state_manager.AV_state_hist[-1].scenario

            for VRU_state_hist in self.state_manager.VRU_states_hist:
                if VRU_state_hist[-1].scenario is None:
                    continue
                if (
                    self.state_manager.AV_state_hist[-1].scenario
                    != VRU_state_hist[-1].scenario
                ):
                    continue
                if self.collision[VRU_state_hist[-1].scenario]:
                    continue
                if self.state_manager.scenario_states[
                    VRU_state_hist[-1].scenario
                ].complete:
                    continue
                self.record[self.state_manager.AV_state_hist[-1].scenario].append(
                    [
                        self.state_manager.AV_state_hist[0].timestamp,
                        self.state_manager.AV_state_hist[0].x,
                        self.state_manager.AV_state_hist[0].y,
                        self.state_manager.AV_state_hist[0].speed,
                        self.state_manager.AV_state_hist[0].lon_speed,
                        self.state_manager.AV_state_hist[0].lat_speed,
                        self.state_manager.AV_state_hist[0].acceleration,
                        self.state_manager.AV_state_hist[0].lon_acceleration,
                        self.state_manager.AV_state_hist[0].lat_acceleration,
                        self.state_manager.AV_state_hist[0].heading,
                        VRU_state_hist[0].x,
                        VRU_state_hist[0].y,
                        VRU_state_hist[0].speed,
                        VRU_state_hist[0].lon_speed,
                        VRU_state_hist[0].lat_speed,
                        VRU_state_hist[0].acceleration,
                        VRU_state_hist[0].lon_acceleration,
                        VRU_state_hist[0].lat_acceleration,
                        VRU_state_hist[0].heading,
                    ]
                )
                self.record_scenario = self.state_manager.AV_state_hist[-1].scenario

            for traffic_signal_state in self.state_manager.traffic_signal_states:
                if (
                    self.state_manager.AV_state_hist[-1].scenario
                    not in traffic_signal_state.scenario
                ):
                    continue
                if self.state_manager.scenario_states[
                    self.state_manager.AV_state_hist[-1].scenario
                ].complete:
                    continue
                self.record[self.state_manager.AV_state_hist[-1].scenario].append(
                    [
                        self.state_manager.AV_state_hist[0].timestamp,
                        self.state_manager.AV_state_hist[0].x,
                        self.state_manager.AV_state_hist[0].y,
                        self.state_manager.AV_state_hist[0].speed,
                        self.state_manager.AV_state_hist[0].lon_speed,
                        self.state_manager.AV_state_hist[0].lat_speed,
                        self.state_manager.AV_state_hist[0].acceleration,
                        self.state_manager.AV_state_hist[0].lon_acceleration,
                        self.state_manager.AV_state_hist[0].lat_acceleration,
                        self.state_manager.AV_state_hist[0].heading,
                        traffic_signal_state.state,
                    ]
                )
                self.record_scenario = self.state_manager.AV_state_hist[-1].scenario

        if self.record_scenario is not None:
            if (
                self.state_manager.scenario_states[self.record_scenario].complete
                or self.collision[self.record_scenario]
            ):
                for BV_state_hist in self.state_manager.BV_states_hist:
                    for i in range(1, len(BV_state_hist)):
                        self.record[self.state_manager.AV_state_hist[-1].scenario].append(
                            [
                                self.state_manager.AV_state_hist[i].timestamp,
                                self.state_manager.AV_state_hist[i].x,
                                self.state_manager.AV_state_hist[i].y,
                                self.state_manager.AV_state_hist[i].speed,
                                self.state_manager.AV_state_hist[i].lon_speed,
                                self.state_manager.AV_state_hist[i].lat_speed,
                                self.state_manager.AV_state_hist[i].acceleration,
                                self.state_manager.AV_state_hist[i].lon_acceleration,
                                self.state_manager.AV_state_hist[i].lat_acceleration,
                                self.state_manager.AV_state_hist[i].heading,
                                BV_state_hist[i].x,
                                BV_state_hist[i].y,
                                BV_state_hist[i].speed,
                                BV_state_hist[i].lon_speed,
                                BV_state_hist[i].lat_speed,
                                BV_state_hist[i].acceleration,
                                BV_state_hist[i].lon_acceleration,
                                BV_state_hist[i].lat_acceleration,
                                BV_state_hist[i].heading,
                            ]
                        )
                for VRU_state_hist in self.state_manager.VRU_states_hist:
                    for i in range(1, len(VRU_state_hist)):
                        self.record[self.state_manager.AV_state_hist[-1].scenario].append(
                            [
                                self.state_manager.AV_state_hist[i].timestamp,
                                self.state_manager.AV_state_hist[i].x,
                                self.state_manager.AV_state_hist[i].y,
                                self.state_manager.AV_state_hist[i].speed,
                                self.state_manager.AV_state_hist[i].lon_speed,
                                self.state_manager.AV_state_hist[i].lat_speed,
                                self.state_manager.AV_state_hist[i].acceleration,
                                self.state_manager.AV_state_hist[i].lon_acceleration,
                                self.state_manager.AV_state_hist[i].lat_acceleration,
                                self.state_manager.AV_state_hist[i].heading,
                                VRU_state_hist[i].x,
                                VRU_state_hist[i].y,
                                VRU_state_hist[i].speed,
                                VRU_state_hist[i].lon_speed,
                                VRU_state_hist[i].lat_speed,
                                VRU_state_hist[i].acceleration,
                                VRU_state_hist[i].lon_acceleration,
                                VRU_state_hist[i].lat_acceleration,
                                VRU_state_hist[i].heading,
                            ]
                        )
                for traffic_signal_states in self.state_manager.traffic_signal_states_hist:
                    for i in range(1, len(traffic_signal_states)):
                        self.record[self.state_manager.AV_state_hist[-1].scenario].append(
                            [
                                self.state_manager.AV_state_hist[i].timestamp,
                                self.state_manager.AV_state_hist[i].x,
                                self.state_manager.AV_state_hist[i].y,
                                self.state_manager.AV_state_hist[i].speed,
                                self.state_manager.AV_state_hist[i].lon_speed,
                                self.state_manager.AV_state_hist[i].lat_speed,
                                self.state_manager.AV_state_hist[i].acceleration,
                                self.state_manager.AV_state_hist[i].lon_acceleration,
                                self.state_manager.AV_state_hist[i].lat_acceleration,
                                self.state_manager.AV_state_hist[i].heading,
                                traffic_signal_states[i].state,
                            ]
                        )
                if self.record_scenario == "Traffic Signal":
                    keys = [
                        "timestamp",
                        "AV x",
                        "AV y",
                        "AV sp",
                        "AV lon sp",
                        "AV lat sp",
                        "AV acc",
                        "AV lon acc",
                        "AV lat acc",
                        "AV heading",
                        "traffic light state",
                    ]
                else:
                    keys = [
                        "timestamp",
                        "AV x",
                        "AV y",
                        "AV sp",
                        "AV lon sp",
                        "AV lat sp",
                        "AV acc",
                        "AV lon acc",
                        "AV lat acc",
                        "AV heading",
                        "challenger x",
                        "challenger y",
                        "challenger sp",
                        "challenger lon sp",
                        "challenger lat sp",
                        "challenger acc",
                        "challenger lon acc",
                        "challenger lat acc",
                        "challenger heading",
                    ]
                if len(self.record[self.record_scenario]) > 0:
                    if self.collision[self.record_scenario]:
                        self.record[self.record_scenario].append([0 for _ in range(len(keys))])
                    path = (
                        self.test_data_path
                        + "/"
                        + get_scenario_folder_name(self.record_scenario)
                    )
                    if not os.path.exists(path):
                        os.mkdir(path)
                    
                    save_csv(
                        self.record[self.record_scenario],
                        keys,
                        path,
                        str(self.case_num),
                        False,
                    )

                    file_name = path + "/record.csv"
                    if not os.path.exists(file_name):
                        file = open(file_name, "w")
                        keys = "risk level,"
                        for key in list(
                            self.original_init_conditions[self.record_scenario].keys()
                        ):
                            keys += key
                            keys += ","
                        keys += "init timestamp,dis err,sp err,AV init sp,collision\n"
                        file.writelines(keys)
                    else:
                        file = open(file_name, "a")
                    values = self.risk_level + ","
                    for value in list(
                        self.original_init_conditions[self.record_scenario].values()
                    ):
                        values += str(value)
                        values += ","
                    for veh_id in list(self.vehicle_list.keys()):
                        if veh_id != "CAV":
                            if (
                                self.record_scenario
                                in self.vehicle_list[
                                    veh_id
                                ].decision_model.scenario_list
                            ):
                                if "BV" in veh_id:
                                    values += (
                                        str(
                                            self.vehicle_list[veh_id]
                                            .decision_model.planner[
                                                self.record_scenario
                                            ]
                                            .init_timestamp
                                        )
                                        + ","
                                        + str(
                                            self.vehicle_list[veh_id]
                                            .decision_model.planner[
                                                self.record_scenario
                                            ]
                                            .dis_err
                                        )
                                        + ","
                                        + str(
                                            self.vehicle_list[veh_id]
                                            .decision_model.planner[
                                                self.record_scenario
                                            ]
                                            .sp_err
                                        )
                                        + ","
                                        + str(
                                            self.vehicle_list[veh_id]
                                            .decision_model.planner[
                                                self.record_scenario
                                            ]
                                            .init_AV_sp
                                        )
                                    )
                                elif "VRU" in veh_id:
                                    values += (
                                        str(
                                            self.vehicle_list[
                                                veh_id
                                            ].decision_model.init_timestamp[
                                                self.record_scenario
                                            ]
                                        )
                                        + ","
                                        + str(
                                            self.vehicle_list[
                                                veh_id
                                            ].decision_model.dis_err[
                                                self.record_scenario
                                            ]
                                        )
                                        + ","
                                        + str(
                                            self.vehicle_list[
                                                veh_id
                                            ].decision_model.sp_err[
                                                self.record_scenario
                                            ]
                                        )
                                        + ","
                                        + str(
                                            self.vehicle_list[
                                                veh_id
                                            ].decision_model.init_AV_sp[
                                                self.record_scenario
                                            ]
                                        )
                                    )
                    if self.record_scenario == "Traffic Signal":
                        values += (
                            str(
                                self.traffic_signal_list[
                                    "Traffic Signal 1"
                                ].init_timestamp[self.record_scenario]
                            )
                            + ","
                            + str(
                                self.traffic_signal_list["Traffic Signal 1"].dis_err[
                                    self.record_scenario
                                ]
                            )
                            + ","
                            + str(
                                self.traffic_signal_list["Traffic Signal 1"].sp_err[
                                    self.record_scenario
                                ]
                            )
                            + ","
                            + str(
                                self.traffic_signal_list["Traffic Signal 1"].init_AV_sp[
                                    self.record_scenario
                                ]
                            )
                        )
                    if self.record[self.record_scenario][-1][0] == 0:
                        values += ",1\n"
                    else:
                        values += ",0\n"
                    file.writelines(values)
                    file.close()

                    self.record[self.record_scenario] = []
                    self.record_scenario = None

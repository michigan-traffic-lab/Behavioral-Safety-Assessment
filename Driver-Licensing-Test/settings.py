import pathlib
import numpy as np
import os
import math
import yaml
from typing import List
from dataclasses import dataclass


class agentState:
    timestamp = 0.0
    x = 0.0
    y = 0.0
    speed = 0.0
    acceleration = 0.0
    heading = 0.0


DELTA_T = 0.1
turning_heading_threshold = 0.6
straight_heaading_threshold = 0.2

max_dec = -0.72 * 9.81
min_dec = -0.1
max_acc = 2.87
min_acc = 0.1
max_lc_acc = 1.81
lane_width = 3.8
min_lc_time = math.sqrt(lane_width / 2 / max_lc_acc)

COLORS = {
    "infs": "red",
    "high": "orange",
    "mid": "yellow",
    "low": "green",
    "trivial": "blue",
}

ALPHAS = {
    "infs": 0.6,
    "high": 0.6,
    "mid": 0.6,
    "low": 0.6,
    "trivial": 0.3,
}

FONTSIZE = 20


@dataclass
class RightTurnState:
    d1: float
    v1: float
    a1: float
    d2: float
    v2: float
    a2: float


@dataclass
class RightTurnData:
    AV_yeild: List[RightTurnState]
    BV_yeild: List[RightTurnState]


class Settings:
    root = str(pathlib.Path(__file__).parent.resolve())

    def __init__(self, test_name="test", test_round=0, scenario_info_folder="cut_in"):
        # set test basic info
        self.test_name = test_name
        self.time_step = DELTA_T
        self.risk_levels = ["infs", "high", "mid", "low", "trivial"]

        # set all scenario information
        if scenario_info_folder != "":
            self.test_info_path = (
                self.root + "/env/route/" + test_name + "/" + scenario_info_folder
            )
            self.sumo_map_file = self.test_info_path + "/map/mcity.net.xml"
            self.sumo_config_file = self.test_info_path + "/map/mcity_micro.sumocfg"

            # load yaml
            with open(self.test_info_path + "/config.yaml", "r") as test_info_file:
                self.test_info = yaml.safe_load(test_info_file)

            self.scenario_list = self.test_info["scenario list"]
            self.AV_speed_list = self.test_info["AV"]["speed"]
            self.AV_length = self.test_info["AV"]["length"]
            self.AV_width = self.test_info["AV"]["width"]
            self.BV_length = self.test_info["BV"]["length"]
            self.BV_width = self.test_info["BV"]["width"]
            self.VRU_length = self.test_info["VRU"]["length"]
            self.VRU_width = self.test_info["VRU"]["width"]
            self.scenario_category = {
                "fixed conflict point": [
                    "VRU Crossing the Street without Crosswalk",
                    "Right Turn (AV goes straight)",
                    "Right Turn (AV turns right)",
                    "VRU Crossing the Street at the Crosswalk",
                    "Left Turn (AV goes straight)",
                    "Left Turn (AV turns left)",
                    "AV Merging into the Roundabout",
                    "BV Merging into the Roundabout",
                    "Merge",
                ],
                "flexible conflict point": [
                    "Cut-In",
                    "Lane Departure (same direction)",
                    "Lane Departure (opposite direction)",
                ],
                "no conflict point": ["Car Following"],
                "single agent": ["Vehicle Encroachment", "Traffic Signal"],
            }
            self.scenario_info = self.test_info["scenario info"]

            # set AV information
            self.AV_sumo_route_name = self.test_info["AV"]["sumo route name"]
            self.AV_start_lane_id = self.test_info["AV"]["start lane id"]
            self.AV_route = np.loadtxt(self.test_info_path + "/AV_route.txt")
            self.AV_route_path = self.test_info_path + "/AV_route.txt"

            # set BV information
            self.BV_num = self.test_info["BV"]["num"]
            self.BV_routes = [
                np.loadtxt(self.test_info_path + "/BV_route_" + str(i + 1) + ".txt")
                for i in range(self.BV_num)
            ]
            self.BV_scenarios = self.test_info["BV"]["scenarios"]
            self.BV_start_lane_id = self.test_info["BV"]["start lane id"]
            self.BV_init_position = self.test_info["BV"]["init position"]

            # set VRU information
            self.VRU_num = self.test_info["VRU"]["num"]
            self.VRU_routes = [
                np.loadtxt(self.test_info_path + "/VRU_route_" + str(i + 1) + ".txt")
                for i in range(self.VRU_num)
            ]
            self.VRU_scenarios = self.test_info["VRU"]["scenarios"]
            self.VRU_start_lane_id = self.test_info["VRU"]["start lane id"]

            # set traffic light information
            self.traffic_signal_num = 0
            self.traffic_signal_scenarios = []
            self.traffic_signal_node_id = []
            self.traffic_signal_index = []
            if "traffic signal" in list(self.test_info.keys()):
                self.traffic_signal_num = self.test_info["traffic signal"]["num"]
                self.traffic_signal_scenarios = self.test_info["traffic signal"][
                    "scenarios"
                ]
                self.traffic_signal_node_id = self.test_info["traffic signal"][
                    "node id"
                ]
                self.traffic_signal_index = self.test_info["traffic signal"]["index"]

            # set test ground information
            self.lane_width = self.test_info["lane width"]

            self.colors = COLORS

            self.alphas = ALPHAS

        self.fontsize = FONTSIZE

        # set test data location
        self.save_path = self.root + "/output/" + test_name
        os.makedirs(self.save_path, exist_ok=True)
        self.feasibility_path = self.save_path + "/feasibility"
        os.makedirs(self.feasibility_path, exist_ok=True)
        self.risk_level_path = self.save_path + "/risk_level"
        os.makedirs(self.risk_level_path, exist_ok=True)
        self.test_case_path = self.save_path + "/case"
        os.makedirs(self.test_case_path, exist_ok=True)
        self.test_data_path = self.save_path + "/test_data" + "/test_round_" + str(test_round)
        os.makedirs(self.test_data_path, exist_ok=True)
        self.evaluation_path = self.save_path + "/evaluation" + "/test_round_" + str(test_round)
        os.makedirs(self.evaluation_path, exist_ok=True)
        self.plot_path = self.save_path + "/figure" + "/test_round_" + str(test_round)
        os.makedirs(self.plot_path, exist_ok=True)
        self.visualization_path = self.save_path + "/visualization" + "/test_round_" + str(test_round)
        os.makedirs(self.visualization_path, exist_ok=True)

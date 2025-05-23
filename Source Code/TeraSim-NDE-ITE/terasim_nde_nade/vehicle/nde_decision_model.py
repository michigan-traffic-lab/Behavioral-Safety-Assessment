from terasim.vehicle.decision_models.idm_model import IDMModel
import terasim.utils as utils
import numpy as np
import logging
from terasim.overlay import traci
import os
import json
import terasim_nde_nade.vehicle.nde_vehicle_utils as nde_utils
from terasim_nde_nade.vehicle.nde_vehicle_utils import (
    get_collision_type_and_prob,
    Command,
    NDECommand,
    TrajectoryPoint,
    is_car_following,
    get_location,
)
import attr
import random
import traci.constants as tc
from typing import List, Tuple
from addict import Dict


class NDEDecisionModel(IDMModel):
    def __init__(self, reroute=True, dynamically_change_vtype=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reroute = reroute
        self.dynamically_change_vtype = dynamically_change_vtype

    @staticmethod
    def change_vehicle_type_according_to_location(veh_id, vehicle_location):
        if "CAV" in veh_id:
            return
        if "highway" in vehicle_location:  # highway/freeway scenario
            traci.vehicle.setType(veh_id, "NDE_HIGHWAY")
        elif (
            "intersection" in vehicle_location or "roundabout" in vehicle_location
        ):  # urban scenario
            traci.vehicle.setType(veh_id, "NDE_URBAN")
        else:
            raise ValueError(f"location {vehicle_location} not supported")

    @staticmethod
    def reroute_vehicle_if_necessary(veh_id: str, current_lane_id: str) -> bool:
        bestlanes = traci.vehicle.getBestLanes(veh_id)
        # get the bestlane with current lane id
        for lane in bestlanes:
            if lane[0] == current_lane_id:
                if not lane[4]:  # the current lane does not allow continuing the route
                    next_links = traci.lane.getLinks(current_lane_id)
                    if next_links:
                        next_lane_id = next_links[0][0]
                        next_edge_id = traci.lane.getEdgeID(next_lane_id)
                        try:
                            traci.vehicle.changeTarget(veh_id, next_edge_id)
                            return True  # reroute the vehicle
                        except:
                            return False
        return False  # do not reroute the vehicle

    def derive_control_command_from_observation(self, obs_dict):
        # change the IDM and MOBIL parameters based on the location
        vehicle_location = get_location(
            obs_dict["ego"]["veh_id"], obs_dict["ego"]["lane_id"]
        )
        # for highway and urban scenarios, change the vehicle type
        if self.dynamically_change_vtype:
            self.change_vehicle_type_according_to_location(
                obs_dict["ego"]["veh_id"], vehicle_location
            )
            # if the vehicle needs to be rerouted (e.g., emergency lane change caused by negligence model), reroute it
        if self.reroute:
            self.reroute_vehicle_if_necessary(
                obs_dict["ego"]["veh_id"], obs_dict["ego"]["lane_id"]
            )
        # let the vehicle to be controlled by SUMO
        return NDECommand(command_type=Command.DEFAULT, prob=1), {
            "ndd_command_distribution": {
                "normal": NDECommand(command_type=Command.DEFAULT, prob=1)
            }
        }

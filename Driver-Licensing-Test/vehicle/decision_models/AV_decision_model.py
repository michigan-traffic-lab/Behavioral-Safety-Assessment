from terasim.vehicle.decision_models.idm_model import IDMModel

import numpy as np


class AVDecisionModel(IDMModel):
    """dummy decision model:

    This decision model will constantly move the vehicle to the given x, y coordinates

    """

    def __init__(
        self,
        MOBIL_lc_flag=True,
        stochastic_acc_flag=False,
        IDM_parameters=None,
        MOBIL_parameters=None,
    ):
        super(AVDecisionModel, self).__init__(
            MOBIL_lc_flag=MOBIL_lc_flag,
            stochastic_acc_flag=stochastic_acc_flag,
            IDM_parameters=IDM_parameters,
            MOBIL_parameters=MOBIL_parameters,
        )
        self.lanechange_flag = False

    def update_lanechange_flag(self, lanechange_flag):
        self.lanechange_flag = lanechange_flag

    def decision(self, observation):
        return None, None

    def decision_no_lc(self, observation):
        action = None
        mode = None
        ego_vehicle = observation["Ego"]
        front_vehicle = observation["Lead"]
        mode = "IDM"
        if not self.stochastic_acc_flag:
            tmp_acc = self.IDM_acceleration(
                ego_vehicle=ego_vehicle, front_vehicle=front_vehicle
            )
        else:
            tmp_acc = self.stochastic_IDM_acceleration(
                ego_vehicle=ego_vehicle, front_vehicle=front_vehicle
            )
        tmp_acc = np.clip(tmp_acc, self.acc_low, self.acc_high)
        action = {"lateral": "central", "longitudinal": tmp_acc}
        action["type"] = "lon_lat"
        return action, mode

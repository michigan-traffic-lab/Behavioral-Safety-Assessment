import numpy as np
from loguru import logger
import terasim.utils as utils
from terasim.overlay import traci
from terasim_nde_nade.envs.safetest_nade_with_av import SafeTestNADEWithAV


class SafeTestNADEWithAVCosim(SafeTestNADEWithAV):

    def on_start(self, ctx):
        super().on_start(ctx)

        self.NODE_17_time = 0.0
        self.NODE_18_time = 0.0

        self.start_wait_time = 0.0

    def get_IS_prob(
        self,
        veh_id,
        ndd_control_command_dicts,
        maneuver_challenge_dicts,
        veh_ctx_dicts,
    ):
        if not maneuver_challenge_dicts[veh_id].get("negligence"):
            raise ValueError("The vehicle is not in the negligence mode.")

        IS_magnitude = 3000

        predicted_collision_type = ndd_control_command_dicts[veh_id].negligence.info[
            "predicted_collision_type"
        ]
        if predicted_collision_type == "intersection_tfl":
            upper_bound = 0.1
        else:
            upper_bound = 0.01

        return np.clip(
            ndd_control_command_dicts[veh_id]["negligence"].prob * IS_magnitude,
            0,
            upper_bound,
        )

    def check_intersection(self):
        location = traci.vehicle.getPosition3D("CAV")
        x, y = location[0], location[1]

        if x >= 45.0 and x <= 65.0 and y >= 185.0 and y <= 211.0:
            self.NODE_17_time += 0.1
        else:
            self.NODE_17_time = 0.0

        if x >= 45.0 and x <= 62.0 and y >= 130.0 and y <= 145.0:
            self.NODE_18_time += 0.1
        else:
            self.NODE_18_time = 0.0

    def check_autoware_start(self):
        location = traci.vehicle.getPosition3D("CAV")
        x, y = location[0], location[1]

        if abs(x - 134.78) <= 0.5 and abs(y - 22.75) <= 0.5:
            self.start_wait_time += 0.1

    def should_continue_simulation(self):

        num_colliding_vehicles = self.simulator.get_colliding_vehicle_number()
        self._vehicle_in_env_distance("after")

        self.check_intersection()
        self.check_autoware_start()

        print(traci.vehicle.getRoadID("CAV"))

        if "CAV" not in traci.vehicle.getIDList():
            logger.info("CAV left the simulation, stop the simulation.")
            self.record.update(
                {
                    "veh_1_id": None,
                    "veh_2_id": None,
                    "warmup_time": self.warmup_time,
                    "run_time": self.run_time,
                    "finish_reason": "CAV_left",
                }
            )
            return False

        elif self.start_wait_time >= 60.0:
            logger.info("CAV fail to start, stop the simulation.")
            self.record.update(
                {
                    "veh_1_id": None,
                    "veh_2_id": None,
                    "warmup_time": self.warmup_time,
                    "run_time": self.run_time,
                    "finish_reason": "fail_to_start",
                }
            )
            return False

        elif utils.get_time() >= self.warmup_time + self.run_time:
            logger.info("Simulation timeout, stop the simulation.")
            self.record.update(
                {
                    "veh_1_id": None,
                    "veh_2_id": None,
                    "warmup_time": self.warmup_time,
                    "run_time": self.run_time,
                    "finish_reason": "timeout",
                }
            )
            return False

        if num_colliding_vehicles >= 2:
            finish_reason = "collision"

            colliding_vehicles = self.simulator.get_colliding_vehicles()
            veh_1_id = colliding_vehicles[0]
            veh_2_id = colliding_vehicles[1]
            self.record.update(
                {
                    "veh_1_id": veh_1_id,
                    "veh_1_obs": self.vehicle_list[veh_1_id].observation,
                    "veh_2_id": veh_2_id,
                    "veh_2_obs": self.vehicle_list[veh_2_id].observation,
                    "warmup_time": self.warmup_time,
                    "run_time": self.run_time,
                    "finish_reason": finish_reason,
                }
            )
            return False

        elif self.NODE_17_time >= 15.0:
            finish_reason = "stuck_17"

            logger.info("CAV stuck at NODE 17, stop the simulation.")
            self.record.update(
                {
                    "veh_1_id": None,
                    "veh_2_id": None,
                    "warmup_time": self.warmup_time,
                    "run_time": self.run_time,
                    "finish_reason": finish_reason,
                }
            )
            return False

        elif self.NODE_18_time >= 18.0:
            finish_reason = "stuck_18"

            logger.info("CAV stuck at NODE 18, stop the simulation.")
            self.record.update(
                {
                    "veh_1_id": None,
                    "veh_2_id": None,
                    "warmup_time": self.warmup_time,
                    "run_time": self.run_time,
                    "finish_reason": finish_reason,
                }
            )
            return False

        elif traci.vehicle.getRoadID("CAV") == "EG_29_1_1":
            finish_reason = "finished"

            logger.info("CAV reached the destination, stop the simulation.")
            self.record.update(
                {
                    "veh_1_id": None,
                    "veh_2_id": None,
                    "warmup_time": self.warmup_time,
                    "run_time": self.run_time,
                    "finish_reason": finish_reason,
                }
            )
            return False

        return True

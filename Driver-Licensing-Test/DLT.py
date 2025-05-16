from terasim.simulator import Simulator

from terasim.logger.infoextractor import InfoExtractor
from terasim.vehicle.factories.vehicle_factory import VehicleFactory
from terasim.vehicle.sensors.ego import EgoSensor
from terasim.vehicle.sensors.local import LocalSensor
from terasim.vehicle.controllers.high_efficiency_controller import (
    HighEfficiencyController,
)
from terasim.vehicle.vehicle import Vehicle
from terasim.vehicle.controllers.sumo_move_controller import SUMOMOVEController
from terasim.vehicle.decision_models.idm_model import IDMModel

from env.dlt_env import DLTEnv
from vehicle.sensors.challenger_sensor import ChallengerSensor
from vehicle.decision_models.BV_decision_model import BVDecisionModel
from vehicle.decision_models.AV_decision_model import AVDecisionModel
from vehicle.decision_models.VRU_decision_model import VRUDecisionModel
from settings import *
from env.case_generation.case_gen import CaseGeneration

from terasim_cosim.terasim_plugin.terasim_tls_plugin import TeraSimTLSPlugin
from terasim_cosim.terasim_plugin.terasim_cosim_plugin import TeraSimCoSimPlugin

import time
import argparse


class VehicleFactory(VehicleFactory):

    def create_vehicle(self, veh_id, simulator):
        """Generate a vehicle with the given vehicle id in the simulator, composed of a decision model, a controller, and a list of sensors, which should be defined or customized by the user.

        Args:
            veh_id (_type_): vehicle id
            simulator (_type_): simulator (sumo)

        Returns:
            Vehicle: the contructed vehicle object
        """
        if "CAV" in veh_id:
            sensor_list = [EgoSensor(), LocalSensor(obs_range=50.0)]
            decision_model = AVDecisionModel()
            control_params = {
                "v_high": 8.3,
                "v_low": 0,
                "acc_duration": 0.1,  # the acceleration duration will be 0.1 second
                "lc_duration": 1,  # the lane change duration will be 1 second
            }
            controller = SUMOMOVEController(simulator)
        elif "BV" in veh_id:
            sensor_list = [EgoSensor(), ChallengerSensor()]
            decision_model = BVDecisionModel()
            controller = SUMOMOVEController(simulator)
        elif "VRU" in veh_id:
            sensor_list = [EgoSensor(), ChallengerSensor()]
            decision_model = VRUDecisionModel()
            controller = SUMOMOVEController(simulator)
        else:
            sensor_list = [EgoSensor(), LocalSensor()]
            # decision_model = DummyDecisionModel(mode="random")  # mode="random" "constant"
            decision_model = IDMModel(MOBIL_lc_flag=False, stochastic_acc_flag=True)
            control_params = {
                "v_high": 40,
                "v_low": 20,
                "acc_duration": 0.1,  # the acceleration duration will be 0.1 second
                "lc_duration": 1,  # the lane change duration will be 1 second
            }
            controller = HighEfficiencyController(simulator, control_params)
        return Vehicle(
            veh_id,
            simulator,
            sensors=sensor_list,
            decision_model=decision_model,
            controller=controller,
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Driver licensing test simulator.")
    argparser.add_argument("--test-name", type=str, default="Autoware.Universe")
    argparser.add_argument("--scenario-folder", type=str, default="cut_in")
    argparser.add_argument(
        "--realtime-flag",
        type=bool,
        default=True,
        help="Flag to indicate whether to run SUMO in real time",
    )
    argparser.add_argument(
        "--round-num",
        type=int,
        default=1,
        help="The test round number, starting from 1",
    )
    argparser.add_argument(
        "--case-num",
        type=int,
        default=-1,
        help="The test case number to run, -1 for all cases",
    )
    argparser.add_argument(
        "--gui", 
        action="store_true", 
        help="run simulation with sumo gui"
    )

    args = argparser.parse_args()
    test_name = args.test_name
    scenario_folder = args.scenario_folder
    gui = args.gui
    realtime_flag = args.realtime_flag
    round_num = args.round_num
    case_num = args.case_num

    settings = Settings(test_name, round_num, scenario_folder)

    # generate detailed risk levels and test cases
    case_generator = CaseGeneration()
    case_generator.init(settings)

    for i in range(case_generator.total_case_num):
        if case_generator.total_case_num > case_num >= 0:
            i = case_num
        elif case_num >= case_generator.total_case_num:
            print('Invalid case number, please check the case number.')
            break
        init_conditions = case_generator.choose_cases(i)
        env = DLTEnv(
            vehicle_factory=VehicleFactory(),
            info_extractor=InfoExtractor,
            init_conditions=init_conditions,
            case_num=i,
            settings=settings,
        )
        output_path = settings.visualization_path + '/' + scenario_folder + "/" + str(i)
        os.makedirs(output_path, exist_ok=True)
        sim = Simulator(
            sumo_net_file_path=settings.sumo_map_file,
            sumo_config_file_path=settings.sumo_config_file,
            num_tries=10,
            gui_flag=gui,
            output_path=output_path,
            realtime_flag=realtime_flag,
            sumo_output_file_types=["fcd_all"],
            additional_sumo_args=["--start", "--quit-on-end"],
        )
        # Vehicle Co-simulation
        sim.add_plugin(
            TeraSimCoSimPlugin(
                remote_flag=False,  # connect to mcityos, disable for local testing
                control_cav=False,  # allow outside source to synchronize av state, like CARLA
                keepRoute=2,  # Traci keep route defition, 2 being the most flexible
                CAVSpeedOverride=True,  # Allow speed override for CAV in Terasim
                pub_channels=[],  # Publish channels (mcityos remote only)
                sub_channels=[],  # Subscribe channels (mcityos remote only)
                latency_src_channels=[],  # Latency source channels (mcityos remote only)
            )
        )

        # Traffic Light Co-simulation
        sim.add_plugin(
            TeraSimTLSPlugin(
                remote_flag=False,  # connect to mcityos, disable for local testing
                control_tls=True,  # TeraSim controlled traffic lights
                pub_channels=[],  # Publish channels (mcityos remote only)
                sub_channels=[],  # Subscribe channels (mcityos remote only)
                latency_src_channels=[],  # Latency source channels (mcityos remote only)
            )
        )
        sim.bind_env(env)
        sim.run()
        time.sleep(1)
        if case_num >= 0:
            break

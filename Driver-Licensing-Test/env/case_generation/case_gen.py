from termcolor import colored
import time


class CaseGeneration:
    def __init__(self):
        self.case_init_conditions = {}

        self.total_case_num = 0

        self.test_name = None
        self.scenario_list = []

    def init(self, settings):
        self.test_name = settings.test_name
        self.scenario_list = settings.scenario_list

        total_case_num = []
        for scenario in self.scenario_list:
            if scenario == 'Cut-In':
                from env.case_generation.scenario.cut_in import CutInGen
                self.lane_change_gen = CutInGen(settings)
                self.lane_change_gen.init()
                total_case_num.append(sum([len(self.lane_change_gen.cases[risk_level])  for risk_level in self.lane_change_gen.risk_levels]))
            elif scenario == 'Car Following':
                from env.case_generation.scenario.car_following import CarFollowingGen
                self.car_following_gen = CarFollowingGen(settings)
                self.car_following_gen.init()
                total_case_num.append(sum([len(self.car_following_gen.cases[risk_level])  for risk_level in self.car_following_gen.risk_levels]))
            elif scenario == 'Right Turn (AV goes straight)':
                from env.case_generation.scenario.right_turn_straight import RightTurnStraightGen
                self.right_turn_gen = RightTurnStraightGen(settings)
                self.right_turn_gen.init()
                total_case_num.append(sum([len(self.right_turn_gen.cases[risk_level])  for risk_level in self.right_turn_gen.risk_levels]))
            elif scenario == 'Right Turn (AV turns right)':
                from env.case_generation.scenario.right_turn_turn import RightTurnTurnGen
                self.right_turn_turn_gen = RightTurnTurnGen(settings)
                self.right_turn_turn_gen.init()
                total_case_num.append(sum([len(self.right_turn_turn_gen.cases[risk_level])  for risk_level in self.right_turn_turn_gen.risk_levels]))
            elif scenario == 'Left Turn (AV goes straight)':
                from env.case_generation.scenario.left_turn_straight import LeftTurnStraightGen
                self.left_turn_gen = LeftTurnStraightGen(settings)
                self.left_turn_gen.init()
                total_case_num.append(sum([len(self.left_turn_gen.cases[risk_level])  for risk_level in self.left_turn_gen.risk_levels[1:4]]))
            elif scenario == 'Left Turn (AV turns left)':
                from env.case_generation.scenario.left_turn_turn import LeftTurnTurnGen
                self.left_turn_turn_gen = LeftTurnTurnGen(settings)
                self.left_turn_turn_gen.init()
                total_case_num.append(sum([len(self.left_turn_turn_gen.cases[risk_level])  for risk_level in self.left_turn_turn_gen.risk_levels[1:4]]))
            elif scenario == 'VRU Crossing the Street at the Crosswalk':
                from env.case_generation.scenario.vru_at_crosswalk import VRUAtCrosswalkGen
                self.vru_at_crosswalk = VRUAtCrosswalkGen(settings)
                self.vru_at_crosswalk.init()
                total_case_num.append(sum([len(self.vru_at_crosswalk.cases[risk_level])  for risk_level in self.vru_at_crosswalk.risk_levels]))
            elif scenario == 'VRU Crossing the Street without Crosswalk':
                from env.case_generation.scenario.vru_without_crosswalk import VRUWithoutCrosswalkGen
                self.jwlk_gen = VRUWithoutCrosswalkGen(settings)
                self.jwlk_gen.init()
                total_case_num.append(sum([len(self.jwlk_gen.cases[risk_level])  for risk_level in self.jwlk_gen.risk_levels]))
            elif scenario == 'Lane Departure (same direction)':
                from env.case_generation.scenario.lane_departure_same import LaneDepartureSameGen
                self.lane_departure_same_gen = LaneDepartureSameGen(settings)
                self.lane_departure_same_gen.init()
                total_case_num.append(sum([len(self.lane_departure_same_gen.cases[risk_level])  for risk_level in self.lane_departure_same_gen.risk_levels]))
            elif scenario == 'Lane Departure (opposite direction)':
                from env.case_generation.scenario.lane_departure_opposite import LaneDepartureOppositeGen
                self.lane_departure_opposite_gen = LaneDepartureOppositeGen(settings)
                self.lane_departure_opposite_gen.init()
                total_case_num.append(sum([len(self.lane_departure_opposite_gen.cases[risk_level])  for risk_level in self.lane_departure_opposite_gen.risk_levels]))
            elif scenario == 'BV Merging into the Roundabout':
                from env.case_generation.scenario.roundabout import RoundaboutGen
                self.roundabout_inside_gen = RoundaboutGen(scenario, settings)
                self.roundabout_inside_gen.init()
                total_case_num.append(sum([len(self.roundabout_inside_gen.cases[risk_level])  for risk_level in self.roundabout_inside_gen.risk_levels]))
            elif scenario == 'AV Merging into the Roundabout':
                from env.case_generation.scenario.roundabout import RoundaboutGen
                self.roundabout_av_outside_gen = RoundaboutGen(scenario, settings)
                self.roundabout_av_outside_gen.init()
                total_case_num.append(sum([len(self.roundabout_av_outside_gen.cases[risk_level])  for risk_level in self.roundabout_av_outside_gen.risk_levels]))
            elif scenario == 'Vehicle Encroachment':
                from env.case_generation.scenario.vehicle_encroachment import VehicleEncroachmentGen
                self.vehicle_encroachment_gen = VehicleEncroachmentGen(settings)
                self.vehicle_encroachment_gen.init()
                total_case_num.append(sum([len(self.vehicle_encroachment_gen.cases[risk_level])  for risk_level in self.vehicle_encroachment_gen.risk_levels]))
            elif scenario == 'Traffic Signal':
                from env.case_generation.scenario.traffic_signal import TrafficSignalGen
                self.traffic_signal_gen = TrafficSignalGen(settings)
                self.traffic_signal_gen.init()
                total_case_num.append(sum([len(self.traffic_signal_gen.cases[risk_level])  for risk_level in self.traffic_signal_gen.risk_levels]))
            else:
                total_case_num.append(0)
        self.total_case_num = max(total_case_num)

    def choose_cases(self, case_num=-1):
        print('--------------------------------------------------------------------')
        print('[', case_num, '/', self.total_case_num, '] Cases settings are as follows:')

        original_init_conditions = {}
        for scenario in self.scenario_list:
            if scenario == 'Cut-In':
                risk_level, selected_case, original_case = self.lane_change_gen.choose_case(case_num)
            elif scenario == 'Car Following':
                risk_level, selected_case, original_case = self.car_following_gen.choose_case(case_num)
            elif scenario == 'Right Turn (AV goes straight)':
                risk_level, selected_case, original_case = self.right_turn_gen.choose_case(case_num)
            elif scenario == 'Right Turn (AV turns right)':
                risk_level, selected_case, original_case = self.right_turn_turn_gen.choose_case(case_num)
            elif scenario == 'Left Turn (AV goes straight)':
                risk_level, selected_case, original_case = self.left_turn_gen.choose_case(case_num)
            elif scenario == 'Left Turn (AV turns left)':
                risk_level, selected_case, original_case = self.left_turn_turn_gen.choose_case(case_num)
            elif scenario == 'VRU Crossing the Street at the Crosswalk':
                risk_level, selected_case, original_case = self.vru_at_crosswalk.choose_case(case_num)
            elif scenario == 'VRU Crossing the Street without Crosswalk':
                risk_level, selected_case, original_case = self.jwlk_gen.choose_case(case_num)
            elif scenario == 'Lane Departure (same direction)':
                risk_level, selected_case, original_case = self.lane_departure_same_gen.choose_case(case_num)
            elif scenario == 'Lane Departure (opposite direction)':
                risk_level, selected_case, original_case = self.lane_departure_opposite_gen.choose_case(case_num)
            elif scenario == 'BV Merging into the Roundabout':
                risk_level, selected_case, original_case = self.roundabout_inside_gen.choose_case(case_num)
            elif scenario == 'AV Merging into the Roundabout':
                risk_level, selected_case, original_case = self.roundabout_av_outside_gen.choose_case(case_num)
            elif scenario == 'Vehicle Encroachment':
                risk_level, selected_case, original_case = self.vehicle_encroachment_gen.choose_case(case_num)
            elif scenario == 'Traffic Signal':
                risk_level, selected_case, original_case = self.traffic_signal_gen.choose_case(case_num)
            else:
                risk_level = "None"
                selected_case = {}
                original_case = {}
                print(colored(scenario + ' is not ready yet.', 'red'))
            self.case_init_conditions[scenario] = selected_case
            original_init_conditions[scenario] = original_case
            print(scenario, risk_level, selected_case)

        case_settings = {}
        case_settings['timestamp'] = time.time()
        case_settings['case_num'] = case_num
        case_settings['scenario_list'] = self.scenario_list
        case_settings['risk_level'] = risk_level
        case_settings['case_init_conditions'] = self.case_init_conditions
        case_settings['original_case_init_conditions'] = original_init_conditions
        return case_settings

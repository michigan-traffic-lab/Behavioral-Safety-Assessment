from terasim.agent.agent_decision_model import AgentDecisionModel

from terasim.overlay import traci

from settings import *
from utils import dis_from_ind_to_ind, cal_dis, str_replacer, find_next_route_points


class TrafficSignalDecisionModel(AgentDecisionModel):
    """dummy decision model:

        This decision model will constantly move the vehicle to the given x, y coordinates

    """
    def __init__(self):
        super().__init__()
    
    def init(self, id, init_conditions, settings):
        self.scenario_list = settings.traffic_signal_scenarios[id-1]
        self.AV_route = settings.AV_route
        self.init_conditions = init_conditions
        self.node_id = settings.traffic_signal_node_id[id-1]
        self.index = settings.traffic_signal_index[id-1]
        self.traffic_signal_ind = {
            scenario: settings.scenario_info[scenario]['AV']['traffic light ind'] for scenario in self.scenario_list
        }
        self.time_step = 0.1

        self.start_flags = {scenario: False for scenario in self.scenario_list}
        self.yellow_flags = {scenario: False for scenario in self.scenario_list}
        self.red_flags = {scenario: False for scenario in self.scenario_list}

        self.dis_err = {scenario: 0 for scenario in self.scenario_list}
        self.sp_err = {scenario: 0 for scenario in self.scenario_list}
        self.init_AV_sp = {scenario: 0 for scenario in self.scenario_list}
        self.init_timestamp = {scenario: 0 for scenario in self.scenario_list}
        self.AV_hist = {scenario: [] for scenario in self.scenario_list}
    
    def update(self, AV_state):
        self.AV_state = AV_state

    def derive_control_command_from_observation(self, obs_dict):
        """derive control command from observation

        Args:
            obs_dict (dict): vehicle observation dictionary

        Returns:
            dict: command
        """
        
        self._manage_state(self.AV_state)
        
        self._set_traffic_signal(self.AV_state)
        return None, None

    def _set_traffic_signal(self, AV_state):
        for scenario in self.scenario_list:
            if AV_state.scenario == scenario:
                traffic_signal_state_len = len(traci.trafficlight.getRedYellowGreenState(self.node_id))
                traffic_signal_state = ''
                for _ in range(traffic_signal_state_len):
                    traffic_signal_state += 'r'
                if self.start_flags[scenario] and not self.yellow_flags[scenario]:
                    for i in self.index:
                        traffic_signal_state = str_replacer(traffic_signal_state, 'g', i)
                    traci.trafficlight.setRedYellowGreenState(self.node_id, traffic_signal_state)
                elif self.start_flags[scenario] and self.yellow_flags[scenario] and not self.red_flags[scenario]:
                    for i in self.index:
                        traffic_signal_state = str_replacer(traffic_signal_state, 'y', i)
                    traci.trafficlight.setRedYellowGreenState(self.node_id, traffic_signal_state)
                elif self.start_flags[scenario] and self.yellow_flags[scenario] and self.red_flags[scenario]:
                    for i in self.index:
                        traffic_signal_state = str_replacer(traffic_signal_state, 'r', i)
                    traci.trafficlight.setRedYellowGreenState(self.node_id, traffic_signal_state)

    def _manage_state(self, AV_state):
        for scenario in self.scenario_list:
            if AV_state.scenario == scenario and not self.start_flags[scenario]:
                self.start_flags[scenario] = True
            elif AV_state.scenario == scenario and self.start_flags[scenario] and not self.yellow_flags[scenario]:
                NNPN = find_next_route_points(self.AV_route, [AV_state.x, AV_state.y], AV_state.NPN)
                AV_dis = cal_dis([AV_state.x, AV_state.y], self.AV_route[NNPN][:2]) + dis_from_ind_to_ind(self.AV_route, NNPN, self.traffic_signal_ind[AV_state.scenario])
                if AV_dis <= self.init_conditions[scenario]['thw'] * AV_state.speed:
                    self.yellow_flags[scenario] = True
                    self.dis_err[scenario] = AV_dis - self.init_conditions[scenario]['thw'] * AV_state.speed
                    self.init_AV_sp[scenario] = AV_state.speed
                    self.init_timestamp[scenario] = AV_state.timestamp
            elif AV_state.scenario == scenario and self.start_flags[scenario] and self.yellow_flags[scenario] and not self.red_flags[scenario]:
                if AV_state.timestamp - self.init_timestamp[scenario] >= 3:
                    self.red_flags[scenario] = True

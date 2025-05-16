import copy
from termcolor import colored
from terasim.overlay import traci

from settings import *
from utils import *


class AgentState():
    timestamp = .0
    x = .0
    y = .0
    speed = .0
    lon_speed = .0
    lat_speed = .0
    acceleration = .0
    lon_acceleration = .0
    lat_acceleration = .0
    heading = .0
    in_route = False
    scenario = None
    NPN = -1
    start = False
    complete = False


class ScenarioState():
    start = False
    on_going = False
    end = False
    complete = False


class TrafficSignalState():
    timestamp = .0
    state = None
    scenario = None
    node_id = ''
    index = []


class StateManager():
    def __init__(self, settings):
        self.scenario_list = settings.scenario_list
        self.scenario_states = {scenario: ScenarioState() for scenario in settings.scenario_list}
        self.scenario_states['start'] = True
        self.AV_state = AgentState()
        self.AV_scenario_list = settings.scenario_list
        self.AV_route = settings.AV_route
        self.AV_state_hist = [AgentState() for _ in range(3)]

        self.BV_num = settings.BV_num
        self.BV_states = [AgentState() for _ in range(self.BV_num)]
        self.BV_scenario_lists = settings.BV_scenarios
        self.BV_routes = copy.deepcopy(settings.BV_routes)
        self.BV_states_hist = [[AgentState() for _ in range(3)] for _ in range(self.BV_num)]

        self.VRU_num = settings.VRU_num
        self.VRU_states = [AgentState() for _ in range(self.VRU_num)]
        self.VRU_scenario_lists = settings.VRU_scenarios
        self.VRU_states_hist = [[AgentState() for _ in range(3)] for _ in range(self.VRU_num)]

        self.traffic_signal_num = settings.traffic_signal_num
        self.traffic_signal_states = [TrafficSignalState() for _ in range(self.traffic_signal_num)]
        self.traffic_signal_states_hist = [[TrafficSignalState() for _ in range(3)] for _ in range(self.traffic_signal_num)]
        self.traffic_signal_scenario_lists = settings.traffic_signal_scenarios
        for scenario, traffic_signal_state, node_id, ind in zip(self.traffic_signal_scenario_lists, self.traffic_signal_states, settings.traffic_signal_node_id, settings.traffic_signal_index):
            traffic_signal_state.scenario = scenario
            traffic_signal_state.node_id = node_id
            traffic_signal_state.index = ind

        self.scenario_info = settings.scenario_info
        self.lane_width = settings.lane_width
        self.time_step = settings.time_step
        self.AV_length = settings.AV_length
        self.BV_length = settings.BV_length
        self.VRU_length = settings.VRU_length
        self.scenario_step = {scenario: 0 for scenario in settings.scenario_list}

    def update_agent_routes(self, i, BV_route):
        if self.AV_state.scenario == self.BV_states[i].scenario and self.AV_state.scenario is not None:
            self.BV_routes[i] = copy.deepcopy(BV_route)

    def update_agent_state(self, current_time, all_vehicle_observation: dict):
        for veh_id, state in all_vehicle_observation.items():
            if state['position'] is not None:
                if 'CAV' in veh_id:
                    self.AV_state_hist.pop(0)
                    self._update_agent_state(self.AV_state, current_time, state['position'][0], state['position'][1], state['velocity'], state['acceleration'], state['heading'], 'AV', self.AV_scenario_list, self.AV_route)
                    self.AV_state_hist.append(copy.deepcopy(self.AV_state))
                    self._update_agent_previous_state(self.AV_state_hist, self.AV_length)
                    if (self.AV_state.NPN > self.scenario_info[self.AV_scenario_list[-1]]['AV']['end ind'] or self.scenario_states[self.AV_scenario_list[-1]].complete) and not self.AV_state.complete:
                        self.AV_state.complete = True
                elif 'BV' in veh_id:
                    for i in range(self.BV_num):
                        if str(i+1) in veh_id:
                            self.BV_states_hist[i].pop(0)
                            self._update_agent_state(self.BV_states[i], current_time, state['position'][0], state['position'][1], state['velocity'], state['acceleration'], state['heading'], 'challenger', self.BV_scenario_lists[i], self.BV_routes[i])
                            self.BV_states_hist[i].append(copy.deepcopy(self.BV_states[i]))
                            self._update_agent_previous_state(self.BV_states_hist[i], self.BV_length[i])
                elif 'VRU' in veh_id:
                    for i in range(self.VRU_num):
                        if str(i+1) in veh_id:
                            self.VRU_states_hist[i].pop(0)
                            self._update_agent_state(self.VRU_states[i], current_time, state['position'][0], state['position'][1], state['velocity'], state['acceleration'], state['heading'], 'challenger', self.VRU_scenario_lists[i])
                            self.VRU_states_hist[i].append(copy.deepcopy(self.VRU_states[i]))
                            self._update_agent_previous_state(self.VRU_states_hist[i], self.VRU_length[i])

        for i in range(self.traffic_signal_num):
            self.traffic_signal_states[i].timestamp = traci.simulation.getTime()
            self.traffic_signal_states[i].state = traci.trafficlight.getRedYellowGreenState(self.traffic_signal_states[i].node_id)
        self._update_traffic_signal_previous_state()

    def _update_traffic_signal_previous_state(self):
        for i in range(self.traffic_signal_num):
            self.traffic_signal_states_hist[i].append(copy.deepcopy(self.traffic_signal_states[i]))
            self.traffic_signal_states_hist[i].pop(0)
    
    def _update_agent_state(self, agent_state, timestamp, x, y, speed, acceleration, heading, agent_category, scenario_list, route=None):
        agent_state.timestamp = timestamp
        agent_state.x = x
        agent_state.y = y
        agent_state.speed = speed
        agent_state.acceleration = acceleration
        agent_state.heading = heading
        if route is not None:
            if agent_category == "CAV":
                agent_state.NPN, agent_state.in_route = check_in_route(route, [agent_state.x, agent_state.y], agent_state.NPN, self.lane_width * 1.5)
            else:
                agent_state.NPN, agent_state.in_route = check_in_route(route, [agent_state.x, agent_state.y], agent_state.NPN, self.lane_width * 2)
            flag = True
            for scenario in scenario_list:
                if check_in_scenario(self.scenario_info[scenario][agent_category]['start ind'], self.scenario_info[scenario][agent_category]['end ind'], agent_state.in_route, agent_state.NPN):
                    agent_state.scenario = scenario
                    flag = False
            if flag:
                agent_state.scenario = None
        else:
            for scenario in scenario_list:
                agent_state.NPN = -1
                agent_state.scenario = scenario
                agent_state.in_route = True

    def _update_agent_previous_state(self, agent_state_hist, agent_len):
        agent_state_hist[-1].x -= agent_len / 2 * math.sin(agent_state_hist[-1].heading / 180 * math.pi)
        agent_state_hist[-1].y -= agent_len / 2 * math.cos(agent_state_hist[-1].heading / 180 * math.pi)
        if agent_state_hist[1].timestamp > 0:
            rotated_position = rotate_point2_around_point1([agent_state_hist[1].x, agent_state_hist[1].y], [agent_state_hist[2].x, agent_state_hist[2].y], (90 - agent_state_hist[1].heading) / 180 * math.pi)
            lon_speed = (rotated_position[0] - agent_state_hist[1].x) / self.time_step
            lat_speed = (rotated_position[1] - agent_state_hist[1].y) / self.time_step
            merged_speed = math.sqrt(lon_speed ** 2 + lat_speed ** 2)
            if agent_state_hist[1].speed != 0 and merged_speed != 0:
                agent_state_hist[1].lon_speed = lon_speed / merged_speed * agent_state_hist[1].speed
                agent_state_hist[1].lat_speed = lat_speed / merged_speed * agent_state_hist[1].speed
            else:
                agent_state_hist[1].lon_speed = 0
                agent_state_hist[1].lat_speed = 0
        if agent_state_hist[0].timestamp > 0:
            agent_state_hist[0].lon_acceleration = (agent_state_hist[1].lon_speed - agent_state_hist[0].lon_speed) / self.time_step
            agent_state_hist[0].lat_acceleration = (agent_state_hist[1].lat_speed - agent_state_hist[0].lat_speed) / self.time_step

    def update_scenario_state(self):
        for scenario in self.AV_scenario_list:
            if not self.scenario_states[scenario].complete:
                if self.scenario_states[scenario].start:
                    self.scenario_step[scenario] += 1
                if self.scenario_info[scenario]['AV']['start ind'] <= self.AV_state.NPN and not self.scenario_states[scenario].start:  # <= self.scenario_info[scenario]['AV']['start ind'] + 2
                    self.scenario_states[scenario].start = True
                    print(colored(scenario + ' is started...', 'green'))
                    if self.scenario_states['start']:
                        self.scenario_states['start'] = False
                if self.AV_state.scenario == scenario and self.scenario_states[scenario].start and not self.scenario_states[scenario].on_going:
                    self.scenario_states[scenario].on_going = True
                if self.scenario_info[scenario]['AV']['end ind'] - 2 <= self.AV_state.NPN <= self.scenario_info[scenario]['AV']['end ind'] and self.scenario_states[scenario].on_going and not self.scenario_states[scenario].end:
                    self.scenario_states[scenario].end = True
                if self.scenario_states[scenario].start and self.scenario_states[scenario].on_going and self.scenario_states[scenario].end:
                    self.scenario_states[scenario].start = False
                    self.scenario_states[scenario].on_going = False
                    self.scenario_states[scenario].end = False
                    self.scenario_states[scenario].complete = True
                    print(colored(scenario + ' is completed!', 'green'))
                ind = self.AV_scenario_list.index(scenario)
                if self.AV_state.scenario == scenario and self.scenario_states[scenario].start and self.scenario_states[scenario].on_going:
                    if cal_dis([self.AV_state_hist[0].x, self.AV_state_hist[0].y], [self.AV_state_hist[-1].x, self.AV_state_hist[-1].y]) < 0.1 and self.scenario_step[scenario] > 200 and self.AV_state.speed < 0.1:
                        if self.BV_num > 0:
                            BV_state_hist = self.BV_states_hist[ind]
                            if cal_dis([BV_state_hist[0].x, BV_state_hist[0].y], [BV_state_hist[-1].x, BV_state_hist[-1].y]) < 0.1 and self.BV_states[ind].speed < 0.1:
                                self.scenario_states[scenario].complete = True
                                print(colored(scenario + ' is completed!', 'green'))
                        else:
                            self.scenario_states[scenario].complete = True
                            print(colored(scenario + ' is completed!', 'green'))

    def reset_scenario_states(self):
        self.scenario_states = {scenario: ScenarioState() for scenario in self.scenario_list}

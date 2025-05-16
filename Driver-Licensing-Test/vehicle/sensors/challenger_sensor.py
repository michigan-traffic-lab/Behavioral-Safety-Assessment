from terasim.agent.agent_sensor import AgentSensor

from terasim.overlay import traci


class ChallengerSensor(AgentSensor):
    """A sensor for reporting basic states (position, speed, heading, etc.)"""

    DEFAULT_PARAMS = dict(
        fields={
            "velocity": traci.vehicle.getSpeed,
            "position": traci.vehicle.getPosition,
            "position3d": traci.vehicle.getPosition3D,
            "heading": traci.vehicle.getAngle,
            "edge_id": traci.vehicle.getRoadID,
            "lane_id": traci.vehicle.getLaneID,
            "lane_index": traci.vehicle.getLaneIndex,
            "acceleration": traci.vehicle.getAcceleration,
        }
    )

    def __init__(self, name="bv_cav_sensor", **params):
        super().__init__(name, **params)

    def fetch(self) -> dict:
        veh_id = "CAV"
        data = {"data": {}}
        for field, getter in self.params.fields.items():
            data[field] = getter(veh_id)
        return data

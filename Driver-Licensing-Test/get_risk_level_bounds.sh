#!/bin/bash

# get risk level bounds of the cut-in and lane departure (same direction) scenarios
python get_lane_departure_same_bounds.py

# get risk level bounds of lane departure (opposite direction) scenario
python get_lane_departure_opposite_bounds.py

# get risk level bounds of car-following scenario
python get_car_following_bounds.py

# get risk level bounds of left-turn (AV goes straight) and left-turn (AV turns left) scenarios
python get_left_turn_bounds.py

# get risk level bounds of right-turn (AV goes straight) and right-turn (AV turns right) scenarios
python get_right_turn_bounds.py

# get risk level bounds of VRU crossing the street at the crosswalk and VRU crossing the street without the crosswalk scenarios
python get_VRU_bounds.py

# get risk level bounds of AV merging into the roundabout and BV merging into the roundabout scenarios
python get_roundabout_bounds.py

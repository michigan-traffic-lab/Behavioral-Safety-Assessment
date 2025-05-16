#!/bin/bash

# generate folder list of Argoverse2 dataset to save time
python env/data_process/generate_folder_list.py --dataset-path path/to/argoverse2/dataset -- save-path output/data_process

# extract parameters in cut-in, lane departure (same direction), and lane departure (opposite direction) scenarios
python env/data_process/analyze_lane_departure.py --dataset-path path/to/argoverse2/dataset -- save-path output/data_process

# extract parameters in car-following scenario
python env/data_process/analyze_car_following.py --dataset-path path/to/argoverse2/dataset -- save-path output/data_process

# extract parameters in left-turn (AV goes straight) and left-turn (AV turns left) scenarios
python env/data_process/analyze_left_turn.py --dataset-path path/to/argoverse2/dataset -- save-path output/data_process

# extract parameters in right-turn (AV goes straight) and right-turn (AV turns right) scenarios
python env/data_process/analyze_right_turn.py --dataset-path path/to/argoverse2/dataset -- save-path output/data_process

# extract parameters in VRU crossing the street at the crosswalk and VRU crossing the street without the crosswalk scenarios
python env/data_process/analyze_pedestrian.py --dataset-path path/to/argoverse2/dataset -- save-path output/data_process

# extract parameters in AV merging into the roundabout and BV merging into the roundabout scenarios
python env/data_process/analyze_roundabout.py --dataset-path path/to/processed/rounD/dataset -- save-path output/data_process

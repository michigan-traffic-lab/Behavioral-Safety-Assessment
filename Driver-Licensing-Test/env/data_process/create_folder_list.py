import sys
import os

# Get the directory of the current script
dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(dir)

import argparse


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--dataset-path', type=str, default='/path/to/Argoverse2/dataset')  # 
    argparse.add_argument('--save-path', type=str, default='output/data_process')
    args = argparse.parse_args()

    train_path = args.dataset_path + '/train'

    data = []
    same_direction_num = 0
    opposite_direction_num = 0
    count = 0
    with os.scandir(train_path) as entries:
        for entry in entries:
            with open('env/data_process/folder_list.txt', 'a') as f:
                f.write(entry.name + '\n')
                count += 1
    print(f'Total folders: {count}')

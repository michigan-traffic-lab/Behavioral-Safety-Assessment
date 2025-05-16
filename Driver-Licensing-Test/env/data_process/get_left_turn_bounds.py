import sys
import os

# Get the directory of the current script
dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(dir)

import pandas as pd
from utils import *
from settings import *

colors = list(COLORS.values())
alphas = list(ALPHAS.values())
fontsize = FONTSIZE


if __name__ == '__main__':
    folder = 'output/data_process/left_turn'

    # obtain and plot risk level bounds
    data_file = folder + '/data.csv'
    data = pd.read_csv(data_file)

    left_turn_straight_acc = []
    left_turn_straight_dec = []
    left_turn_turn_acc = []
    left_turn_turn_dec = []
    for d1, v1, a1, d2, v2, a2, flag in data.values:
        if min_acc < a1 < max_acc:
            left_turn_straight_acc.append(a1)
        elif max_dec < a1 < min_dec:
            left_turn_straight_dec.append(a1)
        if min_acc < a2 < max_acc:
            left_turn_turn_acc.append(a2)
        elif max_dec < a2 < min_dec:
            left_turn_turn_dec.append(a2)
    left_turn_straight_acc.sort(reverse=True)
    left_turn_straight_dec.sort()
    left_turn_turn_acc.sort(reverse=True)
    left_turn_turn_dec.sort()

    straight_acc_bounds = plot_acc_distribution(left_turn_straight_acc, max_acc, fontsize, colors, alphas, folder, 'straight_acc_distribution')
    straight_dec_bounds = plot_acc_distribution(left_turn_straight_dec, max_dec, fontsize, colors, alphas, folder, 'straight_dec_distribution', reverse=True)
    turn_acc_bounds = plot_acc_distribution(left_turn_turn_acc, max_acc, fontsize, colors, alphas, folder, 'turn_acc_distribution')
    turn_dec_bounds = plot_acc_distribution(left_turn_turn_dec, max_dec, fontsize, colors, alphas, folder, 'turn_dec_distribution', reverse=True)

    with open(folder + '/bounds.txt', 'w') as f:
        f.write(f'straight_acc_bounds: {straight_acc_bounds}\n')
        f.write(f'straight_dec_bounds: {straight_dec_bounds}\n')
        f.write(f'turn_acc_bounds: {turn_acc_bounds}\n')
        f.write(f'turn_dec_bounds: {turn_dec_bounds}\n')\

    # plot the initial conditions
    data_file = folder + '/init_conditions.csv'
    key, data = read_csv(data_file)

    left_turn_straight_plot_data = []
    left_turn_turn_plot_data = []
    for d1, v1, a1, d2, v2, a2, flag in data:
        left_turn_straight_plot_data.append([v1, d1 + d2])
        left_turn_turn_plot_data.append([v2, d1 + d2])

    plot_init_condition_distribution(
        left_turn_straight_plot_data,
        [0, 20],
        [0, 50],
        1,
        1,
        fontsize,
        r'BV speed ($m/s$)',
        r'Distance ($m$)',
        folder,
        'straight_init_condition_distribution'
    )

    plot_init_condition_distribution(
        left_turn_turn_plot_data,
        [0, 20],
        [0, 50],
        1,
        1,
        fontsize,
        r'BV speed ($m/s$)',
        r'Distance ($m$)',
        folder,
        'turn_init_condition_distribution'
    )

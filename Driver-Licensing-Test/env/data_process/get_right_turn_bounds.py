import sys
import os

# Get the directory of the current script
dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(dir)

from utils import *
from settings import *

colors = list(COLORS.values())
alphas = list(ALPHAS.values())
fontsize = FONTSIZE


if __name__ == '__main__':
    folder = 'output/data_process/right_turn'

    # obtain and plot risk level bounds
    data_file = folder + '/data.csv'
    key, data = read_csv(data_file)

    AV_straight_acc = []
    AV_straight_dec = []
    AV_turn_acc = []
    AV_turn_dec = []
    for d1, v1, a1, d2, v2, a2, flag in data:
        if flag == 'straight':
            if a2 < min_dec and a2 > max_dec:
                AV_turn_dec.append(a2)
            if a1 > min_acc and a1 < max_acc:
                AV_straight_acc.append(a1)
        if flag == 'turn':
            if a1 < min_dec and a1 > max_dec:
                AV_straight_dec.append(a1)
            if a2 > min_acc and a2 < max_acc:
                AV_turn_acc.append(a2)
    AV_straight_acc.sort(reverse=True)
    AV_straight_dec.sort()
    AV_turn_acc.sort(reverse=True)
    AV_turn_dec.sort()

    straight_acc_bounds = plot_acc_distribution(AV_straight_acc, max_acc, fontsize, colors, alphas, folder, 'straight_acc_distribution')
    straight_dec_bounds = plot_acc_distribution(AV_straight_dec, max_dec, fontsize, colors, alphas, folder, 'straight_dec_distribution', reverse=True)
    turn_acc_bounds = plot_acc_distribution(AV_turn_acc, max_acc, fontsize, colors, alphas, folder, 'turn_acc_distribution')
    turn_dec_bounds = plot_acc_distribution(AV_turn_dec, max_dec, fontsize, colors, alphas, folder, 'turn_dec_distribution', reverse=True)

    with open(folder + '/bounds.txt', 'w') as f:
        f.write(f'straight_acc_bounds: {straight_acc_bounds}\n')
        f.write(f'straight_dec_bounds: {straight_dec_bounds}\n')
        f.write(f'turn_acc_bounds: {turn_acc_bounds}\n')
        f.write(f'turn_dec_bounds: {turn_dec_bounds}\n')

    # plot the initial conditions
    data_file = folder + '/init_conditions.csv'
    key, data = read_csv(data_file)

    right_turn_straight_plot_data = []
    right_turn_turn_plot_data = []
    for straight_dis, straight_vel, straight_acc, turn_dis, turn_vel, turn_acc, flag in data:
        if straight_vel > 0 and turn_vel > 0:
            right_turn_straight_plot_data.append([straight_vel, turn_dis])
            right_turn_turn_plot_data.append([turn_vel, straight_dis])

    plot_init_condition_distribution(
        right_turn_straight_plot_data,
        [0, 20],
        [0, 50],
        1,
        5,
        fontsize,
        r'BV speed ($m/s$)',
        r'Distance ($m$)',
        folder,
        'right_turn_straight_init_condition_distribution'
    )

    plot_init_condition_distribution(
        right_turn_turn_plot_data,
        [0, 20],
        [0, 50],
        1,
        5,
        fontsize,
        r'BV speed ($m/s$)',
        r'Distance ($m$)',
        folder,
        'right_turn_turn_init_condition_distribution'
    )

import sys
import os

# Get the directory of the current script
dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(dir)

from settings import *
from utils import *

colors = list(COLORS.values())
alphas = list(ALPHAS.values())
fontsize = FONTSIZE


if __name__ == '__main__':
    folder = 'output/data_process/roundabout'

    # obtain and plot risk level bounds
    data_file = folder + '/data.csv'
    data = pd.read_csv(data_file)

    AV_inner_acc = []
    AV_inner_dec = []
    AV_outer_acc = []
    AV_outer_dec = []
    for inner_acc, outer_acc in data.values:
        if min_acc < inner_acc < max_acc:
            AV_inner_acc.append(inner_acc)
        elif max_dec < inner_acc < min_dec:
            AV_inner_dec.append(inner_acc)
        if min_acc < outer_acc < max_acc:
            AV_outer_acc.append(outer_acc)
        elif max_dec < outer_acc < min_dec:
            AV_outer_dec.append(outer_acc)
    AV_inner_acc.sort(reverse=True)
    AV_inner_dec.sort()
    AV_outer_acc.sort(reverse=True)
    AV_outer_dec.sort()

    inner_acc_bounds = plot_acc_distribution(AV_inner_acc, max_acc, fontsize, colors, alphas, folder, 'inner_acc_distribution')
    inner_dec_bounds = plot_acc_distribution(AV_inner_dec, max_dec, fontsize, colors, alphas, folder, 'inner_dec_distribution', reverse=True)
    outer_acc_bounds = plot_acc_distribution(AV_outer_acc, max_acc, fontsize, colors, alphas, folder, 'outer_acc_distribution')
    outer_dec_bounds = plot_acc_distribution(AV_outer_dec, max_dec, fontsize, colors, alphas, folder, 'outer_dec_distribution', reverse=True)
    with open(folder + '/bounds.txt', 'w') as f:
        f.write(f'inner_acc_bounds: {inner_acc_bounds}\n')
        f.write(f'inner_dec_bounds: {inner_dec_bounds}\n')
        f.write(f'outer_acc_bounds: {outer_acc_bounds}\n')
        f.write(f'outer_dec_bounds: {outer_dec_bounds}\n')
    # plot the initial conditions
    data_file = folder + '/init_conditions.csv'
    key, data = read_csv(data_file)

    roundabout_inner_plot_data = []
    roundabout_outer_plot_data = []
    for inner_dis, inner_sp, outer_dis, outer_sp in data:
        if inner_dis > 50 or outer_dis > 50:
            continue
        if inner_sp > 20 or outer_sp > 20:
            continue
        roundabout_inner_plot_data.append([outer_sp, inner_dis])
        roundabout_outer_plot_data.append([inner_sp, outer_dis])

    plot_init_condition_distribution(
        roundabout_inner_plot_data,
        [0, 20],
        [0, 50],
        2,
        5,
        fontsize,
        r'BV speed ($m/s$)',
        r'Distance ($m$)',
        folder,
        'inner_init_condition_distribution'
    )

    plot_init_condition_distribution(
        roundabout_outer_plot_data,
        [0, 20],
        [0, 50],
        2,
        5,
        fontsize,
        r'BV speed ($m/s$)',
        r'Distance ($m$)',
        folder,
        'outer_init_condition_distribution'
    )

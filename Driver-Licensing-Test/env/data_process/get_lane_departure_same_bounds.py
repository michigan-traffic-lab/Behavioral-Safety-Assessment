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
    folder = 'output/data_process/lane_departure_same'

    # obtain and plot risk level bounds
    file = folder + '/data.csv'

    acc = []
    data = pd.read_csv(file)
    for relative_dis, lateral_dis, BV_sp, AV_sp, AV_acc in data.values:
        if max_dec < AV_acc < min_dec and 30 > relative_dis > 0:
            acc.append(AV_acc)
    acc.sort()

    acc_bounds = plot_acc_distribution(
        acc,
        max_dec,
        fontsize,
        colors,
        alphas,
        folder,
        'dec_distribution',
        reverse=True,
    )
    with open(folder + '/bounds.txt', 'w') as f:
        f.write('acc bounds: ' + str(acc_bounds) + '\n')

    # plot initial condition distribution
    file = folder + '/init_conditions.csv'
    data = pd.read_csv(file)
    plot_data = []
    for init_dis, init_sp in data.values:
        if init_dis < 30 and init_sp > 0:
            plot_data.append([init_sp, init_dis])

    plot_init_condition_distribution(
        plot_data,
        [0, 4],
        [0, 30],
        0.3,
        3,
        fontsize,
        r'Relative speed ($m/s$)',
        r'Relative distance ($m$)',
        folder
    )

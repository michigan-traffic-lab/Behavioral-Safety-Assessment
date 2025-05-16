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
    folder = 'output/data_process/lane_departure_opposite'

    # obtain and plot risk level bounds
    file = folder + '/data.csv'

    lc_time_list = []

    data = pd.read_csv(file)
    for lc_time, in data.values:
        if lc_time > min_lc_time:
            lc_time_list.append(lc_time)
    lc_time_list.sort()

    lc_bounds = plot_acc_distribution(
        lc_time_list,
        min_lc_time,
        fontsize,
        colors,
        alphas,
        folder,
        'lane_changing_time_distribution',
        reverse=True,
    )

    with open(folder + '/bounds.txt', 'w') as f:
        f.write('lane changing time bounds: ' + str(lc_bounds) + '\n')
    
    # plot initial condition distribution
    file = folder + '/init_conditions.csv'
    data = pd.read_csv(file)
    plot_data = []
    for init_dis, init_sp in data.values:
        if init_sp > 0:
            plot_data.append([init_sp, init_dis])

    plot_init_condition_distribution(
        plot_data,
        [0, 20],
        [0, 100],
        1,
        5,
        fontsize,
        r'Relative speed ($m/s$)',
        r'Relative distance ($m$)',
        folder
    )

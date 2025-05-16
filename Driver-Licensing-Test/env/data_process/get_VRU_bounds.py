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
    folder = 'output/data_process/vru'
    file = folder + '/data.csv'
    AV_dec_list = []
    key, data = read_csv(file)
    for line in data:
        if max_dec < line[2] < min_dec:
            AV_dec_list.append(line[2])

    AV_dec_list.sort()
    dec_bounds = plot_acc_distribution(
        AV_dec_list,
        max_dec,
        fontsize,
        colors,
        alphas,
        folder,
        'dec_distribution',
        reverse=True
    )

    with open(folder + '/bounds.txt', 'w') as f:
        f.write('dec bounds: ' + str(dec_bounds) + '\n')

    file = folder + '/init_conditions.csv'
    key, data = read_csv(file)
    plot_data = []
    for dis, veh_sp, veh_acc, vru_sp in data:
        plot_data.append([vru_sp, dis])

    plot_init_condition_distribution(
        plot_data,
        [0, 10],
        [0, 50],
        1,
        2,
        fontsize,
        r'VRU speed ($m/s$)',
        r'Longitudinal distance ($m$)',
        folder
    )

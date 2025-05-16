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
    folder = 'output/data_process/car_following'
    file = folder + '/data.csv'
    AV_acc_list = []
    AV_dec_list = []
    data = pd.read_csv(file)
    for dis, AV_sp, AV_acc, BV_sp, BV_acc in data.values:
        if AV_acc < min_dec and AV_acc > max_dec:
            AV_dec_list.append(AV_acc)
        if AV_acc > min_acc and AV_acc < max_acc:
            AV_acc_list.append(AV_acc)
    AV_acc_list.sort(reverse=True)
    AV_dec_list.sort()

    acc_bounds = plot_acc_distribution(
        AV_acc_list,
        max_acc,
        fontsize,
        colors,
        alphas,
        folder,
        'acc_distribution'
    )
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
        f.write('acc bounds: ' + str(acc_bounds) + '\n')
        f.write('dec bounds: ' + str(dec_bounds) + '\n')

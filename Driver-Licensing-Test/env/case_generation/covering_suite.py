import math
import matplotlib.pyplot as plt
import random


class CoveringSuite():
    def __init__(self) -> None:
        self.matrix = [[]]
        self.cases = {
            'low': [],
            'mid': [],
            'high': []
        }

    def k_wise_covering_suite(self, risk_levels, risk_level_bounds, x_range, x_res_dict, y_res_dict, k=1):
        if isinstance(risk_level_bounds[0], dict):
            for i in range(len(risk_level_bounds)):
                if risk_level_bounds[i]['risk_level'] in risk_levels:
                    x_res = x_res_dict[risk_level_bounds[i]['risk_level']]
                    y_res = y_res_dict[risk_level_bounds[i]['risk_level']]
                    self._divide_area_into_square(risk_level_bounds[i]['risk_level'], risk_level_bounds[i]['upper'], risk_level_bounds[i]['lower'], risk_level_bounds[i]['v_BV'], x_res, y_res, k)
        else:
            for i in range(1, len(risk_level_bounds) - 2):
                x_res = x_res_dict[risk_levels[i-1]]
                y_res = y_res_dict[risk_levels[i-1]]
                self._divide_area_into_square(risk_levels[i-1], risk_level_bounds[i + 1], risk_level_bounds[i], x_range, x_res, y_res, k)
        return self.cases

    def _divide_area_into_square(self, risk_level, upper_bound, lower_bound, x_range, x_res, y_res, k=1):
        _upper_bound = []
        _lower_bound = []
        _x_range = []
        y_range = max(upper_bound) - min(lower_bound)
        for x, y_upper, y_lower in zip(x_range, upper_bound, lower_bound):
            if y_upper != y_lower:
                _upper_bound.append(y_upper)
                _lower_bound.append(y_lower)
                _x_range.append(x)
        if len(_x_range) > 0:
            x_min = min(_x_range)
            x_max = max(_x_range)
            y_min = min(_lower_bound)
            y_max = max(_upper_bound)
            x_n = math.ceil((x_max - x_min) / x_res)
            y_n = math.ceil((y_max - y_min) / y_res)
            self.matrix = [[1 for i in range(x_n)] for j in range(y_n)]
            inds = []
            for i in range(x_n):
                ind = 0
                while _x_range[ind + 1] <= x_min + i * x_res:
                    ind += 1
                    if ind >= len(_x_range) - 1:
                        break
                inds.append(ind)
            for j in range(y_n):
                for i in range(x_n):
                    ind = inds[i]
                    if x_max - (x_min + i * x_res) < 0.2:
                        self.matrix[j][i] = 0
                    else:
                        flag = True
                        for tmp_ind in range(len(_x_range)):
                            if x_min + i * x_res <= _x_range[tmp_ind] < x_min + (i+1) * x_res:
                                if not (_upper_bound[tmp_ind] < y_max - (j+1) * y_res or _lower_bound[tmp_ind] > y_max - j * y_res):
                                    flag = False
                                    break
                        if flag:
                            self.matrix[j][i] = 0
            number = 0
            for j in range(y_n):
                for i in range(x_n):
                    number += self.matrix[j][i]
                    if self.matrix[j][i]:
                        if i == x_n - 1:
                            x, y = self._get_single_case(_upper_bound, _lower_bound, _x_range, inds[i], -1, [x_min + x_res * i, y_max - y_res * j], x_res, y_res)
                        else:
                            x, y = self._get_single_case(_upper_bound, _lower_bound, _x_range, inds[i], inds[i+1], [x_min + x_res * i, y_max - y_res * j], x_res, y_res)
                        if x!= 0 and y != 0:
                            self.cases[risk_level].append([x, y])

    def _plot(self, x_min, y_max, x_res, y_res):
        for j in range(len(self.matrix)):
            for i in range(len(self.matrix[j])):
                if self.matrix[j][i]:
                    plt.plot([x_min + i * x_res, x_min + (i+1) * x_res], [y_max - j * y_res, y_max - j * y_res], 'k')
                    plt.plot([x_min + (i+1) * x_res, x_min + (i+1) * x_res], [y_max - j * y_res, y_max - (j+1) * y_res], 'k')
                    plt.plot([x_min + i * x_res, x_min + (i+1) * x_res], [y_max - (j+1) * y_res, y_max - (j+1) * y_res], 'k')
                    plt.plot([x_min + i * x_res, x_min + i * x_res], [y_max - j * y_res, y_max - (j+1) * y_res], 'k')
    
    def _get_single_case(self, upper_bound, lower_bound, x_array, ind1, ind2, square_point, x_res, y_res):
        square_x, square_y = square_point
        if x_array[1] - x_array[0] >= x_res:
            ratio1 = (square_x - x_array[ind1]) / (x_array[ind1 + 1] - x_array[ind1])
            ratio2 = (square_x + x_res - x_array[ind2]) / (x_array[ind2 + 1] - x_array[ind2])
            _upper_bound1 = upper_bound[ind1] + ratio1 * (upper_bound[ind1 + 1] - upper_bound[ind1])
            _lower_bound1 = lower_bound[ind1] + ratio1 * (lower_bound[ind1 + 1] - lower_bound[ind1])
            if ind2 == -1:
                _upper_bound2 = upper_bound[ind2]
                _lower_bound2 = lower_bound[ind2]
            else:
                _upper_bound2 = upper_bound[ind2] + ratio2 * (upper_bound[ind2 + 1] - upper_bound[ind2])
                _lower_bound2 = lower_bound[ind2] + ratio2 * (lower_bound[ind2 + 1] - lower_bound[ind2])
            if (_lower_bound1 > square_y and _lower_bound2 > square_y) or (_upper_bound1 < square_y - y_res and _upper_bound2 < square_y - y_res):
                return [0, 0]
            else:
                _upper_bound = []
                _lower_bound = []
                _x_array = []
                N = 50
                if ind2 == -1:
                    step = (x_array[-1] - square_x) / N
                else:
                    step = x_res / N
                for i in range(N):
                    x = square_x + i * step
                    tmp_upper_bound = min(_upper_bound1 + i / N * (_upper_bound2 - _upper_bound1), square_y)
                    tmp_lower_bound = max(_lower_bound1 + i / N * (_lower_bound2 - _lower_bound1), square_y - y_res)
                    if tmp_upper_bound > tmp_lower_bound:
                        _upper_bound.append(tmp_upper_bound)
                        _lower_bound.append(tmp_lower_bound)
                        _x_array.append(x)
        else:
            _upper_bound = []
            _lower_bound = []
            _x_array = []
            N = 50
            if ind2 == -1:
                step = (x_array[-1] - square_x) / N
            else:
                step = x_res / N
            for i in range(N):
                x = square_x + i * step
                for tmp_ind in range(len(x_array) - 1):
                    if x_array[tmp_ind] <= x and x_array[tmp_ind + 1] > x:
                        tmp_upper_bound = min((upper_bound[tmp_ind + 1] - upper_bound[tmp_ind]) * (x - x_array[tmp_ind]) / (x_array[tmp_ind + 1] - x_array[tmp_ind]) + upper_bound[tmp_ind], square_y)
                        tmp_lower_bound = max((lower_bound[tmp_ind + 1] - lower_bound[tmp_ind]) * (x - x_array[tmp_ind]) / (x_array[tmp_ind + 1] - x_array[tmp_ind]) + lower_bound[tmp_ind], square_y - y_res)
                        break
                if tmp_upper_bound > tmp_lower_bound:
                    _upper_bound.append(tmp_upper_bound)
                    _lower_bound.append(tmp_lower_bound)
                    _x_array.append(x)
        
        if len(_x_array) == 0:
            return [0, 0]
        else:
            length = []
            for tmp_upper_bound, tmp_lower_bound, tmp_x in zip(_upper_bound, _lower_bound, _x_array):
                if len(length) == 0:
                    length.append(tmp_upper_bound - tmp_lower_bound)
                else:
                    length.append(length[-1] + tmp_upper_bound - tmp_lower_bound)
            rand_n = random.random() * length[-1]
            tmp_ind = 0
            for l in length:
                if rand_n < l:
                    x = step * tmp_ind + _x_array[0] + random.random() * step
                    y = _lower_bound[tmp_ind] + l - rand_n
                    break
                tmp_ind = tmp_ind + 1
        return [x, y]

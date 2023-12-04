# Decision-making by the combined BPA
import math

def decision_making(BPA):
    dimension = len(BPA)
    types_num = round(math.log2(dimension))
    prob = [0 for ii in range(types_num)]
    for index in range(dimension):
        if BPA[index] != 0:
            mass = BPA[index]
            bin_index = "{:0>10b}".format(index)
            set_num = 0
            for ii in range(types_num):
                if bin_index[-(ii+1)] == '1':
                    set_num += 1
            if set_num != 0:
                mass_divided = mass/set_num
                for ii in range(types_num):
                    if bin_index[-(ii+1)] == '1':
                        prob[ii] += mass_divided
    return prob.index(max(prob))
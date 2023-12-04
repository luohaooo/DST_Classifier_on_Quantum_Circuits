import numpy as np

# Generate BPAs
def calc_Gaussian(mean,cov,x):
    return (1/np.sqrt(2*np.pi*cov))*np.exp(-np.square(x-mean)/(2*cov))

def calc_mix_Gaussian(weights,means,covs,x):
    components_num = len(weights)
    prob = 0
    for item in range(components_num):
        prob += weights[item]*calc_Gaussian(means[item],covs[item],x)
    return prob[0]

def generate_BPAs(x, weights, means, covariances):
    # x: input testing data (4 atrributes)
    attributes_num = len(x)
    types_num = len(means)
    # BPAs for each attribute
    BPAs = [[] for attribute in range(attributes_num)]
    alphas = [[] for attribute in range(attributes_num)]
    for attribute in range(attributes_num):
        attribute_value = x[attribute]
        # calculate GMM distribution value
        f = [[] for type in range(types_num)]
        for type in range(types_num):
            f[type] = calc_mix_Gaussian(weights[type][attribute],means[type][attribute],covariances[type][attribute],attribute_value)
        f = np.array(f)
        # calculate pi_0 and pi_1
        pi1 = f/max(f)
        pi0 = np.ones(types_num)-pi1
        # calculate the rotation angles
        alpha = np.arctan((pi1+1e-100)/(pi0+1e-100))
        alphas[attribute] = alpha
        # calculate BPA and record
        dimension = 2**types_num
        BPA = [[] for ii in range(dimension)]
        for ii in range(dimension):
            # binary represenation
            bits = "{:0>10b}".format(ii)
            # mass
            m = 1
            for jj in range(types_num):
                bit = bits[-(jj+1)]
                if bit == '1':
                    m *= pi1[jj]
                if bit == '0':
                    m *= pi0[jj]
            BPA[ii] = m
        BPAs[attribute] = BPA
    return BPAs, alphas
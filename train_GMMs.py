from sklearn.mixture import GaussianMixture

import warnings
warnings.filterwarnings('ignore')

# GMM coefficients
def learn_GMM_coefficients(training_set, components_num):
    n = len(training_set)
    m = len(training_set[0])
    weights = [[[[] for item in range(components_num)] for col in range(m)] for row in range(n)]
    means = [[[[] for item in range(components_num)] for col in range(m)] for row in range(n)]
    covariances = [[[[] for item in range(components_num)] for col in range(m)] for row in range(n)]
    for ii in range(n):
        for jj in range(m):
            gmm_coefficients = GaussianMixture(n_components=3).fit(training_set[ii][jj].reshape(-1, 1))
            for item in range(components_num):
                weights[ii][jj][item] = gmm_coefficients.weights_[item]
                means[ii][jj][item] = gmm_coefficients.means_[item][0]
                covariances[ii][jj][item] = gmm_coefficients.covariances_[item][0]
    return weights, means, covariances
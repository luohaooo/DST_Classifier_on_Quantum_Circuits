import numpy as np

# divide data into train_data and test_data
def train_test(all_data, train_ratio):
    train_len = round(len(all_data)*train_ratio)
    np.random.shuffle(all_data)
    train_data = np.array(all_data[0:train_len])
    test_data = np.array(all_data[train_len:len(all_data)])
    return train_data, test_data

def save_training_testing_data(pro_data, train_ratio):
    n = len(pro_data) # types num
    m = len(pro_data[0][0])-1 # attribute num
    training_set = [[[] for col in range(m)] for row in range(n)]
    testing_set = [[] for row in range(n)]
    # for each type
    for ii in range(n):
        train_data, test_data = train_test(pro_data[ii], train_ratio)
        testing_set[ii] = test_data[:,0:m]
    #     for each attribute
        for jj in range(m):
            training_set[ii][jj] = train_data[:,jj]
    return training_set, testing_set
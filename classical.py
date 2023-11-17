import numpy as np
def QfrM(n):
    pattern = np.array([[1, 1],[0, 1]])
    matrix = pattern
    if n>1:
        for ii in range(n-1):
            matrix = np.kron(matrix,pattern)
    return matrix

def MfrQ(n):
    pattern = np.array([[1, -1],[0, 1]])
    matrix = pattern
    if n>1:
        for ii in range(n-1):
            matrix = np.kron(matrix,pattern)
    return matrix

def combine_two_BPAs_classical(m1,m2):
    dimension = len(m1)
    size = round(np.log2(dimension))
    
    QfrM_matrix = QfrM(size)
    MfrQ_matrix = MfrQ(size)
    q1 = np.dot(QfrM_matrix,m1.reshape(dimension,1))
    q2 = np.dot(QfrM_matrix,m2.reshape(dimension,1))
    q = q1*q2
    b = np.dot(MfrQ_matrix,q)
    return b.reshape(1,dimension)[0]

def combine_BPAS_classical(BPAs):
    num = len(BPAs)
    combined_BPA = combine_two_BPAs_classical(np.array(BPAs[0]),np.array(BPAs[1]))
    if num == 2:
        return combined_BPA
    else:
        for ii in range(num-2):
            combined_BPA = combine_two_BPAs_classical(combined_BPA,np.array(BPAs[ii+2]))
    return combined_BPA


# m1 = np.array([0.1,0.2,0.3,0.4])
# m2 = np.array([0,0,0.8,0.2])
# print(combine_two_BPAs_classical(m1,m2))
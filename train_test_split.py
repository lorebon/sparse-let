import numpy as np


def cross_validate(x, y, k):
    '''performs cross-validation split while 
    ensuring the ratio of instances for each class
    remains the same'''
    indexes = [[i for i in range(len(y)) if y[i]==e] for e in set(y)]
    f_indexes = [[i for ind in indexes for i in ind[m*len(ind)//k : (m+1)*len(ind)//k]] for m in range(k)]

    result = []
    for m in range(k):
        complement = set([i for i in range(len(y))])-set(f_indexes[m])
        train_x = [x[i] for i in complement]
        test_x = [x[i] for i in f_indexes[m]]
        train_y = [y[i] for i in complement]
        train_y = np.array(train_y)
        test_y = [y[i] for i in f_indexes[m]]
        test_y = np.array(test_y)
        result.append((train_x, train_y, test_x, test_y))

    return result

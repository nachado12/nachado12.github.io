import numpy as np
from matplotlib import pyplot as plt


def data_loader(X, Y=None, batch_size=64, shuffle=False):
    """Iterator that yields one batch of data at a time"""
    N = X.shape[1]
    if shuffle:
        perm = np.random.permutation(N)
        X = X[:,perm]
        print(Y)
        if Y is not None:
            Y = Y[:,perm]
    start = 0
    while start < N:
        end = start + batch_size
        if end > N:
            end = N
        if Y is not None:
            yield X[:, start:end], Y[:, start:end]
        else:
            yield X[:,start:end]
        start = end
    return

def acc(pred_label, Y):
    ''' pred_label: (N,) vector; Y: (N,K) one hot encoded ground truth'''
    num = len(pred_label)
    return sum(Y[pred_label[i], i] == 1 for i in range(num))*1.0/num

def raiseNotDefined():
  print("Method not implemented: %s" % inspect.stack()[1][3])
  sys.exit(1)


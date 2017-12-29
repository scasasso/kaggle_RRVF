import numpy as np


def rmsle(y_true, y_pred):

    assert len(y_true) == len(y_pred), 'You must provide prediction/observation arrays of same length'

    n = len(y_true)

    return np.sqrt(np.sum(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)) / n)


import numpy as np


def get_output(data):
    return np.maximum(0, data)


def hidden_error(layer, error):
    gradient = np.ones(layer.shape)
    gradient[layer < 0] = 0
    return gradient * error

import numpy as np
import standardize as sd


def get_output(data):
    return data


def hidden_error(layer, error):
    return error


def cross_entropy(output, y):
    # loss as cross entropy error
    return ((y-output)**2).sum()/2


def output_error(output, y):
    # output error for sigmoid function and cross entropy error
    return output - y


def get_predict(output):
    return output


def label_process(label):
    return label


def data_processing(dataset, stat_list=-1):
    feature, data, label, stat_list0 = sd.data_processing(dataset, stat_list)
    label = label.astype(int)

    return feature, data, label, stat_list0

import numpy as np
import function.standardize as sd


def get_output(data):
    return 2 / (1 + np.exp(-2*data)) - 1


def hidden_error(layer, error):
    return (1 - layer**2) * error


def cross_entropy(output, y):
    # loss as cross entropy error
    output[output == 1] = 0.999999999999999
    output[output == -1] = -0.999999999999999
    return -y * np.log(1+output) - (1 - y) * np.log(1-output) + np.log(2)


def output_error(output, y):
    # output error for sigmoid function and cross entropy error
    return output + 1 - 2 * y


def get_predict(output):
    return output >= 0


def label_process(label):
    return label


def data_processing(dataset, stat_list=-1):
    feature, data, label, stat_list0 = sd.data_processing(dataset, stat_list)

    # convert label to 0/1
    label[label == feature[-1][1][0]] = 0
    label[label == feature[-1][1][1]] = 1
    label = label.astype(int)

    return feature, data, label, stat_list0

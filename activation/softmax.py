import numpy as np
import function.standardize as sd


def cross_entropy(output, y):
    # loss as cross entropy error
    # if y is not one-hot encoded
    # return -np.log(output[np.arange(y.shape[0]), y[:, 0]])
    output[output == 0] = 0.000000000000001
    return -y*np.log(output)


def get_output(data):
    # output of neural network for softmax function
    data = data - np.max(data)
    output = np.exp(data)
    outsum = np.sum(output, axis=0)
    return output/outsum


def output_error(output, y):
    # output error for sigmoid function and cross entropy error
    return output.T-y


def get_predict(output):
    # if y is not one-hot encoded
    # return np.argmax(output, axis=1)[:, None]
    return np.argmax(output, axis=1)[:, None]


def label_process(label):
    return np.argmax(label, axis=1)[:, None]


def data_processing(dataset, stat_list=-1):
    feature, data, label, stat_list0 = sd.data_processing(dataset, stat_list)

    data_num = data.shape[1]
    # convert label to 0/1
    label, stat = sd.hot_encoding(-1, label, feature, data_num)
    label = label.astype(int)

    return feature, data, label, stat_list0

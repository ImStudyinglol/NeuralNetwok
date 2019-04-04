import numpy as np


# index 0 feature
# index 1 data
# index 2 label
# index 3 mean and std


def standardize(i, data, stat_list):
    if stat_list == -1:
        # stat_list = -1 means standardize train data
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            std = 1
    else:
        # else means standardize test data
        mean = stat_list[i][0]
        std = stat_list[i][1]
    return (data-mean) / std, [mean, std]


def hot_encoding(i, data, feature, data_num):
    # initialize one hot encoding matrix
    hot = np.zeros((data_num, len(feature[i][1])))
    for j in range(data_num):
        # change the index th value to 1
        # index is the index of value in the feature value list
        index = feature[i][1].index(data[j])
        hot[j, index] = 1
    return hot, [0, 0]


def data_processing(dataset, stat_list):
    # split dataset into data and feature
    feature = dataset['metadata']['features']
    dataset = np.array(dataset['data'])

    # get the size of feature and data
    feature_num = len(feature)
    data_num = dataset.shape[0]
    # get the label of data
    label = dataset[:, -1][:, None]
    # initialize bias of weights
    data = np.ones((0, data_num))
    stat_list0 = []

    for i in range(feature_num-1):
        f_i = dataset[:, i]
        if feature[i][1] == 'numeric':
            # standardize numeric features
            f_i = f_i.astype(float)
            f_i, stat = standardize(i, f_i, stat_list)
            data = np.append(data, [f_i], axis=0)
        else:
            # one hot encode discrete features
            f_i, stat = hot_encoding(i, f_i, feature, data_num)
            data = np.append(data, f_i.T, axis=0)
        # update mean and std
        stat_list0.append(stat)

    return feature, data, label, stat_list0

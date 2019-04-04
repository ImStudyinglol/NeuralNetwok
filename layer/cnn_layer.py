from layer.nn_layer import *


class Convolution(Layer):

    type = 'v'

    # para = 0, tuple of filter size
    # para = 1, number of filter
    # para = 2, number of padding
    # para = 3, number of stride

    def __init__(self, para, previous=None):
        self.set_previous(previous)
        previous_data = previous.data
        self._previous_shape = previous_data.shape
        previous_length = previous_data.shape[-2]
        previous_width = previous_data.shape[-1]
        self._channel = previous_data.shape[-3]
        self._filter_length = para[0][0]
        self._filter_width = para[0][1]
        self._filter_number = para[1]
        self._padding = para[2]
        self._stride = para[3]
        self._data_length = self._output_size(previous_length, self._filter_length)
        self._data_width = self._output_size(previous_width, self._filter_width)
        self.data = np.empty((self._filter_number, self._data_length, self._data_width))
        self._padding_data = np.empty((self._channel,
                                       previous_length + 2 * self._padding,
                                       previous_width + 2 * self._padding))
        self.filter = np.random.uniform(low=-0.01, high=0.01,
                                        size=(self._filter_number, self._channel,
                                              self._filter_length, self._filter_width))
        self.bias = np.zeros((self._filter_number, 1, 1))
        self.error = np.empty(self.data.shape)
        self._expend_length = self._expend_size(self._data_length)
        self._expend_width = self._expend_size(self._data_width)
        self._expend_error = self._error_expand()
        self._output = LReLU.get_output
        self._hidden_error = LReLU.hidden_error

    def _output_size(self, previous_size, filter_size):
        # compute the size of output data
        return int((previous_size - filter_size + 2 * self._padding) / self._stride + 1)

    def _expend_size(self, size):
        # compute the size of expend error
        return int((size - 1) * self._stride + 1)

    @staticmethod
    def _zero_padding(data, pd):
        if pd == 0:
            return data
        else:
            depth = data.shape[-3]
            length = data.shape[-2]
            width = data.shape[-1]
            # construct a new matrix with current size + 2*padding
            new = np.zeros((depth, length + pd * 2, width + pd * 2))
            # put current data at the center of new matrix
            new[:, pd: length + pd, pd: pd + width] = data
            return new

    @staticmethod
    def _convolution(data, filters, out, stride):
        length = filters.shape[-2]
        width = filters.shape[-1]
        for i in np.arange(out.shape[-2]):
            s_i = i * stride
            for j in np.arange(out.shape[-1]):
                s_j = j * stride
                # compute convolution of # filter number elements at a time
                # draw a (channel, filter_len, filter_wid) matrix from current data
                # filter is a (filter_num, channel, filter_len, filter_wid) matrix
                out[..., i, j] = np.sum(data[..., s_i: s_i + length, s_j: s_j + width] * filters,
                                        axis=(-1, -2, -3))
        return out

    @staticmethod
    def _backward_convolution(data, filters, out):
        length = filters.shape[-2]
        width = filters.shape[-1]
        for i in np.arange(out.shape[-2]):
            for j in np.arange(out.shape[-1]):
                # compute convolution of # filter number elements at a time
                # draw a (channel, filter_len, filter_wid) matrix from current data
                # filter is a (filter_num, channel, filter_len, filter_wid) matrix
                out[..., i, j] = np.sum(data[..., i: i + length, j: j + width] * filters,
                                        axis=(-1, -2))
        return out

    def _error_expand(self):
        # expend error to stride = 1
        error = np.zeros((self._filter_number,
                          self._expend_length, self._expend_width))
        for i in np.arange(self.error.shape[-2]):
            s_i = i * self._stride
            for j in np.arange(self.error.shape[-2]):
                s_j = j * self._stride
                error[:, s_i, s_j] = self.error[:, i, j]
        return error

    def forward(self, data):
        self._padding_data = self._zero_padding(data, self._padding)
        data = self._convolution(self._padding_data, self.filter,
                                 self.data, self._stride) + self.bias
        self.data = self._output(data)
        self._next.forward(self.data)

    def backward(self):
        self.error = self._hidden_error(self.data, self._next.cal_error())
        self._expend_error = self._error_expand()
        self._previous.backward()
        gradient = self._backward_convolution(self._padding_data,
                                              self._expend_error[:, None, ...],
                                              np.empty(self.filter.shape))
        self.filter = self.filter - self.rate * gradient
        self.bias = self.bias - self.rate * np.sum(self.error,
                                                   axis=(-1, -2), keepdims=True)

    def cal_error(self):
        error = self._zero_padding(self._expend_error,
                                   self._filter_length - self._padding - 1)
        # invert last 2 axis
        filters = self.filter[..., ::-1, ::-1]
        # inter-change first 2 axis
        filters = np.swapaxes(filters, 0, 1)
        return self._convolution(error, filters, self._previous.error, 1)

    def predict(self, data):
        # zero padding
        pd = self._padding
        if pd == 0:
            self._padding_data = data
        else:
            depth = data.shape[-3]
            length = data.shape[-2]
            width = data.shape[-1]
            num = data.shape[0]
            self._padding_data = np.zeros((num, depth,
                                           length + pd * 2, width + pd * 2))
            self._padding_data[..., pd: length + pd, pd: pd + width] = data
        self._padding_data = self._padding_data[:, None, ...]
        # convolution
        self.data = np.empty((data.shape[0], self._filter_number,
                              self._data_length, self._data_width))
        self.data = self._convolution(self._padding_data, self.filter,
                                      self.data, self._stride) + self.bias
        self.data = self._output(self.data)
        self._next.predict(self.data)
        # reinitialize data
        self.data = np.empty((self._filter_number,
                              self._data_length, self._data_width))

    def set_activation(self, activation):
        self._output = activation.get_output
        self._hidden_error = activation.hidden_error
        self._next.set_activation(activation)


class MaxPooling(Layer):

    type = 'x'

    # para = 0, tuple of filter size
    # para = 1, number of stride

    def __init__(self, para, previous=None):
        self.set_previous(previous)
        previous_data = previous.data
        self._previous_shape = previous_data.shape
        previous_length = previous_data.shape[1]
        previous_width = previous_data.shape[2]
        self._channel = previous_data.shape[0]
        self._filter_length = para[0][0]
        self._filter_width = para[0][1]
        self._stride = para[1]
        self._data_length = self._output_size(previous_length, self._filter_length)
        self._data_width = self._output_size(previous_width, self._filter_width)
        self.data = np.empty((self._channel, self._data_length, self._data_width))
        self.error = np.empty(self.data.shape)
        # a boolean matrix store the index of max pooling value in previous data
        self._index = np.empty((previous_data.shape[1], previous_data.shape[2],
                                self._channel, self._filter_length, self._filter_width), dtype=bool)

    def _output_size(self, previous_size, filter_size):
        # compute the size of output data
        return int((previous_size - filter_size) / self._stride + 1)

    def forward(self, data):
        for i in np.arange(self.data.shape[-2]):
            s_i = i * self._stride
            for j in np.arange(self.data.shape[-1]):
                s_j = j * self._stride
                # draw a (channel, filter_len, filter_wid) matrix from current data
                data_it = data[:, s_i: s_i + self._filter_length, s_j: s_j + self._filter_width]
                # compute pooling of # channel elements at a time
                # max pooling of data matrix
                out_it = self.data[:, i, j] = np.max(data_it, axis=(-1, - 2))
                # find the index of that max value in data matrix
                self._index[i, j, ...] = data_it == out_it[:, None, None]
        self._next.forward(self.data)

    def backward(self):
        self.error = self._next.cal_error()
        self._previous.backward()

    def cal_error(self):
        formal_error = np.zeros(self._previous_shape)
        for i in np.arange(self.error.shape[-2]):
            s_i = i * self._stride
            for j in np.arange(self.error.shape[-1]):
                s_j = j * self._stride
                # update error with input error, at the index stored before
                formal_error[:, s_i: s_i + self._filter_length, s_j: s_j + self._filter_width] \
                    = formal_error[:, s_i: s_i + self._filter_length, s_j: s_j + self._filter_width] \
                    + self._index[i, j, ...] * self.error[:, i, j][:, None, None]
        return formal_error

    def predict(self, data):
        self.data = np.empty((data.shape[0], self._channel,
                              self._data_length, self._data_width))
        for i in np.arange(self.data.shape[-2]):
            s_i = i * self._stride
            for j in np.arange(self.data.shape[-1]):
                s_j = j * self._stride
                self.data[..., i, j] = \
                    np.max(data[..., s_i: s_i + self._filter_length, s_j: s_j + self._filter_width],
                           axis=(-1, - 2))
        self._next.predict(self.data)
        # reinitialize data
        self.data = np.empty((self._channel, self._data_length, self._data_width))

    def set_activation(self, activation):
        self._next.set_activation(activation)


class MeanPooling(Layer):

    type = 'm'

    def __init__(self, para, previous=None):
        self.set_previous(previous)
        previous_data = previous.data
        self._previous_shape = previous_data.shape
        previous_length = previous_data.shape[1]
        previous_width = previous_data.shape[2]
        self._channel = previous_data.shape[0]
        self._filter_length = para[0][0]
        self._filter_width = para[0][1]
        self._stride = para[1]
        self._data_length = self._output_size(previous_length, self._filter_length)
        self._data_width = self._output_size(previous_width, self._filter_width)
        self.data = np.empty((self._channel, self._data_length, self._data_width))
        self.error = np.empty(self.data.shape)

    def _output_size(self, previous_size, filter_size):
        # compute the size of output data
        return int((previous_size - filter_size) / self._stride + 1)

    def forward(self, data):
        for i in np.arange(self.data.shape[-2]):
            s_i = i * self._stride
            for j in np.arange(self.data.shape[-1]):
                s_j = j * self._stride
                # compute pooling of # channel elements at a time
                # mean pooling of data matrix
                self.data[:, i, j] = \
                    np.mean(data[:, s_i: s_i + self._filter_length, s_j: s_j + self._filter_width],
                            axis=(-1, - 2))
        self._next.forward(self.data)

    def backward(self):
        self.error = self._next.cal_error()
        self._previous.backward()

    def cal_error(self):
        bottom = 1 / (self._filter_width * self._filter_length)
        formal_error = np.zeros(self._previous_shape)
        for i in np.arange(self.error.shape[-2]):
            s_i = i * self._stride
            for j in np.arange(self.error.shape[-1]):
                s_j = j * self._stride
                formal_error[:, s_i: s_i + self._filter_length, s_j: s_j + self._filter_width] \
                    = formal_error[:, s_i: s_i + self._filter_length, s_j: s_j + self._filter_width] \
                    + self.error[:, i, j] * bottom
        return formal_error

    def predict(self, data):
        self.data = np.empty((data.shape[0], self._channel,
                              self._data_length, self._data_width))
        for i in np.arange(self.data.shape[-2]):
            s_i = i * self._stride
            for j in np.arange(self.data.shape[-1]):
                s_j = j * self._stride
                self.data[..., i, j] = \
                    np.mean(data[..., s_i: s_i + self._filter_length, s_j: s_j + self._filter_width],
                            axis=(-1, - 2))
        self._next.predict(self.data)
        # reinitialize data
        self.data = np.empty((self._channel, self._data_length, self._data_width))

    def set_activation(self, activation):
        self._next.set_activation(activation)


class Connect(Layer):

    # connect layer will not be set in input argv of NeuralNetwork
    # connect pooling/convolution layer with hidden layer
    # since data in these layers are 3-dim matrix
    # and data in fully connected layer is a column vector
    # this layer makes such transform for data and error

    type = 'c'

    def __init__(self, previous=None):
        self.set_previous(previous)
        data = previous.data
        self.previous_shape = data.shape
        self.data = data.reshape((-1, 1))
        self.num = self.data.shape[0]
        self.error = np.empty(self.data.shape)

    def forward(self, data):
        # reshape data from a (channel, length, width) matrix
        # into a column vector
        self.data = data.reshape((-1, 1))
        self._next.forward(self.data)

    def backward(self, error):
        self.error = error
        self._previous.backward()

    def cal_error(self):
        # recover data from a row vector
        # into a (channel, length, width) matrix
        return self.error.reshape(self.previous_shape)

    def predict(self, data):
        data_num = data.shape[0]
        self.data = np.empty((self.num, data_num))
        for i in np.arange(data_num):
            self.data[:, i] = data[i].reshape(-1)
        self._next.predict(self.data)
        # reinitialize data
        self.data = np.empty((self.num, 1))

    def set_dropout(self, prob):
        return

    def set_activation(self, activation):
        self._next.set_activation(activation)

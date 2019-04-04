import numpy as np
import activation.LReLU as LReLU
import activation.softmax as softmax


class Layer:
    _previous = None
    _next = None
    rate = 0

    def get_next(self):
        return self._next

    def get_previous(self):
        return self._previous

    def set_next(self, obj):
        self._next = obj

    def set_previous(self, obj):
        self._previous = obj


class Input(Layer):

    type = 'i'

    def __init__(self, para):
        if isinstance(para, int):
            self.data = np.empty((para, 1), dtype=float)
            self.num = para
        else:
            self.data = np.empty(para)
            self.num = para[0]

    def forward(self):
        self._next.forward(self.data)

    def backward(self, error=0):
        return

    def predict(self):
        self._next.predict(self.data)


class Hidden(Layer):

    type = 'h'

    def __init__(self, num, previous=None):
        self.set_previous(previous)
        self.data = np.empty((num, 1), dtype=float)
        self.b = np.zeros((num, 1))
        self.w = np.random.uniform(low=-0.01, high=0.01,
                                   size=(num, previous.num))
        self.error = np.empty((1, num), dtype=float)
        self.num = num
        self._hidden_output = LReLU.get_output
        self._hidden_error = LReLU.hidden_error

    def forward(self, data):
        # compute data output of this layer without dropout
        data = np.dot(self.w, data) + self.b
        self.data = self._hidden_output(data)
        self._next.forward(self.data)

    def backward(self, error):
        # compute error and update w, b of this layer without dropout
        self.error = self._hidden_error(self.data.T, error)
        self._previous.backward(np.dot(self.error, self.w))
        error_t = self.error.T
        gradient = np.dot(error_t, self._previous.data.T)
        self.b = self.b - self.rate * error_t
        self.w = self.w - self.rate * gradient

    def predict(self, data):
        data = np.dot(self.w, data) + self.b
        self.data = self._hidden_output(data)
        self._next.predict(self.data)

    def set_activation(self, activation):
        self._hidden_output = activation.get_output
        self._hidden_error = activation.hidden_error
        self._next.set_activation(activation)


class Output(Layer):

    type = 'o'
    label = 0

    def __init__(self, num, previous=None):
        self.set_previous(previous)
        self.data = np.empty((num, 1), dtype=float)
        self.b = np.zeros((num, 1))
        self.w = np.random.uniform(low=-0.01, high=0.01,
                                   size=(num, previous.num))
        self.error = np.empty((1, num), dtype=float)
        self.num = num
        self._final_output = softmax.get_output
        self._output_error = softmax.output_error

    def forward(self, data):
        data = np.dot(self.w, data) + self.b
        self.data = self._final_output(data)

    def backward(self):
        self.error = self._output_error(self.data, self.label)
        self._previous.backward(np.dot(self.error, self.w))
        error_t = self.error.T
        gradient = np.dot(error_t, self._previous.data.T)
        self.b = self.b - self.rate * error_t
        self.w = self.w - self.rate * gradient

    def predict(self, data):
        # compute data output of this layer
        data = np.dot(self.w, data) + self.b
        self.data = self._final_output(data,)

    def output_error(self):
        return self._output_error(self.data, self.label)

    def set_activation(self, activation):
        return

    def set_output(self, output):
        self._final_output = output.get_output
        self._output_error = output.output_error


class Dropout(Layer):

    type = 'd'

    def __init__(self, prob, previous=None):
        self.set_previous(previous)
        self.prob = 1 - prob
        previous_shape = previous.data.shape
        if previous_shape[-1] == 1:
            # dropout in hidden layer
            self.num = previous.num
            self._back = lambda x: x.T
        else:
            # dropout in convolution layer
            self._back = lambda x: x
        self.data = np.empty(previous_shape, dtype=float)
        self.drop = np.random.binomial(n=1, p=self.prob, size=previous_shape)

    def forward(self, data):
        # compute data output of this layer with dropout
        # initialize dropout units
        self.drop = np.random.binomial(n=1, p=self.prob, size=data.shape)
        # dropout
        self.data = data * self.drop / self.prob
        self._next.forward(self.data)

    def backward(self, error):
        # compute error and update w, b of this layer with dropout
        # multiply with dropout prob the keep gradient correct
        self._previous.backward(error * self._back(self.drop))

    def predict(self, data):
        self.data = data
        self._next.predict(self.data)

    def set_activation(self, activation):
        self._next.set_activation(activation)


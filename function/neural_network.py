from layer.cnn_layer import *
import importlib
import time


class NeuralNetwork:

    result = []
    correct = 0
    incorrect = 0

    def __init__(self, model):
        self._cross_entropy = softmax.cross_entropy
        self._get_predict = softmax.get_predict
        self._label_process = softmax.label_process
        self.lay_num = len(model)-1
        self.model = model
        self.activation_func = 'LReLU'
        self.output_func = 'tanh'

        # initialize input layer
        layer = model[0]
        para = layer[1]
        self.input = Input(para)
        curr = self.input

        # initialize hidden layer
        for layer in model[1:-1]:
            if layer[0] == 'hidden':
                # initialize connect layer
                if curr.type == 'x' or curr.type == 'm' or curr.type == 'v':
                    curr.set_next(Connect(curr))
                    curr = curr.get_next()
                hidden = layer[1]
                curr.set_next(Hidden(hidden, curr))
                curr = curr.get_next()
            elif layer[0] == 'dropout':
                # initialize connect layer
                if curr.type == 'x' or curr.type == 'm' or curr.type == 'v':
                    curr.set_next(Connect(curr))
                    curr = curr.get_next()
                para = layer[1]
                curr.set_next(Dropout(para, curr))
                curr = curr.get_next()
            elif layer[0] == 'convolution':
                para = layer[1]
                curr.set_next(Convolution(para, curr))
                curr = curr.get_next()
            elif layer[0] == 'maxpooling':
                para = layer[1]
                curr.set_next(MaxPooling(para, curr))
                curr = curr.get_next()
            elif layer[0] == 'meanpooling':
                para = layer[1]
                curr.set_next(MeanPooling(para, curr))
                curr = curr.get_next()

        # initialize output layer
        layer = model[-1]
        outp = layer[1]
        self.output = Output(outp, curr)
        curr.set_next(self.output)

    def method(self, activation=None, output=None):
        # set method
        if activation:
            self.activation_func = activation
            # import method
            activation = importlib.import_module('activation.'+activation)
            self.input.get_next().set_activation(activation)
        if output:
            self.output_func = output
            output = importlib.import_module('activation.'+output)
            self.output.set_output(output)
            self._cross_entropy = output.cross_entropy
            self._get_predict = output.get_predict
            self._label_process = output.label_process

    def _loss(self, result, label, relabel):
        # compute loss and correct/incorrect during this epoch
        loss = self._cross_entropy(result, label).sum()
        predict = self._get_predict(result)
        correct = np.sum(predict == relabel)
        incorrect = relabel.shape[0] - correct
        return loss, correct, incorrect

    def nn_train(self, data, label, rate, epoch):
        train_error = []
        num = data.shape[1]
        # aim to convert one hot encoded label back
        relabel = self._label_process(label)
        # set learning rate
        Layer.rate = rate
        for i in range(epoch):
            result = []
            # stochastic gradient descent
            for j in range(num):
                # set label
                self.output.label = label[j, :]
                # calculate the value from input to output layer
                # every output is a vector, axis=0
                self.input.data = data[:, j][:, None]
                self.input.forward()
                # store loss and accuracy
                result.append(self.output.data[:, 0])
                # update weight
                self.output.backward()
            result = np.array(result)
            train_error.append(self._loss(result, label, relabel))
            print(i + 1, train_error[-1])
        return train_error

    def cnn_train(self, data, label, rate, epoch):
        train_error = []
        num = data.shape[0]
        # aim to convert one hot encoded label back
        relabel = self._label_process(label)
        # set learning rate
        Layer.rate = rate
        for i in range(epoch):
            time1 = time.time()
            result = []
            # stochastic gradient descent
            for j in range(num):
                # set label
                self.output.label = label[j, :]
                # calculate the value from input to output layer
                # every output is a vector, axis=0
                self.input.data = data[j, :]
                self.input.forward()
                # store loss and accuracy
                result.append(self.output.data[:, 0])
                # update weight
                self.output.backward()
            result = np.array(result)
            train_error.append(self._loss(result, label, relabel))
            print(i + 1, train_error[-1])
            if train_error[-1][1] == num:
                break
            print(time.time() - time1)
        return train_error

    def predict(self, data, label):
        if len(data.shape) == 2:
            data_num = data.shape[1]
        else:
            data_num = data.shape[0]
        relabel = self._label_process(label)
        self.input.data = data
        self.input.predict()
        # make prediction
        output = self.output.data.T
        predict = self._get_predict(output)
        # count correctly and incorrectly classified
        self.correct = np.sum(predict == relabel)
        self.incorrect = data_num - self.correct
        output = np.max(output, axis=1)
        self.result = np.c_[output, predict, relabel]
        return self.result, self.correct, self.incorrect

    def f1_score(self):
        # number of true positive
        tp = np.sum(self.result[:, 1] + self.result[:, 2] == 2)
        # number of  predicted positive
        predict_positive = np.sum(self.result[:, 1])
        # number of  actual positive
        actual_positive = np.sum(self.result[:, 2])
        if predict_positive == 0 or actual_positive == 0:
            return 0
        else:
            precision = tp / predict_positive
            recall = tp / actual_positive
            return 2 * precision * recall / (precision + recall)


def store(net, path):
    model = [net.model,
             [net.activation_func, net.output_func]]
    curr = net.input
    while curr:
        if curr.type == 'v':
            model.append([curr.filter, curr.bias])
        elif curr.type == 'h' or curr.type == 'o':
            model.append([curr.w, curr.b])
        curr = curr.get_next()
    np.save(path, model)


def load(path):
    model = np.load(path)
    net = NeuralNetwork(model[0])
    net.method(activation=model[1][0], output=model[1][1])
    curr = net.input
    index = 2
    while curr:
        if curr.type == 'v':
            curr.filter = model[index][0]
            curr.bias = model[index][1]
            index += 1
        elif curr.type == 'h' or curr.type == 'o':
            curr.w = model[index][0]
            curr.b = model[index][1]
            index += 1
        curr = curr.get_next()
    return net



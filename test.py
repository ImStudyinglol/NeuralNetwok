import function.neural_network as nn
import numpy as np
import importlib
import minst.open as minst
import time


if __name__ == "__main__":

    Act = 'LReLU'
    Out = 'softmax'

    TrainI = minst.readMINIST('minst/train-images.idx3-ubyte')
    TrainImage = TrainI.getImage()
    TrainImage = minst.image(TrainImage)
    TrainL = minst.readMINIST('minst/train-labels.idx1-ubyte')
    TrainLabel = TrainL.getLabel()
    TrainLabel = minst.hot_encoding(TrainLabel)

    TestI = minst.readMINIST('minst/t10k-images.idx3-ubyte')
    TestImage = TestI.getImage()
    TestImage = minst.image(TestImage)
    TestL = minst.readMINIST('minst/t10k-labels.idx1-ubyte')
    TestLabel = TestL.getLabel()
    TestLabel = minst.hot_encoding(TestLabel)

    FeatNum = TrainImage.shape[1:]
    OutNum = 10

    Model = [['input', FeatNum],
             ['convolution', [(5, 5), 8, 2, 1]],
             ['maxpooling', [(2, 2), 2]],
             ['hidden', 120],
             ['hidden', 84],
             ['output', OutNum]]

    DeepNet = nn.NeuralNetwork(Model)
    DeepNet.method(activation=Act, output=Out)
    TrainError = DeepNet.cnn_train(TrainImage[0:1000],
                                   TrainLabel[0:1000], 0.0005, 1)
    Result = DeepNet.predict(TestImage[0:1000], TestLabel[0:1000])
    print(Result[1], Result[2])

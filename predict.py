import function.neural_network as nn
import numpy as np
import importlib
import minst.open as minst
import time


if __name__ == "__main__":

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

    '''
    Model = [['input', FeatNum],
             ['convolution', [(5, 5), 8, 2, 1]],
             ['maxpooling', [(2, 2), 2]],
             ['convolution', [(5, 5), 16, 0, 1]],
             ['maxpooling', [(2, 2), 2]],
             ['dropout', 0.2],
             ['hidden', 120],
             ['dropout', 0.2],
             ['hidden', 84],
             ['output', OutNum]]
    '''

    DeepNet, model = nn.load('model/cnn_triple_dropout_0.3_shuffle.npy')

    Result = DeepNet.predict(TestImage, TestLabel)
    print(Result[1], Result[2])

    '''
    Result = DeepNet.predict(TrainImage, TrainLabel)
    print(Result[1], Result[2])
    '''


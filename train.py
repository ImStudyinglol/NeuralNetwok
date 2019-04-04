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
             ['convolution', [(5, 5), 16, 0, 1]],
             ['maxpooling', [(2, 2), 2]],
             ['dropout', 0.4],
             ['hidden', 120],
             ['dropout', 0.4],
             ['hidden', 84],
             ['output', OutNum]]

    DeepNet = nn.NeuralNetwork(Model)
    DeepNet.method(activation=Act, output=Out)

    try:
        TrainError = DeepNet.cnn_train(TrainImage, TrainLabel, 0.0005, 5)
        Result = DeepNet.predict(TestImage, TestLabel)
        print(Result[1], Result[2])
        nn.store(DeepNet, 'cnn_dropout_0.4.npy')

        TrainError = DeepNet.cnn_train(TrainImage, TrainLabel, 0.0001, 5)
        TrainError = DeepNet.cnn_train(TrainImage, TrainLabel, 0.00005, 5)
        Result = DeepNet.predict(TestImage, TestLabel)
        print(Result[1], Result[2])
        nn.store(DeepNet, 'cnn_dropout_0.4.npy')

        Para = input('leanring rate and epochs\n').split()
        Rate = float(Para[0])
        Epoch = int(Para[1])

        while 1:

            TrainError = DeepNet.cnn_train(TrainImage, TrainLabel, Rate, Epoch)
            time1 = time.time()
            Result = DeepNet.predict(TestImage, TestLabel)
            time2 = time.time()
            print(Result[1], Result[2])
            print(Rate, Epoch)
            print(Model)
            print(time2 - time1)

            Argv = input('want to end?\n')
            if Argv == '1':
                nn.store(DeepNet, 'cnn_dropout_0.4.npy')
                break
            else:
                Para = Argv.split()
                Rate = float(Para[0])
                Epoch = int(Para[1])

    except KeyboardInterrupt:
        nn.store(DeepNet, 'cnn_dropout_0.4.npy')
        Result = DeepNet.predict(TestImage, TestLabel)
        print(Result[1], Result[2])

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

    Path = 'model/cnn_double_dropout_0.2_shuffle.npy'
    DeepNet = nn.load(Path)
    DeepNet.method(shuffle=False)

    try:

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
            print(time2 - time1)

            nn.save(DeepNet, Path)

            Argv = input('want to end?\n')
            if Argv == '1':
                break
            else:
                Para = Argv.split()
                Rate = float(Para[0])
                Epoch = int(Para[1])

    except KeyboardInterrupt:
        nn.save(DeepNet, Path)
        Result = DeepNet.predict(TestImage, TestLabel)
        print(Result[1], Result[2])

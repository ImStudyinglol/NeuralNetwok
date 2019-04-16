import function.neural_network as nn
import minst.open as minst


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

    # train data is a (60000, 1, 28, 28) matrix
    # train label is a (60000, 10) matrix
    FeatNum = TrainImage.shape[1:]
    OutNum = 10
    Dropout = 0.2

    Model = [['input', FeatNum],
             ['convolution', [(5, 5), 8, 2, 1]],
             ['maxpooling', [(2, 2), 2]],
             ['convolution', [(5, 5), 16, 0, 1]],
             ['maxpooling', [(2, 2), 2]],
             ['dropout', Dropout],
             ['hidden', 120],
             ['dropout', Dropout],
             ['hidden', 84],
             ['output', OutNum]]

    DeepNet = nn.NeuralNetwork(Model)
    DeepNet.method(activation=Act, output=Out, shuffle=True)
    Path = 'cnn.npy'

    print(Model)
    print(Path)

    try:
        # train with first 100 instances
        TrainError = DeepNet.train(TrainImage[0:100], TrainLabel[0:100], 0.0005, 5)
        Result = DeepNet.predict(TestImage[0:100], TestLabel[0:100])
        print(Result[1], Result[2])
        nn.save(DeepNet, Path)

        '''
        TrainError = DeepNet.train(TrainImage, TrainLabel, 0.0002, 5)
        Result = DeepNet.predict(TestImage, TestLabel)
        print(Result[1], Result[2])
        nn.save(DeepNet, Path)

        TrainError = DeepNet.train(TrainImage, TrainLabel, 0.00005, 5)
        Result = DeepNet.predict(TestImage, TestLabel)
        print(Result[1], Result[2])
        nn.save(DeepNet, Path)
        '''

    except KeyboardInterrupt:
        nn.save(DeepNet, Path)
        Result = DeepNet.predict(TestImage, TestLabel)
        print(Result[1], Result[2])

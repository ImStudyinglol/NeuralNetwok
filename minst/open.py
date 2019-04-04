# -*- coding:utf-8
import numpy as np
import struct
import matplotlib.pyplot as plt
import os

class readMINIST(object):
    """MNIST数据集加载
    输出格式为：numpy.array()

    使用方法如下
    from readMINIST import readMINIST
    def main():
        trainfile_X = '../dataset/MNIST/train-images.idx3-ubyte'
        trainfile_y = '../dataset/MNIST/train-labels.idx1-ubyte'
        testfile_X = '../dataset/MNIST/t10k-images.idx3-ubyte'
        testfile_y = '../dataset/MNIST/t10k-labels.idx1-ubyte'

        train_X = DataUtils(filename=trainfile_X).getImage()
        train_y = DataUtils(filename=trainfile_y).getLabel()
        test_X = DataUtils(testfile_X).getImage()
        test_y = DataUtils(testfile_y).getLabel()

        #以下内容是将图像保存到本地文件中
        #path_trainset = "../dataset/MNIST/imgs_train"
        #path_testset = "../dataset/MNIST/imgs_test"
        #if not os.path.exists(path_trainset):
        #    os.mkdir(path_trainset)
        #if not os.path.exists(path_testset):
        #    os.mkdir(path_testset)
        #DataUtils(outpath=path_trainset).outImg(train_X, train_y)
        #DataUtils(outpath=path_testset).outImg(test_X, test_y)

        return train_X, train_y, test_X, test_y
    """

    def __init__(self, filename=None, outpath=None):
        self._filename = filename
        self._outpath = outpath

        self._tag = '>'
        self._twoBytes = 'II'
        self._fourBytes = 'IIII'
        self._pictureBytes = '784B'
        self._labelByte = '1B'
        self._twoBytes2 = self._tag + self._twoBytes
        self._fourBytes2 = self._tag + self._fourBytes
        self._pictureBytes2 = self._tag + self._pictureBytes
        self._labelByte2 = self._tag + self._labelByte
    '''
__init__是readMINST的构造函数，filename是输入文件名，outpath是用于输出图像存储时指定存储路径。定义的几个字符串是为了后面使用struct类的时候更清晰明了。字符串的意义就是表面意思。
'''
    def getImage(self):
        """
        将MNIST的二进制文件转换成像素特征数据
        """
        binfile = open(self._filename, 'rb') #以二进制方式打开文件
        buf = binfile.read()
        binfile.close()
        index = 0
        numMagic,numImgs,numRows,numCols=struct.unpack_from(self._fourBytes2,buf,index)#读4个byte
        index += struct.calcsize(self._fourBytes)#后移4个byte
        images = []
        for i in range(numImgs):
            imgVal = struct.unpack_from(self._pictureBytes2, buf, index)
            index += struct.calcsize(self._pictureBytes2)
            imgVal = list(imgVal)
            '''
            for j in range(len(imgVal)):
                if imgVal[j] > 1:
                    imgVal[j] = 1
'''
            images.append(imgVal)

        return np.array(images)#返回numpy中支持的array型，便于之后直接调用分类函数。

    def getLabel(self):
        """
        将MNIST中label二进制文件转换成对应的label数字特征
        """
        binFile = open(self._filename,'rb')
        buf = binFile.read()
        binFile.close()
        index = 0
        magic, numItems= struct.unpack_from(self._twoBytes2, buf,index)
        index += struct.calcsize(self._twoBytes2)
        labels = [];
        for x in range(numItems):
            im = struct.unpack_from(self._labelByte2,buf,index)
            index += struct.calcsize(self._labelByte2)
            labels.append(im[0])
        return np.array(labels)

    def outImg(self, arrX, arrY):
        """
        根据生成的特征和数字标号，输出png的图像
        """
        m, n = np.shape(arrX)
        #每张图是28*28=784Byte
        for i in range(1):
            img = np.array(arrX[i])
            img = img.reshape(28,28)
            outfile = str(i) + "_" +  str(arrY[i]) + ".png"
            plt.figure()
            plt.imshow(img, cmap = 'binary') #将图像黑白显示
            plt.savefig(self._outpath + "" + outfile)


def hot_encoding(label):
    # initialize one hot encoding matrix
    hot = np.zeros((label.shape[0], label.max() + 1))
    for j in range(label.shape[0]):
        hot[j, label[j]] = 1
    return hot


def image(data):
    num = data.shape[0]
    output = np.empty((num, 1, 28, 28))
    for i in range(num):
        output[i] = data[i].reshape((1, 28, 28))
    return output.astype(int)


if __name__ == "__main__":
    TrainImage = readMINIST('minst/train-images.idx3-ubyte', '')
    image = TrainImage.getImage()
    TrainLabel = readMINIST('minst/train-labels.idx1-ubyte')
    label = TrainLabel.getLabel()
    TrainImage.outImg(image, label)

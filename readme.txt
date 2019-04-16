A neural network implement NN and CNN based on numpy.

F1 score for binary classification included.
Save, load method included.
Dropout layer included, can be applied after any layer.

Required data processing:
It's always better to standardize your numeric features.
For NN, convert your dataset into a (num_data, 1, image_length, image_width) matrix.
For NN and for binary classification, encode your labels into 0/1. For multi-class classification, use one-hot encoding for your labels.

Implementation:
Initialize model: your_net = function.neural_network.NeuralNetwork(your model)
train: your_net.train(traindata, trainlabel, learning_rate, epochs)
	return a list of training loss and # of correct and incorrect predictions in training epochs
predict: your_net.predict(testdata, testlabel)
	retrun a list of predicting probs, predicted label and actual label
F1 score: your_net.f1_score() (For binary classification only)
save: function.neural_network.save(your_net, path)
load: your_net = function.neural_network.load(your_net)

Method Setting:
Activation function for convolution and hidden layer:
	activation = 'linear', 'ReLU', 'LReLU', 'sigmoid', 'tanh'
Activation function for output layer:
	output = 'linear', 'sigmoid', 'softmax'
Random data shuffle in every train epoch:
	shuffle = True, False 

Layer setting for CNN:
Input = ['input', (1, image_length, image_width)]
Convoution = ['convolution', [(filter_length, filter_width), filter_num, padding, stride]]
Maxpooling = ['maxpooling', [(filter_length, filter_width), stride]]
Meanpooling = ['meanpooling', [(filter_length, filter_width), stride]]
Dropout = ['dropout', drop_prob(usually 0 ~ 0.5)]
Hidden = ['hidden', hidden_unit_num]
Output = ['output', output_unit_num]]

Layer setting for NN:
Input = ['input', input_unit_num]
Dropout = ['dropout', drop_prob(usually 0 ~ 0.5)]
Hidden = ['hidden', hidden_unit_num]
Output = ['output', output_unit_num]]

Shortcomings:
1. linear output uses squared loss only, sigmoid and softmax use cross entropy only
2. only SGD for optimization method, mini-batch SGD to be updated
3. L2, L1 penalty regularization methods to be updated
4. momentum, etc. optimization methods to be updated
5. must specify trainning epoch, or training will run until all training instances are correctly predicted
6. corss validation to be updated

Model example:

Model = [['input', (1, 28, 28)],
         ['convolution', [(5, 5), 8, 2, 1]],
         ['maxpooling', [(2, 2), 2]],
         ['convolution', [(5, 5), 16, 0, 1]],
         ['maxpooling', [(2, 2), 2]],
         ['dropout', 0.2],
         ['hidden', 120],
         ['dropout', 0.2],
         ['hidden', 84],
         ['dropout', 0.2],
         ['output', 10]]
DeepNet = nn.NeuralNetwork(Model)
DeepNet.method(activation='LReLU', output='softmax', shuffle=True)
TrainError = DeepNet.train(TrainImage, TrainLabel, 0.0005, 5)
Result = DeepNet.predict(TestImage, TestLabel)

A training epoch throughout minst data (60000 instances) for cnn takes about 1000s.
Prediction throughout minst data (10000 instances) for cnn takes about 60s.

Current best model: model/cnn_triple_dropout_0.3_shuffle.npy
Accuracy on minst: 99.37%

If you have any questions, please contact lshang6@wisc.edu.

An neural network only based on numpy.

Implement general NN, CNN with any classical net structure.

Save, load method included.

Dropout layer included.

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
              ['output', 10]]

For detailed information, please contact: lshang6@wisc.edu


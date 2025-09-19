# yt playlist: https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3

import numpy as np

# working with three neurons

# inputs = [1, 2, 3, 2.5]
# weights = [[0.2, 0.8, -0.5, 1.0],
#             [0.5, -0.91, 0.26, -0.5],
#             [-0.26, -0.27, 0.17, 0.87],
#         ]
# # bias of three neurons
# biases = [2, 3, 0.5]

# '''
# Just a list [2, 3, 0.5] is equivalent to a vector
# List of list [[],[]] whereas, is equivalent to a matrix of vectors
# '''

# # output of three neurons
# output = np.dot(weights, inputs) + biases
# print(output)


'''
Creating Class
'''
np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, X):
        self.output = np.dot(X, self.weights) + self.biases
        
layer1 = Layer_Dense(4,5)
layer1.forward(X)

layer2 = Layer_Dense(5,2)
layer2.forward(layer1.output)
print(layer2.output)
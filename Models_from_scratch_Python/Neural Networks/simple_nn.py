# yt playlist: https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3

import numpy as np
from create_dataset import spiral_data
import math

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

X, y = spiral_data(100, 3) # features, labels

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, X):
        self.output = np.dot(X, self.weights) + self.biases
        
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)  
        
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True)) # axis 1 denotes row wise, & keepdimes True keeps the default dim      
        probabilities = exp_values / np.sum(exp_values, axis= 1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(y, output)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_true, y_pred):
        samples = len(y_true)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis = 1)
            
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

X, y = spiral_data(points = 100, classes=3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss_func = Loss_CategoricalCrossentropy()
loss = loss_func.calculate(activation2.output, y)
print(f'Loss: {loss}')


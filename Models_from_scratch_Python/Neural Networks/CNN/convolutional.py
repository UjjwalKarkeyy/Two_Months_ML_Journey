import numpy as np
from scipy import signal

class Convolutional:
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width  = input_shape
        self.depth = depth # number of kernels
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height-kernel_size + 1, input_width-kernel_size+1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)
        
    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth): # goes through output depth
            for j in range(self.input_depth): # through input depth
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i,j], 'valid')
        return self.output 
    
    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i,j] = signal.correlate2d(self.input[j], output_gradient[i], 'valid')
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i,j], 'full')
                
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient
    
class Reshape:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        
    def forward(self, input):
        return np.reshape(input, self.output_shape)
    
    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)

class Activation:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, lr):
        return np.multiply(output_gradient, self.activation_prime(self.input))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1-y_true) / (1-y_pred) - y_true / y_pred) / np.size(y_true)
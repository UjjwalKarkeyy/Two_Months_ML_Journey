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

# dense layer
class Layer_Dense:
    
    # layer initialization
    def __init__(self, n_inputs, n_neurons):
        # initializing weights and biases
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    # forward pass
    def forward(self, inputs):
        # remembering input values
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    # backward pass
    def backward(self, dvalues):
        # gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims=True)
        # gradients on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        
# ReLU activation class        
class Activation_ReLU:
    # forward pass
    def forward(self, inputs):
        # remember inputs
        self.inputs = inputs
        # calculating output values from inputs
        self.output = np.maximum(0, inputs)  
        
    # backward pass
    def backward(self, dvalues):
        # since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()
        # zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

# softmax activation class        
class Activation_Softmax:
    # forward pass
    def forward(self, inputs):
        # remembering the inputs
        self.inputs = inputs
        # get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True)) # axis 1 denotes row wise, & keepdimes True keeps the default dim      
        # normalizing prob for each sample
        probabilities = exp_values / np.sum(exp_values, axis= 1, keepdims=True)
        self.output = probabilities

    # backward pass
    def backward(self, dvalues):
        # creating uninitialized array
        self.dinputs = np.empty_like(dvalues)
        
        # enumerating outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # flatten output array
            single_output = single_output.reshape(-1,1)
            # calculating Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
            
# common loss class
class Loss:
    # calcuate the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):
        # loss calculation
        sample_losses = self.forward(y, output)
        # mean loss
        data_loss = np.mean(sample_losses)
        return data_loss

# cross-entropy loss class
class Loss_CategoricalCrossentropy(Loss):
    # forward pass
    def forward(self, y_true, y_pred):
        # number of samples in a batch
        samples = len(y_pred)
        
        # clip data to prevent division by 0
        # clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # probabilites for target values
        # only if categorial labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis = 1)
        
        # loss likelihoods
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # backward pass
    def backward(self, dvalues, y_true):
        # number of samples
        samples = len(dvalues)
        # number of labels in every sample
        # we'll use the first sample to count them
        labels = len(dvalues[0])
        
        # if labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1: # [0,1,2]
            y_true = np.eye(labels)[y_true] # eye gives us diagonal of 1s and rest 0s of size 'labels'

        # calculate the gradient
        self.dinputs = -y_true/dvalues
        # normalize the gradients
        self.dinputs = self.dinputs / samples
        
# softmax classifier - combined softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():
    # create activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
        
    # forward pass
    def forward(self, inputs, y_true):
        # output layer's activation function
        self.activation.forward(inputs)
        # set the output
        self.output = self.activation.output
        # calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    
    # backward pass
    def backward(self, dvalues, y_true):
        # number of samples
        samples = len(dvalues)
        
        # if labels are one-hot encoded,
        # turn the into discrete values
        if len(y_true.shape) == 2:
            '''
            [[1,0,0],
             [0,1,0],
             [0,0,1]]
             
            argmax: Finds the index of maximum value, axis = 1 means in the row
            so the result is: [0, 1, 2]
            '''
            y_true = np.argmax(y_true, axis = 1)
            
        # copy so we can modify safely
        self.dinputs = dvalues.copy()
        # calculate gradients
        self.dinputs[range(samples), y_true] -= 1 # y(hat) - y_true
        # normalize gradients
        self.dinputs = self.dinputs / samples

class Optimizer_SGD:
    # initializing optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate = 1., decay = 0, momentum = 0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        
    # call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # update parameters
    def update_params(self, layer):
        # using momentum
        if self.momentum:
            # if layer doesn't contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weights_momentums = np.zeros_like(layer.weights)
                # if there is no momentum array for weights
                # it also doesn't have it for biases
                layer.bias_momentums = np.zeros_like(layer.biases)
                
            # build weights updates with momentus - take previous
            # updates multiplied by retian factor and update with
            # current gradients
            weight_updates = (self.momentum * layer.weights_momentums) - (self.current_learning_rate * layer.dweights)
            layer.weights_momentums = weight_updates
            
            # build bias updates
            bias_updates = (self.momentum * layer.bias_momentums) - (self.current_learning_rate * layer.dbiases)
            layer.bias_momentums = bias_updates
            
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
            
        # update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates
        
    # call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

# RMSprop: Root mean squared propagation
class Optimizer_RMSprop:
    # initialize optimizer - set settings
    def __init__(self, learning_rate = 0.001, decay = 0., epsilon = 1e-7, rho = 0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
        
    # calling once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # update params
    def update_params(self, layer):
        # if layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # update cache with squared current gradients
        layer.weight_cache = (self.rho * layer.weight_cache) + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = (self.rho * layer.bias_cache) + (1 - self.rho) * layer.dbiases**2

        # vanilla SGD param update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    # call once after any param updates
    def post_update_params(self):
        self.iterations += 1

# Adagrad: Adaptive gradient
class Optimizer_Adagrad:
    # initialize optimizer - set settings
    def __init__(self, learning_rate = 1., decay = 0., epsilon = 1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        
    # calling once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # update params
    def update_params(self, layer):
        # if layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # vanilla cache with squared current gradients
        # with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    # call once after any param updates
    def post_update_params(self):
        self.iterations += 1

# create dataset
X, y = spiral_data(points = 100, classes=3)

# create dense layer with 2 input features and 3 output features
dense1 = Layer_Dense(2, 64)

# create ReLU activation (to be used with Dense Layer)
activation1 = Activation_ReLU()

# create second dense layer with 3 input features (as we take output of the previous layer here)
# 3 output values
dense2 = Layer_Dense(64,3)

# create softmax activation combined with loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# create optimizer
optimizer = Optimizer_RMSprop(learning_rate=0.002, decay = 1e-5, rho=0.999)

# Train in loop
for epoch in range(10001):

    # perform forward pass for first layer
    dense1.forward(X)

    # perform a forward pass through activation ReLU function
    # takes the output of the first dense layer
    activation1.forward(dense1.output)

    # perform the forward pass on the seond layer
    dense2.forward(activation1.output)

    # perform forward pass through the activation/loss function
    loss = loss_activation.forward(dense2.output, y)

    # calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis = 1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis = 1)

    '''
    example for one sample:
    predictions = [0.6, 0.3, 0.1]
    y_true = [1, 0, 0]

    what does it mean to do predictions == y?
    [0.6, 0.3, 0.1] ?== [1, 0, 0]
    [False, False, False]
    [0, 0, 0]
    mean = 0 + 0 + 0 / 3 => 0
    '''
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, '+
              f'acc: {accuracy: .3f}, '+
              f'loss: {loss:.3f}, ' +
              f'lr: {optimizer.current_learning_rate}')

    # backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
# equation of the hyperplane
'''
y = wx - b
'''

# Gradient Descent
'''
Gradient descent is an optimization algorithm used for minimizing the loss function in various machine learning algorithms.
It is used for updating the parameters of the learning model.
'''

# Learning Rate
'''
Learning rate is a tuning parameter in an optimization algorithm that determines the step size at each iteration while moving
toward the global minima or say toward a minimum of a loss function
'''

import numpy as np

class SupportVectorClassifier():
    
    # constructor 
    def __init__(self, lr, no_of_iters, lambda_param):
        self.lr = lr
        self.no_of_iters = no_of_iters
        self.lambda_param = lambda_param
        
    # fitting function of the model
    def fit(self, X, y):
        self.num_samples, self.num_feat = X.shape
        
        #initializing the value of initial weight values and bias
        self.w = np.zeros(self.num_feat)
        self.b = 0
        self.X = X
        self.y = y
        
        # implementing gradient descent for optimization
        for i in range(self.no_of_iters):
            self.update_weights()
            
    def update_weights(self):
        # label encoding
        y_label = np.where(self.y <= 0, -1, 1)
        
        # gradients (dw, db)
        for index, x_i in enumerate(self.X):
            condition = y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1
            
            if(condition == True):
                dw = 2 * self.lambda_param * self.w
                db = 0

            else:
                dw = 2 * self.lambda_param * self.w - np.dot(x_i, y_label[index])
                db = y_label[index]
                
            self.w = self.w - (self.lr * dw)
            self.b = self.b - (self.lr * db)
    
    # function to make prediction
    def predict(self, X):
        output = np.dot(X, self.w) - self.b
        
        # takes the output i.e., say [2.5, -1.2, 0.0, 3.8] and 'sign' turns them into [1,-1,0,1]
        predicted_labels = np.sign(output)
        y_hat = np.where(predicted_labels <= -1, 0, 1)
        return y_hat            
            
        
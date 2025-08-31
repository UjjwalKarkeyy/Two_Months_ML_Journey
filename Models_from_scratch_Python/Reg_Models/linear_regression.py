import numpy as np

class LinearRegression():
    
    # constructor
    def __init__(self, lr = 0.01, n_iters = 1000):
        
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    # function for training the model
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features) # creating a zero matrix and setting weight as zero for all features
        self.bias = 0
        
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            
            dw = (1/n_samples) * 2 * np.dot(X.T, (y_pred - y)) # we want to decrease the error per feature so we include transpose
            db = (1/n_samples) * 2 * np.sum(y_pred - y)
            
            self.weights = self.weights - (self.lr * dw)
            self.bias = self.bias - (self.lr * db)
            
    # function to predict on new data
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
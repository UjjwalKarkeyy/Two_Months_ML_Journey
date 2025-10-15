import numpy as np
from sklearn.tree import DecisionTreeRegressor

class gradientBoostingReg():
    def __init__(self, lr=0.001, n_estimators = 25, max_depth = 3):
        self.lr = lr
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
        self.initial_pred = None
        
    def fit(self, X:np.ndarray, y:np.ndarray):
        # initializing base model with mean as prediction
        self.initial_pred = np.mean(y)
        # creates a new array like the one passed i.e., 'y' with 'self.initial_pred' as its values
        current_pred = np.full_like(y, self.initial_pred, dtype=float)
        
        for _ in range(self.n_estimators):
            # calculating pseudo residuals 
            # pseudo means not exact or genuine which makes sense cause our residuals are derivated
            residuals = -LossFunctions.mse_gradient(y, current_pred)
            
            # train new tree
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            # update predictions
            predictions = tree.predict(X)
            current_pred += self.lr * predictions
    
    def predict(self, X: np.ndarray): 
        # starting with initial pred
        predictions = np.full(X.shape[0], self.initial_pred, dtype=float)

        # Add predictions from each tree
        for tree in self.trees:
            predictions += self.lr * tree.predict(X)
            
        return predictions
    
class LossFunctions:
    '''
    The method belongs to the class, but does not depend on any instance (self) of that class.
    LossFunctions.mse_loss(y_true, y_pred)
    without creating an object like loss = LossFunctions().
    '''
    @staticmethod
    def mse_loss(y_true:np.ndarray, y_pred: np.ndarray):
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mse_gradient(y_true:np.ndarray, y_pred: np.ndarray):
        return -1 * (y_true - y_pred)
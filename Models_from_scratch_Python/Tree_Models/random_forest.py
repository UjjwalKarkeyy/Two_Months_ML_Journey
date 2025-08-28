from .decision_trees_reg import DecisionTreeRegressor
import numpy as np
from collections import Counter

class RandomForest():
    # constructor
    def __init__(self, n_trees = 10, max_depth = 50, min_samples_split = 2, n_features = None):
        
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
        
    # function to train the model
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeRegressor(self.min_samples_split, self.max_depth)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    # bootstrapping function
    def _bootstrap_samples(self, X, y): 
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    
    # selecting most common class among all the predictions based on each sample
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common    
        
    # function to predict on new data
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        trees_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in trees_preds])
        return predictions
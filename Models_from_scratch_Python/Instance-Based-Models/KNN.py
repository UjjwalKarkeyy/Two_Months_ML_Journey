import numpy as np
from collections import Counter

def euclidean_dist(x1, x2):
    distance = np.sqrt(np.sum(x1 - x2)**2)
    return distance

class KNN:
    # constructor
    def __init__(self, k = 3, metric = None):
        self.k = k
        
    # model data fitting function
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    # output prediction function
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
     
    # helper function for predict   
    def _predict(self, x):
        # computing distance
        distances = [euclidean_dist(x, x_train) for x_train in self.X_train]
        
        # getting the closest k indices and neighbor lables
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # doing majority vote for classification
        # for regression, we take average of the nearest neighboring values
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
        

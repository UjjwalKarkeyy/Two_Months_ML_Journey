import numpy as np

class Node():
    # constructor
    def __init__(self, feature_index = None, threshold = None, left = None, right = None, var_red = None, value = None):
        # for internal nodes
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red
        # for leaf node
        self.value = value
        
class DecisionTreeRegressor():
    # constructor
    def __init__(self, min_samples_split = 2, max_depth = 2):
        # root of the tree
        self.root = None
        
        # stopping criteria
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
    # recursively builds the trees        
    def build_tree(self, dataset, curr_depth):
        X, y = dataset[:, :-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        best_split = {}
        # splitting until the conditions are met
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            # finding the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # checking if information gain is positive
            if best_split['var_red'] > 0:
                # left sub tree
                left_subtree = self.build_tree(best_split['dataset_left'], curr_depth+1)
                # right sub tree
                right_subtree = self.build_tree(best_split['dataset_right'], curr_depth+1)
                
                return Node(best_split['feature_index'], best_split['threshold'], 
                            left_subtree, right_subtree, best_split['var_red'])
                
        # computing leaf_value
        leaf_value = self.calculate_leaf_value(y)
        return Node(value = leaf_value)
    
    # finding the best split
    def get_best_split(self, dataset, num_samples, num_features):
        
        # dictionary for best split
        best_split = {}
        max_var_red = -float('inf')
        
        # looping over the features
        for feature_index in range(num_samples):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            
            # looping over the thresholds
            for threshold in possible_thresholds:
                # getting the current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # checking if childs are not null
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:,-1], dataset_right[:,-1]
                    # computing variance reduction
                    curr_var_red = self.variance_reduction(y, left_y, right_y)
                    # updating the current split
                    if curr_var_red > max_var_red:
                        best_split['feature_index'] = feature_index
                        best_split['threshold'] = threshold
                        best_split['dataset_left'] = dataset_left
                        best_split['dataset_right'] = dataset_right
                        best_split['var_red'] = curr_var_red
                        
        return best_split
    
    # finding the left and right split
    def split(self, dataset, feature_index, threshold):
        dataset_left = dataset[[row for row in dataset if row[feature_index] <= threshold]]
        dataset_right = dataset[[row for row in dataset if row[feature_index] > threshold]]
        return dataset_left, dataset_right
    
    # function for calculating variance reduction
    def variance_reduction(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        var_red = np.var(parent) - (weight_l *np.var(l_child) + weight_r*np.var(r_child))
        return var_red
    
    # calculates the leaf node's value
    def calculate_leaf_value(self, y):
        value = np.mean(y)
        return value
    
    # to print the tree flow
    def print_tree(self, tree = None, indent = " "):
        if not tree:
            tree = self.root
            
        if tree.value is not None:
            print(tree.value)
            
        else:
            print('X_'+str(tree.feature_index), "<=", tree.threshold, '?', tree.info_gain)
            print("%sleft:" % (indent), end = "")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end = "")
            self.print_tree(tree.right, indent + indent)
        
    # to train the tree    
    def fit(self, X, y):
        dataset = np.concatenate((X,y), axis = 1)
        self.root = self.build_tree(dataset)
      
    # make predictions on the new dataset  
    def make_predictions(self, X, tree):
        
        if tree.value != None: return tree.value
        feature_val = X[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_predictions(X, tree.left)
        else:
            return self.make_predictions(X, tree.right)
        
    # predict on a single data point
    def predict(self, X):
        predictions = [self.make_predictions(X, self.root) for x in X]
        return predictions
            
        
        
        
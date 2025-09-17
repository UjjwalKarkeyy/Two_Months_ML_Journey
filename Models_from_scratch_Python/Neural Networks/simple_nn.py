# yt playlist: https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3

import numpy as np

# working with three neurons

inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87],
        ]
# bias of three neurons
biases = [2, 3, 0.5]

'''
Just a list [2, 3, 0.5] is equivalent to a vector
List of list [[],[]] whereas, is equivalent to a matrix of vectors
'''

# output of three neurons
output = np.dot(weights, inputs) + biases
print(output)
import numpy as np
from collections import Counter

arr = np.array([[1, 0, 1],
                [0, 1, 1]])

counter = Counter([1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 3, 3])
most_common = counter.most_common(1)[0][0]
# Output: [(1, 6), (0, 5)]

print(most_common)

# import tensorflow as tf
import numpy as np
from sklearn.metrics import normalized_mutual_info_score


# print(normalized_mutual_info_score([1, 1, 2, 3], [2, 2, 1, 4]))
vals = np.array([[1, 2], [2, 3]])
x = np.array([3, 5])
print(np.sum(x * vals[0]))

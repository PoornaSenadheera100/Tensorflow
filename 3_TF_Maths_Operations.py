import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Element wise operations
tensor_a = tf.constant([20, 40, 67])
tensor_b = tf.constant([87, 98, 34])

# Addition
tensor_c = tf.add(tensor_a, tensor_b)
# or tensor_c = tensor_a + tensor_b
print(tensor_c)

# Subtraction
tensor_c = tf.subtract(tensor_a, tensor_b)
# or tensor_c = tensor_a - tensor_b
print(tensor_c)

# Division
tensor_c = tf.divide(tensor_a, tensor_b)
# or tensor_c = tensor_a / tensor_b
print(tensor_c)

# Multiplication
tensor_c = tf.multiply(tensor_a, tensor_b)
# or tensor_c = tensor_a * tensor_b
print(tensor_c)

# Dot Product
tensor_a = tf.constant(np.array([[1, 2], [3, 4]]))
tensor_b = tf.constant(np.array([[11, 12], [13, 14]]))
tensor_c = tf.tensordot(tensor_a, tensor_b, axes=1)
print(tensor_c)
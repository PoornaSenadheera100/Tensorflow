import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)

tensor_a = tf.constant(8)
print(tensor_a)

tensor_a = tf.constant(8.0)
print(tensor_a)

tensor_a = tf.constant([[10, 20, 30], [40, 50, 60]], shape=(3, 2))
print(tensor_a)

tensor_a = tf.constant(8, shape=(1, 1), dtype=tf.float32)
print(tensor_a)

tensor_a = tf.ones((4, 4))
print(tensor_a)

tensor_a = tf.zeros((4, 4))
print(tensor_a)

tensor_a = tf.eye(5, dtype=tf.int32)
print(tensor_a)
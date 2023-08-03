import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Normal Distribution
x = tf.random.normal((4, 4), mean=0, stddev=1)
print(x)

# Uniform Distribution
x = tf.random.uniform((1, 5), minval=0, maxval=2)
print(x)

# Range function
x = tf.range(start=5, limit=20, delta=3)
print(x)

# Casting
y = tf.cast(x, tf.float64)
print(y)
import os
import  tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.constant([0, 10, 56, 32, 20, 89, 100])
print(x)

print(x[:])

# Omitting 1st element
print(x[1:])

# 1st 4 elements
print(x[:4])

# Skipping
print(x[::2])
print(x[1::2])

# Reversing
print(x[::-1])
print(x[::-2])

# Indices (only get selected indexes)
indicesIndexes = tf.constant([0, 3])
indexedTensor = tf.gather(x, indices=indicesIndexes)
print(indexedTensor)

y = tf.constant([[10, 20], [30, 40], [50, 60], [70, 80]])
print(y)

# Get 1st row with all elements
print(y[0])

# Get 1st 2 rows only with the 1st elements
print(y[:2, :1])

# Get 2nd and 3rd rows
print(y[1:3])

# Reshaping
z = tf.range(start=0, limit=16)
print(z)
reshapedTensor = tf.reshape(z, (4, 4))
print(reshapedTensor)
reshapedTensor = tf.reshape(z, (2, 8))
print(reshapedTensor)
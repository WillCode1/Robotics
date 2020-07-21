import numpy as np
import tensorflow as tf

# tf.constant常量，不可训练
a = tf.constant([1, 5], dtype=tf.int32)
print(a)
print(a.dtype)
print(a.shape)
print('====================tf.constant====================\n')

# numpy => tensor
x = np.arange(0, 5)
b = tf.convert_to_tensor(x, dtype=tf.int64)
print(x)
print(b)
print('====================tf.convert_to_tensor====================\n')

# use as numpy
a = tf.zeros((2, 3))
b = tf.ones(4)
c = tf.fill([2, 2], 9)
print(a)
print(b)
print(c)
print('====================use as numpy====================\n')

# 生成满足正态分布的随机数
d = tf.random.normal([2, 2], mean=0.5, stddev=1)
print(d)
# 生成满足正态分布的随机数，且截断于正负两倍标准差之内
e = tf.random.truncated_normal([2, 2], mean=0.5, stddev=1)
print(e)
# 生成均匀分布的随机数
f = tf.random.uniform([2, 2], minval=0, maxval=1)
print(f)
print('====================tf.random====================\n')


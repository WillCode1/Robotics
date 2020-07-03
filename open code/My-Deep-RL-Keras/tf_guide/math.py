import tensorflow as tf

# 加减乘除
a = tf.ones([1, 3])
b = tf.fill([1, 3], 3.)
print(a)
print(b)
print(tf.add(a, b))
print(tf.subtract(a, b))
print(tf.multiply(a, b))
print(tf.divide(a, b))
print('====================tf.add====================\n')

# 指数、平方、开方
a = tf.fill([1, 2], 3.)
print(a)
print(tf.pow(a, 3))
print(tf.square(a))
print(tf.sqrt(a))
print('====================tf.square====================\n')

# 矩阵乘法
a = tf.ones([3, 2])
b = tf.fill([2, 3], 3.)
print(tf.matmul(a, b))
print('====================tf.matmul====================\n')

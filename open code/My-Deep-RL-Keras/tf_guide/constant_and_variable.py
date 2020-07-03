import tensorflow as tf

# 强转和min、max
a = tf.constant([1, 2, 3], dtype=tf.float64)
print(a)
b = tf.cast(a, tf.int32)  # 强转
print(b)
print(tf.reduce_max(b), tf.reduce_min(b))
print('====================tf.reduce_max====================\n')

'''
     -----------------------> axis=1
  |  |--------|--------|--------|
  |  |        |  col0  |  col1  |
  |  |--------|--------|--------|
  |  |  row0  |        |        |
  |  |--------|--------|--------|
  |  |  row1  |        |        |
  |  |--------|--------|--------|
  v
axis=0
'''
# 计算指定维度的mean、sum
a = tf.constant([[1, 2, 3],
                 [1, 2, 3]])
print(tf.reduce_mean(a, axis=0))
print(tf.reduce_sum(a, axis=1))
print('====================tf.reduce_mean====================\n')

# tf.Variable变量，可训练
# 下面为神经网路初始化参数的代码
w = tf.Variable(tf.random.normal([2, 2], mean=0, stddev=1))
print('====================tf.Variable====================\n')

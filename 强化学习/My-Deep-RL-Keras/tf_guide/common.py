import numpy as np
import tensorflow as tf

# 条件判断
a = tf.constant([1, 2, 3, 1, 1])
b = tf.constant([0, 1, 3, 4, 5])

c = tf.where(tf.greater(a, b), a, b)  # 若a>b, 返回对应位置的元素，否则返回b 对应位置元素
print("c:", c)
print('====================tf.where====================\n')

# 将两个数组按垂直方向叠加
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.vstack((a, b))
print("c:\n", c)
print('====================np.vstack====================\n')

# np.mgrid[]
# c.ravel()
# np.c_[]

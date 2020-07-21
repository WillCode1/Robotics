# 像NumPy一样使用TensorFlow
import numpy as np
import tensorflow as tf

# 使用tf.constant()创建张量。例如，下面的张量表示的是两行三列的浮点数矩阵
tf.constant([[1., 2., 3.], [4., 5., 6.]])  # matrix
tf.constant(42)  # 标量

# 就像ndarray一样，tf.Tensor也有形状和数据类型（dtype）
t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
print(t.shape)
print(t.dtype)

# 索引和NumPy中很像
print(t[:, 1:])
print(t[..., 1, tf.newaxis])

# 可以看到，t + 10等同于调用tf.add(t, 10)，-和*也支持。
# @运算符是在Python3.5中出现的，用于矩阵乘法，等同于调用函数tf.matmul()。
print(t + 10)
print(tf.square(t))
print(t @ tf.transpose(t))

# 张量和NumPy融合地非常好：使用NumPy数组可以创建张量，张量也可以创建NumPy数组。
# 可以在NumPy数组上运行TensorFlow运算，也可以在张量上运行NumPy运算
a = np.array([2., 4., 5.])
# 警告：NumPy默认使用64位精度，TensorFlow默认用32位精度。这是因为32位精度通常对于神经网络就足够了，另外运行地更快，使用的内存更少。
# 因此当你用NumPy数组创建张量时，一定要设置dtype=tf.float32。
print(tf.constant(a, dtype=tf.float32))
print(t.numpy())  # 或 np.array(t)
print(tf.square(a))
print(np.square(t))
# ============================================

# 类型转换
# TensorFlow不会自动做任何类型转换：只是如果用不兼容的类型执行了张量运算，TensorFlow就会报异常。
# print(tf.constant(2.) + tf.constant(40))
# print(tf.constant(2.) + tf.constant(40., dtype=tf.float64))

# 这点可能一开始有点恼人，但是有其存在的理由。如果真的需要转换类型，可以使用tf.cast()
t2 = tf.constant(40., dtype=tf.float64)
tf.constant(2.0) + tf.cast(t2, tf.float32)
# ============================================

# 变量
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
print(v)
# tf.Variable和tf.Tensor很像：可以运行同样的运算，可以配合NumPy使用，也要注意类型。
# 可以使用assign()方法对其就地修改（或assign_add()、assign_sub()）。
# 使用切片的assign()方法可以修改独立的切片（直接赋值行不通），或使用scatter_update()、scatter_nd_update()方法
v.assign(2 * v)  # => [[2., 4., 6.], [8., 10., 12.]]
v[0, 1].assign(42)  # => [[2., 42., 6.], [8., 10., 12.]]
v[:, 2].assign([0., 1.])  # => [[2., 42., 0.], [8., 10., 1.]]
v.scatter_nd_update(indices=[[0, 0], [1, 2]], updates=[100., 200.])  # => [[100., 42., 0.], [8., 10., 200.]]

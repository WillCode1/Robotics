import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image
from tensorflow import keras

# 加载样本图片
china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255
images = np.array([china, flower])
batch_size, height, width, channels = images.shape

# 创建两个过滤器
filters = np.zeros(shape=(7, 7, channels, 3), dtype=np.float32)
filters[:, 3, :, 0] = 1  # 垂直线
filters[3, :, :, 1] = 1  # 水平线
filters[0, 6:, 0: 6, 2] = 1

outputs = tf.nn.conv2d(images, filters, strides=1, padding="SAME")

plt.imshow(outputs[0, :, :, 1], cmap="gray")  # 画出第1张图的第2个特征映射
plt.show()
plt.imshow(outputs[0, :, :, 0], cmap="gray")
plt.show()
plt.imshow(outputs[0, :, :, 2], cmap="gray")
plt.show()

conv = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation="relu")

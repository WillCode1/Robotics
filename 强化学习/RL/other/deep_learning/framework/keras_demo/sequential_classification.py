from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from keras import backend as K

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

print(X_train_full.shape)
print(X_train_full.dtype)

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# print(class_names[y_train[0]])

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
# model.add(keras.layers.InputLayer(input_shape=[28, 28]))
# model.add(keras.layers.Dense(300, activation=K.relu))
model.add(keras.layers.Dense(300, activation=keras.activations.relu))
model.add(keras.layers.Dense(100, activation=keras.activations.relu))
model.add(keras.layers.Dense(10, activation="softmax"))

# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     keras.layers.Dense(300, activation="relu"),
#     keras.layers.Dense(100, activation="relu"),
#     keras.layers.Dense(10, activation="softmax")
# ])

output_layer = keras.layers.Dense(10)
print(model.summary())
print(model.layers)

hidden1 = model.layers[1]
print(hidden1.name)
print(model.get_layer('dense') is hidden1)

weights, biases = hidden1.get_weights()
print(weights.shape)
print(biases.shape)

# 解释下这段代码。首先，因为使用的是稀疏标签（每个实例只有一个目标类的索引，在这个例子中，目标类索引是0到9），
# 且就是这十个类，没有其它的，所以使用的是"sparse_categorical_crossentropy"损失函数。

# 如果每个实例的每个类都有一个目标概率（比如独热矢量，[0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]，来表示类3），
# 则就要使用"categorical_crossentropy"损失函数。

# 如果是做二元分类（有一个或多个二元标签），输出层就得使用"sigmoid"激活函数，损失函数则变为"binary_crossentropy"。

# 如果要将稀疏标签转变为独热矢量标签，可以使用函数keras.utils.to_categorical()。还以使用函数np.argmax()，axis=1。

# 使用SGD时，调整学习率很重要，必须要手动设置好，optimizer=keras.optimizers.SGD(lr=???)。
# optimizer="sgd"不同，它的学习率默认为lr=0.01。
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# 提示：除了通过参数validation_data传递验证集，也可以通过参数validation_split从训练集分割出一部分作为验证集。
# 比如，validation_split=0.1可以让Keras使用训练数据（打散前）的末尾10%作为验证集。

# 如果训练集非常倾斜，一些类过渡表达，一些欠表达，在调用fit()时最好设置class_weight参数，可以加大欠表达类的权重，
# 减小过渡表达类的权重。Keras在计算损失时，会使用这些权重。
# 如果每个实例都要加权重，可以设置sample_weight（这个参数优先于class_weight）。
# 如果一些实例的标签是通过专家添加的，其它实例是通过众包平台添加的，最好加大前者的权重，此时给每个实例都加权重就很有必要。
# 通过在validation_data元组中，给验证集加上样本权重作为第三项，还可以给验证集添加样本权重。

# fit()方法会返回History对象，包含：训练参数（history.params）、周期列表（history.epoch）、
# 以及最重要的包含训练集和验证集的每个周期后的损失和指标的字典（history.history）。
# 如果用这个字典创建一个pandas的DataFrame，然后使用方法plot()，就可以画出学习曲线
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
plt.show()

# 如果仍然对模型的表现不满意，就需要调节超参数了。首先是学习率。
# 如果调节学习率没有帮助，就尝试换一个优化器（记得再调节任何超参数之后都重新调节学习率）。
# 如果效果仍然不好，就调节模型自身的超参数，比如层数、每层的神经元数，每个隐藏层的激活函数。
# 还可以调节其它超参数，比如批次大小（通过fit()的参数batch_size，默认是32）。
# 只需使用evaluate()方法（evaluate()方法包含参数batch_size和sample_weight）
model.evaluate(X_test, y_test)

X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba.round(2))

y_pred = model.predict_classes(X_new)
print(y_pred)
print(np.array(class_names)[y_pred])
print(y_test[:3])

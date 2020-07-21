import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

root_logdir = os.path.join(os.curdir, "my_logs")


# 我们先定义TensorBoard的根日志目录，还有一些根据当前日期生成子目录的小函数。
# 你可能还想在目录名中加上其它信息，比如超参数的值，方便知道查询的内容：
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()  # e.g., './my_logs/run_2019_06_07-15_15_22'

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# 因为数据集有噪音，我们就是用一个隐藏层，并且神经元也比之前少，以避免过拟合
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])

model.compile(loss="mean_squared_error", optimizer="sgd")

# Keras提供了一个TensorBoard()调回
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid), callbacks=[tensorboard_cb])
# model.save("my_keras_model.h5")

mse_test = model.evaluate(X_test, y_test)

X_new = X_test[:3]  # pretend these are new instances
y_pred = model.predict(X_new)

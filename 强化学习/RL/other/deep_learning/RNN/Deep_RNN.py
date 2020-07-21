import numpy as np
from tensorflow import keras


# 这个函数可以根据要求创建出时间序列（通过batch_size参数），长度为n_steps，每个时间步只有1个值。
# 函数返回NumPy数组，形状是[批次大小, 时间步数, 1]，每个序列是两个正弦波之和（固定强度+随机频率和相位），加一点噪音。
def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    # np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
    # 返回“num”个等间距的样本，在区间[start, stop]中，其中，区间的结束端点可以被排除在外
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))    # wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))   # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)     # + noise
    return series[..., np.newaxis].astype(np.float32)


n_steps = 50
series = generate_time_series(10000, n_steps + 1)
# 当处理时间序列时，输入特征通常用3D数组来表示，其形状是 [批次大小, 时间步数, 维度]
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]


# 单步预测1个值
def forecast_one_step():
    # 因为SimpleRNN层默认使用tanh激活函数，预测值位于-1和1之间。想使用另一个激活函数该怎么办呢？
    # 出于这些原因，最好使用紧密层：运行更快，准确率差不多，可以选择任何激活函数。
    # 如果做了替换，要将第二个循环层的return_sequences=True删掉
    model = keras.models.Sequential([
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.SimpleRNN(20),
        keras.layers.Dense(1)
    ])

    model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999))
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid))
    y_pred = model.predict(X_test)
    # RNN one step model MSE: 0.003408432239666581
    print("RNN one step model MSE: {0}".format(np.mean(keras.losses.mean_squared_error(y_test, y_pred))))
    return model


# 单步预测10个值
def forecast_ten_step():
    series = generate_time_series(10000, n_steps + 10)
    X_train, Y_train = series[:7000, :n_steps], series[:7000, -10:, 0]
    X_valid, Y_valid = series[7000:9000, :n_steps], series[7000:9000, -10:, 0]
    X_test, Y_test = series[9000:, :n_steps], series[9000:, -10:, 0]

    model = keras.models.Sequential([
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.SimpleRNN(20),
        keras.layers.Dense(10)
    ])

    model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999))
    history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid))
    Y_pred = model.predict(X_test)
    # RNN ten step model MSE: 0.007886798121035099
    print("RNN ten step model MSE: {0}".format(np.mean(keras.losses.mean_squared_error(Y_test, Y_pred))))


# 每个时间步预测10个值，在时间步0，模型输出一个包含时间步1到10的预测矢量；
# 在时间步1，模型输出一个包含时间步2到11的预测矢量，以此类推。
def forecast_ten_step_by_once():
    series = generate_time_series(10000, n_steps + 10)
    X_train, Y_train = series[:7000, :n_steps], series[:7000, -10:, 0]
    X_valid, Y_valid = series[7000:9000, :n_steps], series[7000:9000, -10:, 0]
    X_test, Y_test = series[9000:, :n_steps], series[9000:, -10:, 0]

    # 准备目标序列（X_train 和 Y_train有许多重复）
    Y = np.empty((10000, n_steps, 10))  # each target is a sequence of 10D vectors
    for step_ahead in range(1, 10 + 1):
        Y[:, :, step_ahead - 1] = series[:, step_ahead:step_ahead + n_steps, 0]
    Y_train = Y[:7000]
    Y_valid = Y[7000:9000]
    Y_test = Y[9000:]

    model = keras.models.Sequential([
        # 将输入从 [批次大小, 时间步数, 输入维度] 变形为 [批次大小 × 时间步数, 输入维度20]
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.SimpleRNN(20, return_sequences=True),
        # Keras提供了TimeDistributed层：它将任意层（比如，紧密层）包装起来，然后在输入序列的每个时间步上使用
        # 将输出从 [批次大小 × 时间步数, 输出维度20] 变形为 [批次大小, 时间步数, 输出维度10]
        keras.layers.TimeDistributed(keras.layers.Dense(10))
    ])

    def last_time_step_mse(Y_true, Y_pred):
        return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

    optimizer = keras.optimizers.Adam(lr=0.01)
    model.compile(loss="mse", optimizer=optimizer, metrics=[last_time_step_mse])
    # 得到的MSE值为0.006
    history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid))


if __name__ == "__main__":
    # 使用朴素预测创建基线模型
    y_pred = X_valid[:, -1]
    # y_pred = 1~50步，y_valid = 51步
    # Base model MSE: 0.02003837376832962
    print("Base model MSE: {0}".format(np.mean(keras.losses.mean_squared_error(y_valid, y_pred))))

    # 训练单步预测1个值的模型
    model = forecast_one_step()

    # 提前预测几10个时间步
    series = generate_time_series(1, n_steps + 10)
    X_new, Y_new = series[:, :n_steps], series[:, n_steps:]
    X = X_new
    for step_ahead in range(10):
        y_pred_one = model.predict(X[:, step_ahead:])[:, np.newaxis, :]
        X = np.concatenate([X, y_pred_one], axis=1)

    Y_pred = X[:, n_steps:]
    # RNN one-by-one MSE: 0.01862586848437786
    print("RNN one-by-one MSE: {0}".format(np.mean(keras.losses.mean_squared_error(Y_new, Y_pred))))

    # 训练单步预测10个值的模型
    forecast_ten_step()

    # 训练每个时间步预测10个值的模型
    forecast_ten_step_by_once()

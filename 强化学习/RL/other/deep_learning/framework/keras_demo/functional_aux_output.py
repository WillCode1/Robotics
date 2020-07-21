from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# 但当调用fit()时，不是传入矩阵X_train，而是传入一对矩阵(X_train_A, X_train_B)：每个输入一个矩阵。
# 同理调用evaluate()或predict()时，X_valid、X_test、X_new也要变化
X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

# 搭建模型
input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape=[6], name="deep_input")
# 创建一个有30个神经元的紧密层，激活函数是ReLU。创建好之后，将其作为函数，直接将输入传给它。
# 这就是Functional API的得名原因。
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])

# 添加额外的输出，output层前面都一样
output = keras.layers.Dense(1, name="main_output")(concat)
aux_output = keras.layers.Dense(1, name="aux_output")(hidden2)
model = keras.Model(inputs=[input_A, input_B], outputs=[output, aux_output])

# 每个输出都要有自己的损失函数。因此在编译模型时，需要传入损失列表（如果只传入一个损失，Keras会认为所有输出是同一个损失函数）。
# Keras默认计算所有损失，将其求和得到最终损失用于训练。主输出比辅助输出更值得关心，所以要提高它的权重
model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer="sgd")

# 此时若要训练模型，必须给每个输出贴上标签。在这个例子中，主输出和辅输出预测的是同一件事，因此标签相同。
# 传入数据必须是(y_train, y_train)（y_valid和y_test也是如此）
history = model.fit([X_train_A, X_train_B], [y_train, y_train], epochs=20,
                    validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid]))

# 当评估模型时，Keras会返回总损失和各个损失值
total_loss, main_loss, aux_loss = model.evaluate([X_test_A, X_test_B], [y_test, y_test])
# 相似的，方法predict()会返回每个输出的预测值
y_pred = model.predict((X_new_A, X_new_B))

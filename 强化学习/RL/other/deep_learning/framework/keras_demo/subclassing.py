# 动态模型
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


# 这个例子和Functional API很像，除了不用创建输入；只需要在call()使用参数input，另外的不同是将层的创建和和使用分割了。
# 最大的差别是，在call()方法中，你可以做任意想做的事：for循环、if语句、低级的TensorFlow操作，可以尽情发挥想象（见第12章）！
# Subclassing API可以让研究者试验各种新创意。

# 然而代价也是有的：模型架构隐藏在call()方法中，所以Keras不能对其检查；不能保存或克隆；
# 当调用summary()时，得到的只是层的列表，没有层的连接信息。另外，Keras不能提前检查数据类型和形状，所以很容易犯错。
# 所以除非真的需要灵活性，还是使用Sequential API或Functional API吧。
class WideAndDeepModel(keras.Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)  # handles standard args (e.g., name)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)

    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output


model = WideAndDeepModel()

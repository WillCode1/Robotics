# 字母层面有状态
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 下载、读取莎士比亚的作品
shakespeare_url = "https://homl.info/shakespeare"  # shortcut URL
filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    shakespeare_text = f.read()

# 将每个替换编码为一个整数。方法之一是创建一个自定义预处理层，就像之前在第13章做的那样。
# 但在这里，使用Keras的Tokenizer会更加简单。
# 首先，将一个将tokenizer拟合到文本：tokenizer能从文本中发现所有的替换，
# 并将所有替换映射到不同的替换ID，映射从1开始（注意不是从0开始，0是用来做遮挡的，后面会看到）
tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(shakespeare_text)

print(tokenizer.texts_to_sequences(["First"]))
print(tokenizer.sequences_to_texts([[20, 6, 9, 8, 3]]))
max_id = len(tokenizer.word_index)  # number of distinct characters
dataset_size = tokenizer.document_count  # total number of characters

[encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1
print(encoded)

# 切分数据集
train_size = dataset_size * 90 // 100
dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])

tf.random.set_seed(42)
batch_size = 32
n_steps = 100
window_length = n_steps + 1  # target = input 向前移动1个角色
dataset = dataset.window(window_length, shift=n_steps, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(window_length))
dataset = dataset.repeat().batch(1)
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
dataset = dataset.prefetch(1)

encoded_parts = np.array_split(encoded[:train_size], batch_size)
datasets = []
for encoded_part in encoded_parts:
    dataset = tf.data.Dataset.from_tensor_slices(encoded_part)
    dataset = dataset.window(window_length, shift=n_steps, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_length))
    datasets.append(dataset)
dataset = tf.data.Dataset.zip(tuple(datasets)).map(lambda *windows: tf.stack(windows))
dataset = dataset.repeat().map(lambda windows: (windows[:, :-1], windows[:, 1:]))
dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
dataset = dataset.prefetch(1)

model = keras.models.Sequential([
    keras.layers.GRU(128, return_sequences=True, stateful=True, dropout=0.2, recurrent_dropout=0.2,
                     batch_input_shape=[batch_size, None, max_id]),
    keras.layers.GRU(128, return_sequences=True, stateful=True, dropout=0.2, recurrent_dropout=0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(max_id, activation="softmax"))
])


class ResetStatesCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()


model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
steps_per_epoch = train_size // batch_size // n_steps
model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=50, callbacks=[ResetStatesCallback()])


# 写个小函数来做预处理
def preprocess(texts):
    X = np.array(tokenizer.texts_to_sequences(texts)) - 1
    return tf.one_hot(X, max_id)


X_new = preprocess(["How are yo"])
Y_pred = model.predict_classes(X_new)
print(tokenizer.sequences_to_texts(Y_pred + 1)[0][-1])  # 1st sentence, last char
print(tokenizer.sequences_to_texts(Y_pred + 1))


def next_char(text, temperature=1):
    X_new = preprocess([text])
    y_proba = model.predict(X_new)[0, -1:, :]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]


def complete_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text


print(complete_text("t", temperature=0.2))
print(complete_text("w", temperature=1))
print(complete_text("w", temperature=2))

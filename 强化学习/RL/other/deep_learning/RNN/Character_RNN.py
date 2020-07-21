# 字母层面无状态
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

# 使用window()方法，将这个长序列转化为许多小窗口文本
# 可以调节n_steps：用短输入序列训练RNN更为简单，但肯定的是RNN学不到任何长度超过n_steps的规律，所以n_steps不要太短。
n_steps = 100
window_length = n_steps + 1  # target = input 向前移动1个角色
dataset = dataset.window(window_length, shift=1, drop_remainder=True)

# 调用flat_map()方法：它能将嵌套数据集转换成打平的数据集
dataset = dataset.flat_map(lambda window: window.batch(window_length))

np.random.seed(42)
tf.random.set_seed(42)
batch_size = 32
# 窗口长度是11，不是101，批次大小是3，不是32
dataset = dataset.shuffle(10000).batch(batch_size)
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
# 独热编码
dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
dataset = dataset.prefetch(1)
for X_batch, Y_batch in dataset.take(1):
    print(X_batch.shape, Y_batch.shape)

model = keras.models.Sequential([
    keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id], dropout=0.2, recurrent_dropout=0.2),
    keras.layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(max_id, activation="softmax"))
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam")
history = model.fit(dataset, steps_per_epoch=train_size // batch_size, epochs=10)


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

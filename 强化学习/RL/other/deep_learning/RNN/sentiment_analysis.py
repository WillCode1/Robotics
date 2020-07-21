import tensorflow_datasets as tfds
from collections import Counter
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

with tf.name_scope('tool'):
    def preprocess(X_batch, y_batch):
        X_batch = tf.strings.substr(X_batch, 0, 300)
        X_batch = tf.strings.regex_replace(X_batch, rb"<br\s*/?>", b" ")
        X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
        X_batch = tf.strings.split(X_batch)
        return X_batch.to_tensor(default_value=b"<pad>"), y_batch

    def encode_words(X_batch, y_batch):
        return table.lookup(X_batch), y_batch

tf.random.set_seed(42)

(X_train, y_test), (X_valid, y_test) = keras.datasets.imdb.load_data()

word_index = keras.datasets.imdb.get_word_index()
id_to_word = {id_ + 3: word for word, id_ in word_index.items()}
for id_, token in enumerate(("<pad>", "<sos>", "<unk>")):
    id_to_word[id_] = token
" ".join([id_to_word[id_] for id_ in X_train[0][:10]])

datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
datasets.keys()

train_size = info.splits["train"].num_examples
test_size = info.splits["test"].num_examples

print(train_size, test_size)

for X_batch, y_batch in datasets["train"].batch(2).take(1):
    for review, label in zip(X_batch.numpy(), y_batch.numpy()):
        print("Review:", review.decode("utf-8")[:200], "...")
        print("Label:", label, "= Positive" if label else "= Negative")
        print()

preprocess(X_batch, y_batch)

vocabulary = Counter()
for X_batch, y_batch in datasets["train"].batch(32).map(preprocess):
    for review in X_batch:
        vocabulary.update(list(review.numpy()))

print(vocabulary.most_common()[:3])
print(len(vocabulary))

vocab_size = 10000
truncated_vocabulary = [word for word, count in vocabulary.most_common()[:vocab_size]]

word_to_id = {word: index for index, word in enumerate(truncated_vocabulary)}
for word in b"This movie was faaaaaantastic".split():
    print(word_to_id.get(word) or vocab_size)

words = tf.constant(truncated_vocabulary)
word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
num_oov_buckets = 1000
table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)

table.lookup(tf.constant([b"This movie was faaaaaantastic".split()]))

train_set = datasets["train"].repeat().batch(32).map(preprocess)
train_set = train_set.map(encode_words).prefetch(1)

for X_batch, y_batch in train_set.take(1):
    print(X_batch)
    print(y_batch)

embed_size = 128
model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size,
                           mask_zero=True,
                           input_shape=[None]),
    keras.layers.GRU(128, return_sequences=True),
    keras.layers.GRU(128),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(train_set, steps_per_epoch=train_size // 32, epochs=5)

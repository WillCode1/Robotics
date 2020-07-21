# 使用预训练模型做迁移学习
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

# 如果想创建一个图片分类器，但没有足够的训练数据，使用预训练模型的低层通常是不错的主意
# 可以通过设定with_info=True来获取数据集信息。
dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
dataset_size = info.splits["train"].num_examples  # 3670
class_names = info.features["label"].names  # ["dandelion", "daisy", ...]
n_classes = info.features["label"].num_classes  # 5

# 但是，这里只有"train"训练集，没有测试集和验证集，所以需要分割训练集。
# TF Datasets提供了一个API来做这项工作。比如，使用数据集的前10%作为测试集，接着的15%来做验证集，剩下的75%来做训练集
test_split, valid_split, train_split = tfds.Split.TRAIN.subsplit([10, 15, 75])
test_set = tfds.load("tf_flowers", split=test_split, as_supervised=True)
valid_set = tfds.load("tf_flowers", split=valid_split, as_supervised=True)
train_set = tfds.load("tf_flowers", split=train_split, as_supervised=True)


# 使用Xception的preprocess_input()函数来预处理图片
def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image, label


# 如果想做数据增强，可以修改训练集的预处理函数，给训练图片添加一些转换。
# 例如，使用tf.image.random_crop()随机裁剪图片，使用tf.image.random_flip_left_right()做随机水平翻转
batch_size = 32
train_set = train_set.shuffle(1000)
train_set = train_set.map(preprocess).batch(batch_size).prefetch(1)
valid_set = valid_set.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set.map(preprocess).batch(batch_size).prefetch(1)

# 然后加载一个在ImageNet上预训练的Xception模型。通过设定include_top=False，排除模型的顶层：排除了全局平均池化层和紧密输出层。
# 我们然后根据基本模型的输出，添加自己的全局平均池化层，然后添加紧密输出层（没有一个类就有一个单元，使用softmax激活函数）。
base_model = keras.applications.xception.Xception(weights="imagenet", include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation="softmax")(avg)
model = keras.Model(inputs=base_model.input, outputs=output)

# 第11章介绍过，最好冻结预训练层的权重，至少在训练初期如此
for layer in base_model.layers:
    layer.trainable = False

optimizer = keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
history = model.fit(train_set, epochs=5, validation_data=valid_set)

# 模型训练几个周期之后，它的验证准确率应该可以达到75-80%，然后就没什么提升了。
# 这意味着上层训练的差不多了，此时可以解冻所有层（或只是解冻上边的层），然后继续训练（别忘在冷冻和解冻层是编译模型）。
# 此时使用小得多的学习率，以避免破坏预训练的权重
for layer in base_model.layers:
    layer.trainable = True

# 更小的学习率
optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.001)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
history = model.fit(train_set, epochs=5, validation_data=valid_set)

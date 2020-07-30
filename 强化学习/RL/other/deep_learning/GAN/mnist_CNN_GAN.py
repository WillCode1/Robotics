from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(42)
tf.random.set_seed(42)

with tf.name_scope('tool'):
    def plot_multiple_images(images, n_cols=None):
        n_cols = n_cols or len(images)
        n_rows = (len(images) - 1) // n_cols + 1
        images = (images + 1) / 2
        if images.shape[-1] == 1:
            images = np.squeeze(images, axis=-1)
        plt.figure(figsize=(n_cols, n_rows))
        for index, image in enumerate(images):
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(image, cmap="binary")
            plt.axis("off")

    def train_gan(gan, dataset, batch_size, codings_size, n_epochs=50, g_loss=0.5, d_loss=0.1):
        generator, discriminator = gan.layers
        for epoch in range(n_epochs):
            discriminator_train_times = 0
            generator_train_times = 0
            batch_step = 0
            print("\n=======================================")
            print("Epoch {}/{}".format(epoch + 1, n_epochs))
            for X_batch in dataset:
                batch_step += 1
                start = time.time()
                # phase 1 - training the discriminator
                y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
                discriminator.trainable = True
                while True:
                    discriminator_train_times += 1
                    noise = tf.random.normal(shape=[batch_size, codings_size])
                    generated_images = generator(noise)
                    X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
                    discriminator_loss = discriminator.train_on_batch(X_fake_and_real, y1)
                    print("discriminator_train_times: {}, discriminator_loss: {}"
                          .format(discriminator_train_times, discriminator_loss))
                    if discriminator_loss < d_loss:
                        discriminator_train_times = 0
                        break

                # phase 2 - training the generator
                y2 = tf.constant([[1.]] * batch_size)
                discriminator.trainable = False
                while True:
                    generator_train_times += 1
                    noise = tf.random.normal(shape=[batch_size, codings_size])
                    generator_loss = gan.train_on_batch(noise, y2)
                    print("generator_train_times: {}, generator_loss: {}".format(generator_train_times, generator_loss))
                    if generator_loss < g_loss:
                        generator_train_times = 0
                        break
                print("batch_step: {}, discriminator_loss: {}, generator_loss: {}"
                      .format(batch_step, discriminator_loss, generator_loss))
                print("=======================================")
                end = time.time()
                print('Running time: %s Seconds' % (end - start))
                generator.save("generator_fashion_mnist_model.h5")
                if batch_step % 4 == 0:
                    noise = tf.random.normal(shape=[64, codings_size])
                    generated_images = generator(noise)
                    plot_multiple_images(generated_images, 8)
                    plt.show()

with tf.name_scope('data'):
    # 不需要测试集
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    X_train = X_train.astype(np.float32) / 255

    # 样本的数目，高和宽，颜色通道数
    X_train_dcgan = X_train.reshape(-1, 28, 28, 1) * 2. - 1.  # [-1, 1]

    batch_size = 256  # 这里的大批次可以减少过拟合
    dataset = tf.data.Dataset.from_tensor_slices(X_train_dcgan).shuffle(1000)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

codings_size = 100

with tf.name_scope('train'):
    lr = 0.01
    dropout = 0.5
    '''
    下面是我觉得很不错的技巧：
    1.输入的图片经过处理，将0-255的值变为-1到1的值。
    images = (images/255.0)*2 - 1
    2.在generator输出层使用tanh激励函数，使得输出范围在 (-1,1)
    3.保存生成的图片时，将矩阵值缩放到[0,1]之间
    gen_image = （gen_image+1) / 2
    4.使用leaky_relu激励函数，使得负值可以有一定的比重
    5.使用BatchNormalization，使分布更均匀，最后一层不要使用。
    6.在训练generator和discriminator的时候，一定要保证另外一个的参数是不变的，不能同时更新两个网络的参数。
    7.如果训练很容易卡住，可以考虑使用WGAN
    8.可以选择使用RMSprop optimizer
    '''
    generator = keras.models.Sequential([
        # 将输入转换为7 x 7 128通道的feature map
        keras.layers.Dense(7 * 7 * 128, input_shape=[codings_size]),
        keras.layers.Reshape([7, 7, 128]),
        keras.layers.BatchNormalization(),
        # 对应上面，上采样至 14 x 14
        keras.layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding="same", activation="selu"),
        keras.layers.BatchNormalization(),
        # keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same", activation="selu"),
        # keras.layers.BatchNormalization(),
        # 此处1为颜色通道数
        keras.layers.Conv2DTranspose(1, kernel_size=7, strides=2, padding="same", activation="tanh"),
        keras.layers.Reshape([28, 28, 1])
    ])
    discriminator = keras.models.Sequential([
        keras.layers.Conv2D(64, kernel_size=7, strides=2, padding="same", activation=keras.layers.LeakyReLU(0.2),
                            input_shape=[28, 28, 1]),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, kernel_size=3, padding="same", activation=keras.layers.LeakyReLU(0.2)),
        keras.layers.Conv2D(128, kernel_size=3, padding="same", activation=keras.layers.LeakyReLU(0.2)),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(256, kernel_size=3, padding="same", activation=keras.layers.LeakyReLU(0.2)),
        keras.layers.Conv2D(256, kernel_size=3, padding="same", activation=keras.layers.LeakyReLU(0.2)),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=keras.layers.LeakyReLU(0.2)),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(64, activation=keras.layers.LeakyReLU(0.2)),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(1, activation="sigmoid"),
    ])

    discriminator.compile(loss="binary_crossentropy", optimizer="RMSprop")
    # discriminator.compile(loss=keras.losses.KLDivergence(), optimizer="RMSprop")
    discriminator.trainable = False
    # generator = keras.models.load_model("generator_fashion_mnist_model.h5")
    gan = keras.models.Sequential([generator, discriminator])

    gan.compile(loss="binary_crossentropy", optimizer="RMSprop")

    # keras.backend.set_value(gan.optimizer.lr, 0.01)
    # discriminator是判断题，loss不用太低；generator是简答题，loss可以低一点
    train_gan(gan, dataset, batch_size, codings_size, n_epochs=2, g_loss=1., d_loss=0.2)

    # 展示
    noise = tf.random.normal(shape=[64, codings_size])
    generated_images = generator(noise)
    plot_multiple_images(generated_images, 8)
    plt.show()

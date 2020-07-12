import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from rl.CarEnv import CarEnv

K = keras.backend
kl_divergence = keras.losses.kullback_leibler_divergence


class KLDivergenceRegularizer(keras.regularizers.Regularizer):
    def __init__(self, weight, target=0.1):
        self.weight = weight
        self.target = target

    def __call__(self, inputs):
        mean_activities = K.mean(inputs, axis=0)
        return self.weight * (kl_divergence(self.target, mean_activities) +
                              kl_divergence(1. - self.target, 1. - mean_activities))


class AutoEncoder:
    def __init__(self, act_dim, state_dim, act_range, model_path):
        self.act_dim = act_dim
        self.act_range = act_range
        self.state_dim = state_dim

        self.model_path = model_path

    @staticmethod
    def create_conv2d_model(shape):
        inp = keras.layers.Input(shape=shape)
        x = keras.layers.Lambda(lambda image: tf.cast(image, np.float32) / 255)(inp)
        x = keras.layers.SeparableConv2D(64, 7, strides=2, activation="selu", padding="same")(x)
        x = keras.layers.SeparableConv2D(128, 3, strides=2, activation="selu", padding="same")(x)
        x = keras.layers.SeparableConv2D(256, 3, strides=2, activation="selu", padding="same")(x)
        output = keras.layers.SeparableConv2D(128, 3, strides=2, activation="selu", padding="same")(x)
        model = keras.Model(inputs=[inp], outputs=[output])
        # print(model.summary())
        return model

    def create_auto_encoder(self, shape):
        encoder = self.create_conv2d_model(shape)
        decoder = keras.models.Sequential([
            keras.layers.Conv2DTranspose(64, kernel_size=3, strides=4, padding="valid", activation="selu"),
            keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", activation="selu"),
            keras.layers.Conv2DTranspose(3, kernel_size=3, strides=2, padding="same", activation="sigmoid"),
            keras.layers.Reshape(target_shape=shape),
            keras.layers.Lambda(lambda image: image * 255)
        ])
        auto_encoder = keras.models.Sequential([encoder, decoder])
        return auto_encoder

    def unsupervised_pre_training(self, env, batch_size=32, n_epochs=1000, time_delta=30,
                                  lr=0.01, load_weights=True):
        sem_auto_encoder = self.create_auto_encoder(self.state_dim[0])
        depth_auto_encoder = self.create_auto_encoder(self.state_dim[0])
        sem_encoder, sem_decoder = sem_auto_encoder.layers
        depth_encoder, depth_decoder = depth_auto_encoder.layers

        optimizer = keras.optimizers.Adam(lr=lr, clipvalue=1.0)
        sem_auto_encoder.compile(loss=keras.losses.Huber(), optimizer=optimizer)
        depth_auto_encoder.compile(loss=keras.losses.Huber(), optimizer=optimizer)

        if load_weights:
            sem_encoder.load_weights(self.model_path + 'sem_encoder.h5')
            sem_decoder.load_weights(self.model_path + 'sem_decoder.h5')
            depth_encoder.load_weights(self.model_path + 'depth_encoder.h5')
            depth_decoder.load_weights(self.model_path + 'depth_decoder.h5')

        for epoch in range(n_epochs):
            sem_image = []
            depth_image = []

            while len(sem_image) < batch_size:
                time, done = 0, False
                state = env.reset()

                while not done:
                    action = np.array([10, 10])
                    new_state, _, done, _ = env.step(action)
                    if time % time_delta == 0:
                        sem, depth, _ = new_state
                        sem_image.append(sem)
                        depth_image.append(depth)
                        if len(sem_image) == batch_size:
                            break
                    time += 1

            sem_image = np.array(sem_image)
            depth_image = np.array(depth_image)

            index = [i for i in range(len(sem_image))]
            random.shuffle(index)
            sem_image = sem_image[index]
            depth_image = depth_image[index]

            sem_image = sem_image.astype(np.float32)
            depth_image = depth_image.astype(np.float32)

            print("\n=======================================")
            print("Epoch {}/{}".format(epoch + 1, n_epochs))
            for epoch in range(3):
                sem_loss = sem_auto_encoder.train_on_batch(sem_image, sem_image)
                depth_loss = depth_auto_encoder.train_on_batch(depth_image, depth_image)
                print("sem_loss: {}, depth_loss: {}".format(sem_loss, depth_loss))

            if np.isnan(sem_loss) or np.isnan(depth_loss):
                break

            sem_encoder.save_weights(self.model_path + 'sem_encoder.h5')
            sem_decoder.save_weights(self.model_path + 'sem_decoder.h5')
            depth_encoder.save_weights(self.model_path + 'depth_encoder.h5')
            depth_decoder.save_weights(self.model_path + 'depth_decoder.h5')
            print("Save model!")


if __name__ == "__main__":
    if not os.path.isdir("./models"):
        os.makedirs("./models")

    IM_WIDTH = 400
    IM_HEIGHT = 400
    batch_size = 4
    EPISODES = 1000
    image_shape = (IM_HEIGHT, IM_WIDTH, 3)
    state_dim = [image_shape, image_shape, 1]
    action_dim = 2  # [throttle_brake, steer]

    debug = False

    env = CarEnv(IM_HEIGHT, IM_WIDTH, show_sem_camera=False, run_seconds_per_episode=50,
                 no_rendering_mode=True, debug=debug)

    ae = AutoEncoder(act_dim=action_dim, state_dim=state_dim, model_path=f'models/', act_range=1.0)
    ae.unsupervised_pre_training(env, batch_size=batch_size, lr=1e-2, load_weights=False)

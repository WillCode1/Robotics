import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from rl.DDPG.ddpg import DDPG
from utils.networks import OrnsteinUhlenbeckProcess


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

    def create_auto_encoder(self):
        sem_camera, depth_camera, _ = self.state_dim

        sem_encoder = self.create_conv2d_model(sem_camera)
        sem_decoder = keras.models.Sequential([
            keras.layers.Conv2DTranspose(64, kernel_size=3, strides=4, padding="valid", activation="selu"),
            keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", activation="selu"),
            keras.layers.Conv2DTranspose(3, kernel_size=3, strides=2, padding="same", activation="sigmoid"),
            keras.layers.Reshape(target_shape=sem_camera)
        ])
        sem_ae = keras.models.Sequential([sem_encoder, sem_decoder])
        # print(sem_ae.summary())

        depth_encoder = self.create_conv2d_model(depth_camera)
        depth_decoder = keras.models.Sequential([
            keras.layers.Conv2DTranspose(64, kernel_size=3, strides=4, padding="valid", activation="selu"),
            keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", activation="selu"),
            keras.layers.Conv2DTranspose(3, kernel_size=3, strides=2, padding="same", activation="sigmoid"),
            keras.layers.Reshape(target_shape=sem_camera),
            keras.layers.Lambda(lambda image: image * 255)
        ])
        depth_ae = keras.models.Sequential([depth_encoder, depth_decoder])
        return sem_ae, depth_ae

    def unsupervised_pre_training(self, env, image_num=10000):
        algo = DDPG(act_dim=self.act_dim, state_dim=self.state_dim, model_path=f'models/',
                    buffer_size=1, act_range=1.0)
        algo.actor.load_weights(f'models/')

        sem_image = []
        # depth_image = deque(maxlen=image_num)

        while len(sem_image) < image_num:
            time, done = 0, False
            state = env.reset()
            noise = OrnsteinUhlenbeckProcess(size=self.act_dim)

            while not done:
                action = algo.policy_action(state)
                action = np.clip(action + noise.generate(time), -self.act_range, self.act_range)
                new_state, _, done, _ = env.step(action)
                if time % 30 == 0:
                    sem, _, _ = new_state
                    sem_image.append(sem)
                state = new_state
                time += 1

        sem_image = np.array(sem_image)
        index = [i for i in range(len(sem_image))]
        random.shuffle(index)
        sem_image = sem_image[index]
        sem_image = sem_image.astype(np.float32)
        n_train = int(0.7 * len(sem_image))
        X_train, Y_train = sem_image[:n_train]
        X_valid, Y_valid = sem_image[n_train:]


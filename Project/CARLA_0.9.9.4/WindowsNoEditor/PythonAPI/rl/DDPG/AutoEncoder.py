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
        optimizer = keras.optimizers.Adam(lr=0.01, clipvalue=1.0)
        auto_encoder.compile(loss='mse', optimizer=optimizer)
        # print(auto_encoder.summary())
        return auto_encoder, encoder

    def unsupervised_pre_training(self, env, batch_size=128, n_epochs=1000, time_delta=30):
        algo = DDPG(act_dim=self.act_dim, state_dim=self.state_dim, model_path=f'models/',
                    buffer_size=1, act_range=1.0)
        algo.actor.load_weights(f'models/')

        sem_auto_encoder, sem_encoder = self.create_auto_encoder(self.state_dim[0])
        depth_auto_encoder, depth_encoder = self.create_auto_encoder(self.state_dim[0])

        sem_image = []
        depth_image = []

        for epoch in range(n_epochs):
            sem_image.clear()
            depth_image.clear()

            while len(sem_image) < batch_size:
                time, done = 0, False
                state = env.reset()
                noise = OrnsteinUhlenbeckProcess(size=self.act_dim)

                while not done:
                    action = algo.policy_action(state)
                    action = np.clip(action + noise.generate(time), -self.act_range, self.act_range)
                    new_state, _, done, _ = env.step(action)
                    if time % time_delta == 0:
                        sem, depth, _ = new_state
                        sem_image.append(sem)
                        depth_image.append(depth)
                        if len(sem_image) == batch_size:
                            break
                    state = new_state
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

            sem_encoder.save_weights(self.model_path + 'sem_encoder.h5')
            depth_encoder.save_weights(self.model_path + 'depth_encoder.h5')

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import RandomUniform
K = keras.backend


class Critic:
    """ Critic for the DDPG Algorithm, Q-Value function approximator
    """
    def __init__(self, inp_dim, out_dim, lr, tau):
        self.state_dim = inp_dim
        self.action_dim = out_dim
        self.tau = tau

        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        loss_fn = keras.losses.Huber()
        self.model.compile(loss=loss_fn, optimizer=keras.optimizers.Adam(lr=lr))
        self.target_model.compile(loss=loss_fn, optimizer=keras.optimizers.Adam(lr=lr))

    def create_model(self):
        """ Assemble Critic network to predict q-values
        """
        sem_camera, depth_camera, velocity = self.state_dim
        input_1 = keras.layers.Input(shape=sem_camera, name="sem_camera_input")
        input_2 = keras.layers.Input(shape=depth_camera, name="depth_camera_input")
        input_3 = keras.layers.Input(shape=[velocity], name="velocity_input")
        action = keras.layers.Input(shape=self.action_dim, name="action_input")

        x = keras.layers.concatenate([input_1, input_2])
        x = keras.layers.Lambda(lambda image: tf.cast(image, np.float32) / 255)(x)

        x = keras.layers.Conv2D(64, 7, strides=3, activation="relu", padding="same")(x)
        x = keras.layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
        x = keras.layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
        x = keras.layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
        x = keras.layers.GlobalAvgPool2D()(x)

        x = keras.layers.concatenate([x, input_3, action])
        x = keras.layers.Dense(30, activation="selu")(x)
        x = keras.layers.Dense(10, activation="selu")(x)

        state_values = keras.layers.Dense(1)(x)
        raw_advantages = keras.layers.Dense(self.action_dim, kernel_initializer=RandomUniform())(x)
        advantages = raw_advantages - K.mean(raw_advantages, axis=1, keepdims=True)
        q_values = state_values + advantages

        model = keras.Model(inputs=[input_1, input_2, input_3, action], outputs=[q_values])
        return model

    def target_predict(self, inp):
        """ Predict Q-Values using the target network
        """
        states, actions = inp
        return self.target_model.predict([*states, actions])

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        for model_weight, target_weight in zip(self.model.weights, self.target_model.weights):
            target_weight.assign(self.tau * model_weight + (1 - self.tau) * target_weight)

    def save(self, path):
        self.model.save_weights(path + '_critic.h5')

    def load_weights(self, path):
        self.model.load_weights(path)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import RandomUniform


class Actor:
    """ Actor Network for the DDPG Algorithm
    """

    def __init__(self, inp_dim, out_dim, act_range, lr, tau):
        self.state_dim = inp_dim
        self.action_dim = out_dim
        self.act_range = act_range
        self.tau = tau

        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = keras.optimizers.Adam(lr=lr)

    def create_model(self):
        """ Actor Network for Policy function Approximation, using a tanh
        activation for continuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
        """
        sem_camera, depth_camera, velocity = self.state_dim
        input_1 = keras.layers.Input(shape=sem_camera, name="sem_camera_input")
        input_2 = keras.layers.Input(shape=depth_camera, name="depth_camera_input")
        input_3 = keras.layers.Input(shape=[velocity], name="velocity_input")

        x = keras.layers.concatenate([input_1, input_2])
        x = keras.layers.Lambda(lambda image: tf.cast(image, np.float32) / 255)(x)

        x = keras.layers.Conv2D(64, 7, strides=3, activation="relu", padding="same")(x)
        x = keras.layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
        x = keras.layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
        x = keras.layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
        x = keras.layers.GlobalAvgPool2D()(x)

        x = keras.layers.concatenate([x, input_3])
        x = keras.layers.Dense(30, activation="selu")(x)
        x = keras.layers.Dense(10, activation="selu")(x)

        action = keras.layers.Dense(self.action_dim, activation="tanh", kernel_initializer=RandomUniform())(x)
        # action = keras.layers.Lambda(lambda i: i * self.act_range)(action)
        model = keras.Model(inputs=[input_1, input_2, input_3], outputs=[action])
        return model

    def predict(self, state):
        """ Action prediction
        """
        return self.model.predict(state)

    def target_predict(self, inp):
        """ Action prediction (target network)
        """
        return self.target_model.predict(inp)

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        for model_weight, target_weight in zip(self.model.weights, self.target_model.weights):
            target_weight.assign(self.tau * model_weight + (1 - self.tau) * target_weight)

    def save(self, path):
        self.model.save_weights(path + '_actor.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
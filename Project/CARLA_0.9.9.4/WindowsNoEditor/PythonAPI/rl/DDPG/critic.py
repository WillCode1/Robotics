import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import RandomUniform


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

        q_values = keras.layers.Dense(self.action_dim, kernel_initializer=RandomUniform())(x)
        model = keras.Model(inputs=[input_1, input_2, input_3, action], outputs=[q_values])
        return model

    def action_gradients(self, states, actions):
        """ Compute Q-value gradients w.r.t. states and policy-actions
        """
        critic_target = self.model.predict((states, actions))
        action_gradients = tf.gradients(critic_target, actions)
        return action_gradients

    def target_predict(self, inp):
        """ Predict Q-Values using the target network
        """
        return self.target_model.predict(inp)

    def train_on_batch(self, states, actions, critic_target):
        """ Train the critic network on batch of sampled experience
        """
        return self.model.train_on_batch([states, actions], critic_target)

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def save(self, path):
        self.model.save_weights(path + '_critic.h5')

    def load_weights(self, path):
        self.model.load_weights(path)

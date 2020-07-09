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
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def train(self, states, action_gradient):
        """ Actor Training
        """
        with tf.GradientTape() as tape:
            actions = self.model(states)
        # 链式法则
        # 反馈Q值越大损失越小，得到的反馈Q值越小损失越大，因此只要对状态估计网络返回的Q值取负号
        params_grad = tape.gradient(actions, self.model.trainable_variables, -action_gradient)
        self.optimizer.apply_gradients(zip(params_grad, self.model.trainable_variables))

    def save(self, path):
        self.model.save_weights(path + '_actor.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
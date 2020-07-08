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
        self.optimizer = keras.optimizer.Adam(lr=lr)

    def create_model(self):
        """ Actor Network for Policy function Approximation, using a tanh
        activation for continuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
        """
        state = keras.layers.Input(shape=self.state_dim)
        x = keras.layers.Dense(256, activation='relu')(state)
        # x = keras.layers.GaussianNoise(1.0)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(128, activation="relu")(x)
        # x = keras.layers.GaussianNoise(1.0)(x)
        throttle_brake = keras.layers.Dense(self.action_dim, activation="tanh", kernel_initializer=RandomUniform())(x)
        steer = keras.layers.Dense(self.action_dim, activation="tanh", kernel_initializer=RandomUniform())(x)
        action = keras.layers.concatenate([throttle_brake, steer])
        # action = keras.layers.Lambda(lambda i: i * self.act_range)(action)
        return keras.Model(inputs=[state], outputs=[action])

    def predict(self, state):
        """ Action prediction
        """
        return self.model.predict(np.expand_dims(state, axis=0))

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
        actions = self.model.predict(states)
        # 链式法则
        # 反馈Q值越大损失越小，得到的反馈Q值越小损失越大，因此只要对状态估计网络返回的Q值取负号
        params_grad = tf.gradients(actions, self.model.trainable_variables, -action_gradient)
        self.optimizer.apply_gradients(zip(params_grad, self.model.trainable_variables))

    def save(self, path):
        self.model.save_weights(path + '_actor.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
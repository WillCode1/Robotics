import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import RandomUniform


class Actor:
    """ Actor Network for the DDPG Algorithm
    """

    def __init__(self, inp_dim, out_dim, act_range, lr, tau, hidden_layers):
        self.state_dim = inp_dim
        self.action_dim = out_dim
        self.act_range = act_range
        self.tau = tau
        self.hidden_layers = hidden_layers
        self.time = 0

        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = keras.optimizers.Adam(lr=lr)

    def create_model(self):
        """ Actor Network for Policy function Approximation, using a tanh
        activation for continuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
        """
        state = keras.layers.Input(shape=[self.state_dim])
        x = state
        x = keras.layers.BatchNormalization()(x)
        for hidden_size in self.hidden_layers:
            x = keras.layers.Dense(hidden_size)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation("selu")(x)

        action = keras.layers.Dense(1, kernel_initializer=RandomUniform())(x)
        action = keras.layers.BatchNormalization()(action)
        action = keras.layers.Activation("tanh")(action)
        action = keras.layers.Lambda(lambda i: i * self.act_range)(action)
        model = keras.Model(inputs=[state], outputs=[action])
        return model

    def predict(self, state):
        """ Action prediction
        """
        return self.model.predict(state)

    def target_predict(self, inp):
        """ Action prediction (target network)
        """
        return self.target_model.predict(inp)

    def transfer_weights(self, soft_update=True):
        if soft_update:
            for model_weight, target_weight in zip(self.model.weights, self.target_model.weights):
                target_weight.assign(self.tau * model_weight + (1 - self.tau) * target_weight)
        elif self.time % 50 == 0:
            self.target_model.set_weights(self.model.get_weights())

    def save(self, path):
        self.model.save_weights(path + 'actor.h5')

    def load_weights(self, path):
        self.model.load_weights(path + 'actor.h5')
        self.target_model.load_weights(path + 'actor.h5')
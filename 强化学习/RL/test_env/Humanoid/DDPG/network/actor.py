import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import RandomUniform


class Actor:
    def __init__(self, inp_dim, out_dim, act_range, lr, tau, hidden_layers):
        self.state_dim = inp_dim
        self.action_dim = out_dim
        self.act_range = act_range
        self.tau = tau
        self.hidden_layers = hidden_layers

        self.optimizer = keras.optimizers.Adam(lr=lr)
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

    def create_model(self):
        state = keras.layers.Input(shape=self.state_dim)
        x = state
        # x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(300, activation=keras.layers.LeakyReLU(0.2))(x)
        x = keras.layers.Dense(100, activation=keras.layers.LeakyReLU(0.2))(x)
        x = keras.layers.Dense(20, activation=keras.layers.LeakyReLU(0.2))(x)

        action = keras.layers.Dense(self.action_dim, activation="tanh", kernel_initializer=RandomUniform())(x)
        action = keras.layers.Lambda(lambda i: i * self.act_range)(action)
        model = keras.Model(inputs=[state], outputs=[action])
        return model

    def predict(self, state):
        return self.model.predict(state)

    def target_predict(self, inp):
        return self.target_model.predict(inp)

    def transfer_weights(self):
        for model_weight, target_weight in zip(self.model.weights, self.target_model.weights):
            target_weight.assign(self.tau * model_weight + (1 - self.tau) * target_weight)

    def save(self, path):
        self.model.save_weights(path + 'actor.h5')

    def load_weights(self, path):
        self.model.load_weights(path + 'actor.h5')
        self.target_model.load_weights(path + 'actor.h5')
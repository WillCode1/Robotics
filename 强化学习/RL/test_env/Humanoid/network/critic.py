import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import RandomUniform
K = keras.backend


class Critic:
    def __init__(self, inp_dim, out_dim, lr, tau, hidden_layers):
        self.state_dim = inp_dim
        self.action_dim = out_dim
        self.tau = tau
        self.hidden_layers = hidden_layers

        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        loss_fn = keras.losses.mean_squared_error
        self.model.compile(loss=loss_fn, optimizer=keras.optimizers.Adam(lr=lr))
        self.target_model.compile(loss=loss_fn, optimizer=keras.optimizers.Adam(lr=lr))

    def create_model(self):
        state = keras.layers.Input(shape=self.state_dim)
        action = keras.layers.Input(shape=self.action_dim)

        x = keras.layers.concatenate([state, action])
        # x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(300, activation=keras.layers.LeakyReLU(0.2))(x)
        x = keras.layers.Dense(100, activation=keras.layers.LeakyReLU(0.2))(x)
        x = keras.layers.Dense(20, activation=keras.layers.LeakyReLU(0.2))(x)

        state_values = keras.layers.Dense(1)(x)
        raw_advantages = keras.layers.Dense(self.action_dim, kernel_initializer=RandomUniform())(x)
        advantages = raw_advantages - K.mean(raw_advantages, axis=1, keepdims=True)
        q_values = state_values + advantages

        model = keras.Model(inputs=[state, action], outputs=[q_values])
        return model

    def target_predict(self, inp):
        states, actions = inp
        return self.target_model.predict([states, actions])

    def transfer_weights(self):
        for model_weight, target_weight in zip(self.model.weights, self.target_model.weights):
            target_weight.assign(self.tau * model_weight + (1 - self.tau) * target_weight)

    def save(self, path):
        self.model.save_weights(path + 'critic.h5')

    def load_weights(self, path):
        self.model.load_weights(path + 'critic.h5')
        self.target_model.load_weights(path + 'critic.h5')


if __name__ == "__main__":
    state_dim = 3
    action_dim = 1
    lr = 0.05
    tau = 0.01

    critic = Critic(state_dim, action_dim, lr, tau)
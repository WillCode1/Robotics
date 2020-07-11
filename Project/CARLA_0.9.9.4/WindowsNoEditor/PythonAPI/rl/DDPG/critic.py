import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import RandomUniform
K = keras.backend


class Critic:
    """ Critic for the DDPG Algorithm, Q-Value function approximator
    """
    def __init__(self, inp_dim, out_dim, lr, tau, create_conv2d_model):
        self.state_dim = inp_dim
        self.action_dim = out_dim
        self.tau = tau

        self.create_conv2d_model = create_conv2d_model

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
        sem_model = self.create_conv2d_model(sem_camera)
        # for layer in sem_model.layers:
        #     layer.trainable = False
        depth_model = self.create_conv2d_model(depth_camera)
        # for layer in depth_model.layers:
        #     layer.trainable = False

        velocity = keras.layers.Input(shape=[velocity])
        action = keras.layers.Input(shape=self.action_dim)

        sem = keras.layers.GlobalAvgPool2D()(sem_model.output)
        depth = keras.layers.GlobalAvgPool2D()(depth_model.output)

        x = keras.layers.concatenate([sem, depth, velocity, action])
        x = keras.layers.Dense(30, activation="selu")(x)
        x = keras.layers.Dense(10, activation="selu")(x)

        state_values = keras.layers.Dense(1)(x)
        raw_advantages = keras.layers.Dense(self.action_dim, kernel_initializer=RandomUniform())(x)
        advantages = raw_advantages - K.mean(raw_advantages, axis=1, keepdims=True)
        q_values = state_values + advantages

        model = keras.Model(inputs=[sem_model.input, depth_model.input, velocity, action],
                            outputs=[q_values])
        # print(model.summary())
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
        self.model.load_weights(path + '_critic.h5')
        self.target_model.load_weights(path + '_critic.h5')


if __name__ == "__main__":
    IM_WIDTH = 400
    IM_HEIGHT = 400
    image_shape = (IM_HEIGHT, IM_WIDTH, 3)
    state_dim = [image_shape, image_shape, 1]
    action_dim = 2
    lr = 0.05
    tau = 0.01

    from rl.DDPG.ddpg import DDPG
    critic = Critic(state_dim, action_dim, lr, tau, DDPG.create_conv2d_model)

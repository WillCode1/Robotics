import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import RandomUniform


class Actor:
    """ Actor Network for the DDPG Algorithm
    """

    def __init__(self, inp_dim, out_dim, act_range, lr, tau, create_conv2d_model):
        self.state_dim = inp_dim
        self.action_dim = out_dim
        self.act_range = act_range
        self.tau = tau

        self.create_conv2d_model = create_conv2d_model

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
        sem_model = self.create_conv2d_model(sem_camera)
        sem_model.load_weights('models/' + 'sem_encoder.h5')
        # for layer in sem_model.layers:
        #     layer.trainable = False
        depth_model = self.create_conv2d_model(depth_camera)
        depth_model.load_weights('models/' + 'depth_encoder.h5')
        # for layer in depth_model.layers:
        #     layer.trainable = False

        velocity = keras.layers.Input(shape=[velocity])

        sem = keras.layers.GlobalAvgPool2D()(sem_model.output)
        depth = keras.layers.GlobalAvgPool2D()(depth_model.output)

        x = keras.layers.concatenate([sem, depth, velocity])
        x = keras.layers.Dense(30, activation="selu")(x)
        x = keras.layers.Dense(10, activation="selu")(x)
        throttle_brake = keras.layers.Dense(1, activation="tanh",
                                            kernel_initializer=keras.initializers.Ones())(x)
        steer = keras.layers.Dense(1, activation="tanh", kernel_initializer=RandomUniform())(x)
        action = keras.layers.concatenate([throttle_brake, steer])
        # action = keras.layers.Lambda(lambda i: i * self.act_range)(action)
        model = keras.Model(inputs=[sem_model.input, depth_model.input, velocity], outputs=[action])
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
        else:
            self.target_model.set_weights(self.model.get_weights())

    def save(self, path):
        self.model.save_weights(path + '_actor.h5')

    def load_weights(self, path):
        self.model.load_weights(path + '_actor.h5')
        self.target_model.load_weights(path + '_actor.h5')
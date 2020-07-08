import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.optimizers import Adam


def create_model(input_shape, action_num, include_velocity=True):
    input_1 = keras.layers.Input(shape=input_shape, name="sem_camera_input")
    input_2 = keras.layers.Input(shape=input_shape, name="depth_camera_input")
    input_3 = keras.layers.Input(shape=[3], name="acceleration_input")
    input_4 = keras.layers.Input(shape=[3], name="angular_velocity_input")
    input_5 = keras.layers.Input(shape=[1], name="velocity_input")

    x = keras.layers.concatenate([input_1, input_2])
    x = keras.layers.Lambda(lambda image: tf.cast(image, np.float32) / 255)(x)

    x = keras.layers.Conv2D(64, 7, strides=3, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
    x = keras.layers.GlobalAvgPool2D()(x)

    if include_velocity:
        x = keras.layers.concatenate([x, input_3, input_4, input_5])
        x = keras.layers.Dense(30, activation="selu")(x)
        x = keras.layers.Dense(10, activation="selu")(x)

    output = keras.layers.Dense(action_num)(x)

    model = keras.Model(inputs=[input_1, input_2, input_3, input_4, input_5], outputs=[output])
    return model


class DQNAgent:
    def __init__(self, model, discount_rate=0.99, deque_maxlen=5000, update_frequency=50):
        self.discount_rate = discount_rate
        self.update_frequency = update_frequency

        self.model = model
        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = Adam(lr=0.01)
        self.loss_fn = keras.losses.Huber()

        self.replay_memory = deque(maxlen=deque_maxlen)
        self.target_update_counter = 0

    def training_step(self, min_replay_memory_size=1000, batch_size=32, soft_update=True):
        if len(self.replay_memory) < min_replay_memory_size:
            return

        batch_experiences = random.sample(self.replay_memory, batch_size)
        states, next_states = [[experience[field_index] for experience in batch_experiences]
                               for field_index in range(4) if field_index in [0, 3]]
        actions, rewards, dones = [np.array([experience[field_index] for experience in batch_experiences])
                                   for field_index in range(5) if field_index in [1, 2, 4]]
        input1, input2, input3, input4, input5 = [np.array([state[field_index] for state in states])
                                                  for field_index in range(5)]
        states = (input1, input2, input3, input4, input5)
        input1, input2, input3, input4, input5 = [np.array([next_state[field_index] for next_state in next_states])
                                                  for field_index in range(5)]
        next_states = (input1, input2, input3, input4, input5)

        next_Q_values = self.model.predict(next_states)
        best_next_actions = np.argmax(next_Q_values, axis=1)
        next_mask = tf.one_hot(best_next_actions, self.model.output.shape[1]).numpy()
        next_best_Q_values = (self.target_model.predict(next_states) * next_mask).sum(axis=1)
        target_Q_values = (rewards + (1 - dones) * self.discount_rate * next_best_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1)
        mask = tf.one_hot(actions, self.model.output.shape[1])
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        if soft_update:
            target_weights = self.target_model.get_weights()
            online_weights = self.model.get_weights()
            for index in range(len(target_weights)):
                target_weights[index] = 0.99 * target_weights[index] + 0.01 * online_weights[index]
            self.target_model.set_weights(target_weights)
        else:
            self.target_update_counter += 1
            if self.target_update_counter > self.update_frequency:
                self.target_update_counter = 0
                self.target_model.set_weights(self.model.get_weights())

    def epsilon_greedy_policy(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.model.output.shape[1])  # 动作个数
        else:
            Q_values = self.model.predict((state[0][np.newaxis], state[1][np.newaxis], state[2][np.newaxis],
                                           state[3][np.newaxis], state[4][np.newaxis]))
            return np.argmax(Q_values[0])

import os
import gym
import random
import numpy as np
import time
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

# 声明使用的环境
env = gym.make("Breakout-v0")
obs = env.reset()

keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)
env.seed(42)


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

    def training_step(self, min_replay_memory_size=1000, batch_size=32, soft_update=False):
        if len(self.replay_memory) < min_replay_memory_size:
            return

        batch_experiences = random.sample(self.replay_memory, batch_size)
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch_experiences])
            for field_index in range(5)]

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

        self.target_update_counter += 1
        if self.target_update_counter > self.update_frequency:
            self.target_update_counter = 0
            # 更新目标模型
            if not soft_update:
                self.target_model.set_weights(self.model.get_weights())
            else:  # 软更新方式
                target_weights = self.target_model.get_weights()
                online_weights = self.model.get_weights()
                for index in range(len(target_weights)):
                    target_weights[index] = 0.99 * target_weights[index] + 0.01 * online_weights[index]
                self.target_model.set_weights(target_weights)

    def epsilon_greedy_policy(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            # action = env.action_space.sample()
            return np.random.randint(self.model.output.shape[1])  # 动作个数
        else:
            Q_values = self.model.predict(state[np.newaxis])
            return np.argmax(Q_values[0])


def create_model(input_shape, action_num):
    input = keras.layers.Input(shape=input_shape)

    x = keras.layers.Lambda(lambda image: tf.cast(image, np.float32) / 255)(input)

    x = keras.layers.Conv2D(64, 7, activation="relu", padding="same")(x)
    x = keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(x)
    x = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(x)
    x = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = keras.layers.GlobalAvgPool2D()(x)

    output = keras.layers.Dense(action_num)(x)

    model = keras.Model(inputs=[input], outputs=[output])
    return model


if __name__ == "__main__":
    if not os.path.isdir("./models"):
        os.makedirs("./models")

    action_num = env.action_space.n
    batch_size = 8

    model = create_model(input_shape=env.observation_space.shape, action_num=action_num)
    # print(model.summary())
    # model.load_weights(f'models/-5400.50avg_0.28epsilon_50s run_seconds.h5')

    agent = DQNAgent(model, discount_rate=0.99, deque_maxlen=5000)

    EPISODES = 1000
    best_score = -np.inf
    # best_score = -227
    max_epsilon = 0.9
    total_rewards_list = []

    for episode in tqdm(range(EPISODES+1), ascii=True, unit="episodes"):
        state = env.reset()
        episode_reward = 0
        done = False

        while True:
            env.render()
            epsilon = max(max_epsilon - episode / 500, 0.1)
            action = agent.epsilon_greedy_policy(state, epsilon)
            new_state, reward, done, _ = env.step(action)
            agent.replay_memory.append((state, action, reward, new_state, done))
            state = new_state
            episode_reward += reward

            if done:
                break

        total_rewards_list.append(episode_reward)
        if episode % 10 == 0:
            average_reward = sum(total_rewards_list) / len(total_rewards_list)

            if average_reward > best_score:
                best_score = average_reward
                min_reward = min(total_rewards_list)
                max_reward = max(total_rewards_list)

                agent.model.save(f'models/{average_reward:.2f}avg_{epsilon:.2f}epsilon.h5')
                print(f"Save model by {average_reward:_>7.2f}avg__{max_reward:_>7.2f}max__{min_reward:_>7.2f}min")

            total_rewards_list = []

        agent.training_step(batch_size=batch_size)

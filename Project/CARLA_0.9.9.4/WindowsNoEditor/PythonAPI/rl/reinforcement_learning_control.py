# reinforcement_learning_control.
"""
    1. sensor.other.obstacle 用于保持车距
    2. LaneType
    3. 是否设置目标点
    4。连接线有障碍
    5. 传入多个点，拐角处给出3个以上点（3~5个点）
"""

import glob
import os
import sys
import random
import numpy as np
import cv2
import time
import math
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

import carla
from rl.CarEnv import CarEnv
from rl.DDPG.ddpg import DDPG


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


if __name__ == "__main__":
    if not os.path.isdir("./models"):
        os.makedirs("./models")

    IM_WIDTH = 800
    IM_HEIGHT = 600
    action_num = 5
    batch_size = 8
    run_seconds_per_episode = 50

    # print(model.summary())
    # model.load_weights(f'models/-10234.00min_-3670.20avg_0.37epsilon_50s run_seconds.h5')

    algo = DDPG(action_dim, state_dim, act_range, args.consecutive_frames)

    # agent = DQNAgent(model, discount_rate=0.99, deque_maxlen=5000)
    env = CarEnv(IM_HEIGHT, IM_WIDTH, show_sem_camera=True, run_seconds_per_episode=run_seconds_per_episode,
                 no_rendering_mode=False)

    EPISODES = 1000
    best_min = -np.inf
    best_average = -np.inf
    max_epsilon = 0.4
    total_rewards_list = []

    for episode in tqdm(range(EPISODES), ascii=True, unit="episodes"):
        state = env.reset()
        episode_reward = 0
        done = False

        while True:
            epsilon = max(max_epsilon - episode / EPISODES, 0.01)
            action = agent.epsilon_greedy_policy(state, epsilon)
            new_state, reward, done, _ = env.step(action)
            agent.replay_memory.append((state, action, reward, new_state, done))
            state = new_state
            episode_reward += reward

            if done:
                break

        total_rewards_list.append(episode_reward)
        if episode != 0 and episode % 10 == 0:
            average_reward = sum(total_rewards_list) / len(total_rewards_list)
            min_reward = min(total_rewards_list)
            max_reward = max(total_rewards_list)

            if min_reward > best_min or average_reward > best_average:
                if min_reward > best_min:
                    best_min = min_reward
                elif average_reward > best_average:
                    best_average = average_reward

                agent.model.save(f'models/{min_reward:.2f}min_{average_reward:.2f}avg_{epsilon:.2f}epsilon'
                                 f'_{run_seconds_per_episode}s run_seconds.h5')
                print(f"Save model.{average_reward:_>7.2f}avg__{max_reward:_>7.2f}max__{min_reward:_>7.2f}min")
            else:
                print(f"{average_reward:_>7.2f}avg__{max_reward:_>7.2f}max__{min_reward:_>7.2f}min")

            total_rewards_list = []

        agent.training_step(batch_size=batch_size)
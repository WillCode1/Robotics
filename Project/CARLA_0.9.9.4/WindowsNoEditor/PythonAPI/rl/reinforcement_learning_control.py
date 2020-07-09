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
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from rl.CarEnv import CarEnv
from rl.DDPG.ddpg import DDPG
from rl.DDPG.DDQN import DQNAgent


if __name__ == "__main__":
    if not os.path.isdir("./models"):
        os.makedirs("./models")

    IM_WIDTH = 800
    IM_HEIGHT = 600
    batch_size = 8
    EPISODES = 1000
    run_seconds_per_episode = 50
    image_shape = (IM_HEIGHT, IM_WIDTH, 3)
    state_dim = [image_shape, image_shape, 1]
    action_dim = 2  # [throttle_brake, steer]
    # print(model.summary())
    # model.load_weights(f'models/-10234.00min_-3670.20avg_0.37epsilon_50s run_seconds.h5')

    algo = DDPG(act_dim=action_dim, state_dim=state_dim, act_range=1.0)
    # agent = DQNAgent(model, discount_rate=0.99, deque_maxlen=5000)
    env = CarEnv(IM_HEIGHT, IM_WIDTH, show_sem_camera=False, run_seconds_per_episode=run_seconds_per_episode,
                 no_rendering_mode=True)

    stats = algo.play_and_train(env, batch_size=batch_size, n_episode=EPISODES)

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
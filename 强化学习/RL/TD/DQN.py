import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym
from collections import deque

# 平衡车问题
env = gym.make("CartPole-v0")
input_shape = [4]  # 观测数据
n_outputs = 2  # 可选动作

model = keras.models.Sequential([
    keras.layers.Dense(32, activation="elu", input_shape=input_shape),
    keras.layers.Dense(32, activation="elu"),
    keras.layers.Dense(n_outputs)
])


# ε-贪婪策略
def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    else:
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])


# 每个经验包含五个元素：状态，智能体选择的动作，奖励，下一个状态，一个知识是否结束的布尔值（done）。
# 需要一个小函数从接力缓存随机采样。返回的是五个NumPy数组，对应五个经验
def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones


# 使用ε-贪婪策略的单次玩游戏函数，然后将结果经验存储在replay_buffer中
def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done, info


# 不仅只根据最新的经验训练DQN，将所有经验存储在接力缓存（或接力记忆）中，每次训练迭代，从中随机采样一个批次。
# 这样可以降低训练批次中的经验相关性，可以极大的提高训练效果。
replay_buffer = deque(maxlen=2000)

batch_size = 32
discount_factor = 0.95
optimizer = keras.optimizers.Adam(lr=1e-3)
loss_fn = keras.losses.mean_squared_error


def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = model.predict(next_states)      # 两个动作的价值
    max_next_Q_values = np.max(next_Q_values, axis=1)
    # 使用公式18-7计算每个经验的状态-动作对的 目标Q-值
    target_Q_values = (rewards + (1 - dones) * discount_factor * max_next_Q_values)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        # 将DQN的输出乘以这个mask，就可以排除所有不需要的Q-值
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        # 计算损失：即有经验的状态-动作对的目标Q-值和预测Q-值的均方误差
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    # 将这些平均梯度应用于优化器：微调模型的变量
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


for episode in range(600):
    obs = env.reset()
    for step in range(200):
        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        if done:
            break
    if episode > 50:
        training_step(batch_size)

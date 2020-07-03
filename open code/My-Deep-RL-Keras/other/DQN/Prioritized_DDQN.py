import heapq
import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym


class PrioritizedExperienceQueue:
    """ Prioritized replay memory using binary heap """

    def __init__(self, maxlen=10000):
        self.maxlen = maxlen
        self.memory = []

    def size(self):
        return len(self.memory)

    def append(self, experience, TDerror):
        heapq.heappush(self.memory, (experience, -TDerror))
        if self.size() > self.maxlen:
            self.memory = self.memory[:-1]
        heapq.heapify(self.memory)

    def batch(self, batch_size):
        batch = heapq.nsmallest(batch_size, self.memory)
        batch = [e for (_, e) in batch]
        self.memory = self.memory[batch_size:]
        return batch

    def learn_single(self, state, action, reward, next_state, done):
        next_Q_value = model.predict(next_state[np.newaxis])[0]
        best_next_action = np.argmax(next_Q_value, axis=1)
        next_mask = tf.one_hot(best_next_action, n_outputs).numpy()
        next_best_Q_value = (target.predict(next_state) * next_mask).sum(axis=1)
        target_Q_value = (reward + (1 - done) * discount_rate * next_best_Q_value).reshape(-1, 1)
        # predict Q-values for starting state using the main network
        y, _ = self.predict(S)
        y_old = np.array(y)

        # predict best action in ending state using the main network
        _, A_p = self.predict(S_p)

        # predict Q-values for ending state using the target network
        y_target, _ = self.predict(S_p, target_network=True)

        # update Q[S, A]
        if terminal:
            y[A] = R
        else:
            y[A] = R + self.gamma * y_target[A_p]

        return np.abs(y_old[A] - y[A])

    def compute_TDerror(self, experience):
        state, action, reward, next_state, done = experience
        TDerror = self.learn_single(state, action, reward, next_state, done)
        return TDerror


# 平衡车问题
env = gym.make("CartPole-v0")
input_shape = [4]  # 观测数据
n_outputs = 2  # 可选动作

keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)
env.seed(42)
replay_memory = PrioritizedExperienceQueue(maxlen=2000)

model = keras.models.Sequential([
    keras.layers.Dense(32, activation="elu", input_shape=[4]),
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


def sample_experiences(batch_size):
    batch = replay_memory.batch(batch_size)
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones


def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done, info


# 拷贝在线模型，为目标模型
target = keras.models.clone_model(model)
target.set_weights(model.get_weights())

batch_size = 32
discount_rate = 0.95
optimizer = keras.optimizers.Adam(lr=1e-3)
loss_fn = keras.losses.Huber()


def training_step(batch_size):
    batch_experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = batch_experiences
    next_Q_values = model.predict(next_states)
    best_next_actions = np.argmax(next_Q_values, axis=1)
    next_mask = tf.one_hot(best_next_actions, n_outputs).numpy()
    next_best_Q_values = (target.predict(next_states) * next_mask).sum(axis=1)
    target_Q_values = (rewards + (1 - dones) * discount_rate * next_best_Q_values).reshape(-1, 1)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


rewards = []
best_score = 0

for episode in range(600):
    obs = env.reset()
    for step in range(200):
        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        if done:
            break
    rewards.append(step)
    if step > best_score:
        best_weights = model.get_weights()
        best_score = step
    print("\rEpisode: {}, Steps: {}, eps: {:.3f}".format(episode, step + 1, epsilon), end="")
    if episode > 50:
        training_step(batch_size)
    if episode % 50 == 0:
        target.set_weights(model.get_weights())
    # Alternatively, you can do soft updates at each step:
    # if episode > 50:
    # target_weights = target.get_weights()
    # online_weights = model.get_weights()
    # for index in range(len(target_weights)):
    #    target_weights[index] = 0.99 * target_weights[index] + 0.01 * online_weights[index]
    # target.set_weights(target_weights)

model.set_weights(best_weights)

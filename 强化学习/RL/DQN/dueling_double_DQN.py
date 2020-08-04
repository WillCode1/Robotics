import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym
import random
from collections import deque
from utils.plot import plot_log


class DQN:
    def __init__(self, inp_dim, out_dim, lr, tau, gamma):
        self.state_dim = inp_dim
        self.action_dim = out_dim
        self.tau = tau
        self.gamma = gamma

        self.replay_memory = deque(maxlen=2000)
        self.model = self.create_model(inp_dim, out_dim)
        self.target = self.create_model(inp_dim, out_dim)
        self.target.set_weights(self.model.get_weights())

        self.optimizer = keras.optimizers.Adam(lr=lr)
        self.loss_fn = keras.losses.mean_squared_error

    def create_model(self, state_dim, action_dim):
        K = keras.backend
        input_states = keras.layers.Input(shape=[state_dim])
        x = keras.layers.Dense(32, activation="elu")(input_states)
        x = keras.layers.Dense(32, activation="elu")(x)
        state_values = keras.layers.Dense(1)(x)
        raw_advantages = keras.layers.Dense(action_dim)(x)
        advantages = raw_advantages - K.mean(raw_advantages, axis=1, keepdims=True)
        Q_values = state_values + advantages
        model = keras.models.Model(inputs=[input_states], outputs=[Q_values])
        return model

    def epsilon_greedy_policy(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            Q_values = self.model.predict(state[np.newaxis])
            return np.argmax(Q_values[0])

    def sample_experiences(self, batch_size):
        batch = random.sample(self.replay_memory, batch_size)
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)]
        return states, actions, rewards, next_states, dones

    def play_one_step(self, env, state, epsilon):
        action = self.epsilon_greedy_policy(state, epsilon)
        next_state, reward, done, info = env.step(action)
        self.replay_memory.append((state, action, reward, next_state, done))
        return next_state, reward, done, info

    def training_step(self, batch_size=32):
        experiences = self.sample_experiences(batch_size)
        states, actions, rewards, next_states, dones = experiences
        next_Q_values = self.model.predict(next_states)
        best_next_actions = np.argmax(next_Q_values, axis=1)
        next_mask = tf.one_hot(best_next_actions, self.action_dim).numpy()
        next_best_Q_values = (self.target.predict(next_states) * next_mask).sum(axis=1)
        target_Q_values = (rewards + (1 - dones) * self.gamma * next_best_Q_values).reshape(-1, 1)
        mask = tf.one_hot(actions, self.action_dim)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def update_model(self):
        for model_weight, target_weight in zip(self.model.weights, self.target.weights):
            target_weight.assign(self.tau * model_weight + (1-self.tau) * target_weight)


def main(agent, batch_size=32, render=False):
    rewards = []
    best_score = 0

    for episode in range(1000):
        obs = env.reset()
        total_reward = 0
        for step in range(500):
            if render:
                env.render()
            epsilon = max(1 - episode / 500, 0.01)
            obs, reward, done, info = agent.play_one_step(env, obs, epsilon)
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
        if total_reward > best_score:
            best_weights = agent.model.get_weights()
            best_score = total_reward
        print("\rEpisode: {}, Rewards: {}, eps: {:.3f}".format(episode, total_reward, epsilon), end="")
        if episode > 50:
            agent.training_step(batch_size)
            agent.update_model()

    agent.model.set_weights(best_weights)
    return rewards


if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)
    env.seed(42)

    batch_size = 32
    discount_rate = 0.95
    lr = 1e-2
    tau = 0.01

    agent = DQN(state_dim, action_dim, lr=lr, tau=tau, gamma=discount_rate)
    rewards = main(agent, batch_size=batch_size, render=True)

    plot_log(1000, rewards)
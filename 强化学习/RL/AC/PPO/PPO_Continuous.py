# import wandb
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda

import gym
import argparse
import numpy as np

K = keras.backend


class Actor:
    def __init__(self, state_dim, action_dim, action_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.model = self.create_model()
        self.old_model = self.create_model()
        self.opt = keras.optimizers.Adam(args.actor_lr)

    def create_model(self):
        state_input = Input((self.state_dim,))
        x = Dense(32, activation=keras.layers.LeakyReLU(0.2))(state_input)
        x = Dense(32, activation=keras.layers.LeakyReLU(0.2))(x)
        mu = Dense(self.action_dim, activation='tanh')(x)
        mu = Lambda(lambda x: x * self.action_bound)(mu)
        std = Dense(self.action_dim, activation='softplus')(x)
        return keras.models.Model(state_input, [mu, std])

    def get_action(self, state, optimal=False):
        mu, std = self.model.predict(state[np.newaxis])
        if optimal:
            action = mu
        else:
            action = np.random.normal(loc=mu, scale=std, size=self.action_dim)
        return np.clip(action, -self.action_bound, self.action_bound)

    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, 1e-2, 1.0)
        variance = std ** 2
        pdf = 1. / tf.sqrt(2. * np.pi * variance) * tf.exp(-(action - mu) ** 2 / (2. * variance))
        log_pdf = tf.math.log(pdf + 1e-10)
        # log_pdf = -0.5 * (action - mu) ** 2 / variance - 0.5 * tf.math.log(variance * 2 * np.pi)
        return tf.reduce_sum(log_pdf, 1, keepdims=True)

    def compute_loss(self, states, actions, gaes):
        mu, std = self.model(states)
        old_mu, old_std = self.old_model.predict(states)
        log_old_policy = self.log_pdf(old_mu, old_std, actions)
        log_new_policy = self.log_pdf(mu, std, actions)
        # log_old_policy = tf.stop_gradient(log_old_policy)
        ratio = tf.exp(log_new_policy - log_old_policy)

        # 'ppo1'
        # self.tflam = tf.placeholder(tf.float32, None, 'lambda')
        # kl = tf.distributions.kl_divergence(old_nd, nd)
        # self.kl_mean = tf.reduce_mean(kl)
        # self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))

        # 'ppo2'
        gaes = tf.stop_gradient(gaes)
        clipped_ratio = tf.clip_by_value(ratio, 1.0-args.clip_ratio, 1.0+args.clip_ratio)
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
        return tf.reduce_mean(surrogate)

    def train(self, states, actions, gaes):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(states, actions, gaes)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Critic:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.model = self.create_model()
        self.model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=args.critic_lr))

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(32, activation=keras.layers.LeakyReLU(0.2)),
            Dense(32, activation=keras.layers.LeakyReLU(0.2)),
            Dense(1)
        ])


class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]

        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound)
        self.critic = Critic(self.state_dim)

    def update_actor(self):
        for model_weight, target_weight in zip(self.actor.model.weights, self.actor.old_model.weights):
            # target_weight.assign(model_weight * 0.1 + target_weight * 0.9)
            target_weight.assign(model_weight)

    def gae_target(self, rewards, v_values, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        next_v_val = 0

        if not done:
            next_v_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            advantage = rewards[k] + args.gamma * next_v_val - v_values[k]
            gae_cumulative = args.gamma * args.lmbda * gae_cumulative + advantage
            gae[k] = gae_cumulative
            td_targets[k] = gae[k] + v_values[k]
            next_v_val = v_values[k]
        return gae, td_targets

    def train(self, max_episodes=1000):
        for ep in range(max_episodes):
            state_batch = []
            action_batch = []
            reward_batch = []

            episode_reward, done = 0, False
            state = self.env.reset()

            while not done:
                # self.env.render()
                action = self.actor.get_action(state)
                next_state, reward, done, _ = self.env.step(action)

                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append([(reward+8)/8])

                if len(state_batch) >= args.update_interval or done:
                    states = np.array(state_batch)
                    actions = np.array(action_batch)
                    rewards = np.array(reward_batch)

                    v_values = self.critic.model.predict(states)
                    next_v_value = self.critic.model.predict(next_state[np.newaxis])
                    gaes, td_targets = self.gae_target(rewards, v_values, next_v_value, done)

                    for epoch in range(args.epochs):
                        actor_loss = self.actor.train(states, actions, gaes)
                        critic_loss = self.critic.model.train_on_batch(states, td_targets)
                    self.update_actor()

                    state_batch = []
                    action_batch = []
                    reward_batch = []

                episode_reward += reward
                state = next_state

            print('EP{} EpisodeReward={}'.format(ep, episode_reward))
            # wandb.log({'Reward': episode_reward})


if __name__ == "__main__":
    tf.keras.backend.set_floatx('float64')
    # wandb.init(name='PPO', project="deep-rl-tf2")

    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--update_interval', type=int, default=128)
    parser.add_argument('--actor_lr', type=float, default=1.5e-3)
    parser.add_argument('--critic_lr', type=float, default=3e-3)
    parser.add_argument('--clip_ratio', type=float, default=0.1)
    parser.add_argument('--lmbda', type=float, default=0.95)
    parser.add_argument('--epochs', type=int, default=3)    # 大于5，容易过拟合，不收敛

    args = parser.parse_args()

    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    agent = Agent(env)
    agent.train(max_episodes=1000)
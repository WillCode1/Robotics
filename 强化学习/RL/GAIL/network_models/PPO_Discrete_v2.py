# import wandb
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense

import gym
import argparse
import numpy as np


class PPO2:
    def __init__(self, state_dim, action_dim, action_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.model = self.create_model()
        self.old_model = self.create_model()
        self.opt = keras.optimizers.Adam(args.lr)

    def create_model(self):
        state_input = Input((self.state_dim,))
        x = Dense(32, activation=keras.layers.LeakyReLU(0.2))(state_input)
        state_feature = Dense(32, activation=keras.layers.LeakyReLU(0.2))(x)

        # critic
        critic = Dense(32, activation=keras.layers.LeakyReLU(0.2))(state_feature)
        critic = Dense(32, activation=keras.layers.LeakyReLU(0.2))(critic)
        value = Dense(1)(critic)

        # actor
        actor = Dense(32, activation=keras.layers.LeakyReLU(0.2))(state_feature)
        actor = Dense(32, activation=keras.layers.LeakyReLU(0.2))(actor)
        mu = Dense(self.action_dim, activation='tanh')(actor)
        mu = Lambda(lambda x: x * self.action_bound)(mu)
        std = Dense(self.action_dim, activation='softplus')(actor)
        return keras.models.Model(state_input, [mu, std, value])

    def get_action(self, state, optimal=False):
        mu, std, _ = self.model.predict(state[np.newaxis])
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

    def train_on_batch(self, states, actions, gaes, td_targets):
        with tf.GradientTape() as tape:
            mu, std, v_pred = self.model(states)

            # critic_loss
            td_targets = tf.stop_gradient(td_targets)
            critic_loss = tf.reduce_mean(keras.losses.mean_squared_error(td_targets, v_pred))

            # actor_loss
            old_mu, old_std, _ = self.old_model.predict(states)
            log_old_policy = self.log_pdf(old_mu, old_std, actions)
            log_new_policy = self.log_pdf(mu, std, actions)
            ratio = tf.exp(log_new_policy - log_old_policy)

            gaes = tf.stop_gradient(gaes)
            clipped_ratio = tf.clip_by_value(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
            surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
            actor_loss = tf.reduce_mean(surrogate)

            # entropy
            std = tf.clip_by_value(std, 1e-10, 10.0)
            act_probs = 1. / tf.sqrt(2. * np.pi * std ** 2) * tf.exp(-(actions - mu) ** 2 / (2. * std ** 2))
            entropy = -tf.reduce_sum(act_probs * tf.math.log(tf.clip_by_value(act_probs, 1e-10, 1.0)))
            loss = critic_loss + 0.5 * actor_loss - args.entropy_ratio * entropy
            # loss = 0.5 * critic_loss + actor_loss - args.entropy_ratio * entropy
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Agent:
    def __init__(self, env, args):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.args = args

        self.actor = Actor(self.state_dim, self.action_dim, args)
        self.critic = Critic(self.state_dim, args)

    def update_actor(self):
        for model_weight, target_weight in zip(self.actor.model.weights, self.actor.old_model.weights):
            target_weight.assign(model_weight)

    def save_model(self, path):
        self.actor.model.save_weights(path + 'actor.h5')

    def gae_target(self, rewards, v_values, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        next_v_val = 0

        if not done:
            next_v_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            advantage = rewards[k] + self.args.gamma * next_v_val - v_values[k]
            gae_cumulative = self.args.gamma * self.args.lmbda * gae_cumulative + advantage
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
                reward_batch.append([reward])

                if len(state_batch) >= self.args.batch_size or done:
                    states = np.array(state_batch)
                    actions = np.array(action_batch)
                    rewards = np.array(reward_batch)

                    v_values = self.critic.model.predict(states)
                    next_v_value = self.critic.model.predict(next_state[np.newaxis])
                    gaes, td_targets = self.gae_target(rewards, v_values, next_v_value, done)

                    for epoch in range(self.args.epochs):
                        actor_loss = self.actor.train(states, actions, gaes)
                        critic_loss = self.critic.model.train_on_batch(states, td_targets)
                    self.update_actor()
                    self.save_model("models/")

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
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--actor_lr', type=float, default=5e-4)
    parser.add_argument('--critic_lr', type=float, default=1e-3)
    parser.add_argument('--clip_ratio', type=float, default=0.1)
    parser.add_argument('--lmbda', type=float, default=0.95)
    parser.add_argument('--epochs', type=int, default=3)

    args = parser.parse_args()

    import os
    if not os.path.isdir("./models"):
        os.makedirs("./models")

    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    agent = Agent(env, args)
    agent.train(max_episodes=1000)
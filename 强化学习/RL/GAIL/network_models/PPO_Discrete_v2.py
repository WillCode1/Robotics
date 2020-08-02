# import wandb
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense

import gym
import argparse
import numpy as np


class PPO2:
    def __init__(self, state_dim, action_dim, args):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.create_model()
        self.old_model = self.create_model()
        self.opt = keras.optimizers.Adam(args.lr)
        self.args = args

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
        action_probs = Dense(self.action_dim, activation='softmax')(actor)
        return keras.models.Model(state_input, [action_probs, value])

    def get_action(self, state):
        probs, _ = self.model.predict(np.reshape(state, [1, self.state_dim]))
        action = np.random.choice(self.action_dim, p=probs[0])
        return action

    def train_on_batch(self, states, actions, gaes, td_targets):
        actions = tf.one_hot(actions, self.action_dim)
        actions = tf.reshape(actions, [-1, self.action_dim])
        actions = tf.cast(actions, tf.float64)

        with tf.GradientTape() as tape:
            probs, v_pred = self.model(states)

            # critic_loss
            td_targets = tf.stop_gradient(td_targets)
            critic_loss = tf.reduce_mean(keras.losses.mean_squared_error(td_targets, v_pred))

            # actor_loss
            old_probs, _ = self.old_model.predict(states)
            old_log_p = tf.math.log(tf.reduce_sum(old_probs * actions))
            old_log_p = tf.stop_gradient(old_log_p)
            log_p = tf.math.log(tf.reduce_sum(probs * actions))
            ratio = tf.math.exp(log_p - old_log_p)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.args.clip_ratio, 1 + self.args.clip_ratio)
            gaes = tf.stop_gradient(gaes)
            surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
            actor_loss = tf.reduce_mean(surrogate)

            # entropy
            entropy = -tf.reduce_sum(probs * tf.math.log(tf.clip_by_value(probs, 1e-10, 1.0)))
            loss = critic_loss + 0.5 * actor_loss - self.args.entropy_ratio * entropy
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

        self.ppo = PPO2(self.state_dim, self.action_dim, args)

    def update_actor(self):
        for model_weight, target_weight in zip(self.ppo.model.weights, self.ppo.old_model.weights):
            target_weight.assign(model_weight)

    def save_model(self, path):
        self.ppo.model.save_weights(path + 'actor.h5')

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
                action = self.ppo.get_action(state)
                next_state, reward, done, _ = self.env.step(action)

                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append([reward])

                if len(state_batch) >= self.args.batch_size or done:
                    states = np.array(state_batch)
                    actions = np.array(action_batch)
                    rewards = np.array(reward_batch)

                    _, v_values = self.ppo.model.predict(states)
                    _, next_v_value = self.ppo.model.predict(next_state[np.newaxis])
                    gaes, td_targets = self.gae_target(rewards, v_values, next_v_value, done)

                    for epoch in range(self.args.epochs):
                        loss = self.ppo.train_on_batch(states, actions, gaes, td_targets)
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
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--clip_ratio', type=float, default=0.1)
    parser.add_argument('--entropy_ratio', type=float, default=0.01)
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
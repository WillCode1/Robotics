"""
Implementation of PPO
ref: Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
ref: https://github.com/Jiankai-Sun/Proximal-Policy-Optimization-in-Pytorch/blob/master/ppo.py
ref: https://github.com/openai/baselines/tree/master/baselines/ppo2

NOTICE:
    `Tensor2` means 2D-Tensor (num_samples, num_dims)
"""

import gym
# import torch
# import torch.nn as nn
# import torch.optim as opt
# from torch import Tensor
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from collections import namedtuple
import os
import pandas as pd
import numpy as np
import math

Transition = namedtuple('Transition', ('state', 'value', 'action', 'logproba', 'mask', 'next_state', 'reward'))
EPS = 1e-10
if not os.path.isdir("models/PPO"):
    os.makedirs("models/PPO")


class args(object):
    env_name = 'Hopper-v2'
    seed = 1234
    num_episode = 2000
    batch_size = 2048
    max_step_per_round = 2000
    gamma = 0.995
    lamda = 0.97
    log_num_episode = 1
    num_epoch = 10
    minibatch_size = 256
    clip = 0.2
    loss_coeff_value = 0.5
    loss_coeff_entropy = 0.01
    lr = 3e-4
    num_parallel_run = 5
    # tricks
    schedule_adam = 'linear'
    schedule_clip = 'linear'
    layer_norm = True
    state_norm = True
    advantage_norm = True
    lossvalue_norm = True


class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._Mean = np.zeros(shape)
        self._S = np.zeros(shape)   # 误差平方和

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._Mean.shape
        if self._n == 0:
            self._Mean[...] = x
        else:
            oldM = self._Mean.copy()
            self._Mean[...] = oldM + (x - oldM) / (self._n + 1)
            # 近似求法
            self._S[...] = self._S + (x - oldM) * (x - self._Mean)
        self._n += 1

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._Mean

    @property
    def variance(self):
        return self._S / self._n if self._n > 1 else np.square(self._Mean)

    @property
    def std(self):
        return np.sqrt(self.variance)

    @property
    def shape(self):
        return self._Mean.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """
    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip
        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape


class ActorCritic:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.env.action_space.high[0]

        state_input = Input((self.state_dim,))
        x = Dense(64, activation=keras.layers.LeakyReLU(0.2))(state_input)
        state_feature = Dense(32, activation=keras.layers.LeakyReLU(0.2))(x)

        # critic
        critic = Dense(32, activation=keras.layers.LeakyReLU(0.2))(state_feature)
        value = Dense(1)(critic)

        # actor
        actor = Dense(32, activation=keras.layers.LeakyReLU(0.2))(state_feature)
        mu = Dense(self.action_dim, activation='tanh')(actor)
        mu = Lambda(lambda x: x * self.action_bound)(mu)
        log_std = Dense(self.action_dim, activation='softplus')(actor)
        self.model = keras.models.Model(state_input, [mu, log_std, value])
        self.opt = keras.optimizers.Adam(args.lr)

    def select_action(self, mean, log_std, is_train=True, return_logproba=False):
        std = tf.exp(log_std)
        if is_train:
            action = mean
        else:
            action = np.random.normal(loc=mean, scale=std, size=self.action_dim)

        if return_logproba:
            logproba = self._normal_logproba(action, mean, log_std, std)
        return np.clip(action, -self.action_bound, self.action_bound), logproba

    @staticmethod
    def _normal_logproba(action, mean, log_std, std=None):
        if std is None:
            std = tf.exp(log_std)

        variance = std**2
        logproba = - 0.5 * math.log(2 * math.pi) - log_std - (action - mean)**2 / (2 * variance)
        return tf.reduce_sum(logproba, 1, keepdims=True)


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)


def ppo(args):
    env = gym.make(args.env_name)
    env.seed(args.seed)
    network = ActorCritic(env)
    running_state = ZFilter((env.observation_space.shape[0],), clip=5.0)

    # record average 1-round cumulative reward in every episode
    reward_record = []
    global_steps = 0

    lr_now = args.lr
    clip_now = args.clip

    for i_episode in range(args.num_episode):
        # step1: perform current policy to collect trajectories
        # this is an on-policy method!
        memory = Memory()
        num_steps = 0
        reward_list = []
        len_list = []
        while num_steps < args.batch_size:
            state = env.reset()
            if args.state_norm:
                state = running_state(state)
            reward_sum = 0
            for t in range(args.max_step_per_round):
                mean, log_std, value = network.model.predict(state[np.newaxis])
                action, logproba = network.select_action(mean, log_std, return_logproba=True)
                action = action.data.numpy()[0]
                logproba = logproba.data.numpy()[0]
                next_state, reward, done, _ = env.step(action)
                reward_sum += reward
                if args.state_norm:
                    next_state = running_state(next_state)
                mask = 0 if done else 1

                memory.push(state, value, action, logproba, mask, next_state, reward)

                if done:
                    break

                state = next_state

            num_steps += (t + 1)
            global_steps += (t + 1)
            reward_list.append(reward_sum)
            len_list.append(t + 1)
        reward_record.append({
            'episode': i_episode,
            'steps': global_steps,
            'meanepreward': np.mean(reward_list),
            'meaneplen': np.mean(len_list)})

        batch = memory.sample()
        batch_size = len(memory)

        # step2: extract variables from trajectories
        rewards = Tensor(batch.reward)
        values = Tensor(batch.value)
        masks = Tensor(batch.mask)
        actions = Tensor(batch.action)
        states = Tensor(batch.state)
        oldlogproba = Tensor(batch.logproba)

        returns = Tensor(batch_size)
        deltas = Tensor(batch_size)
        advantages = Tensor(batch_size)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(batch_size)):
            returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values[i]
            # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
            advantages[i] = deltas[i] + args.gamma * args.lamda * prev_advantage * masks[i]

            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]
        if args.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)

        for i_epoch in range(int(args.num_epoch * batch_size / args.minibatch_size)):
            # sample from current batch
            minibatch_ind = np.random.choice(batch_size, args.minibatch_size, replace=False)
            minibatch_states = states[minibatch_ind]
            minibatch_actions = actions[minibatch_ind]
            minibatch_oldlogproba = oldlogproba[minibatch_ind]
            minibatch_newlogproba = network.get_logproba(minibatch_states, minibatch_actions)
            minibatch_advantages = advantages[minibatch_ind]
            minibatch_returns = returns[minibatch_ind]
            minibatch_newvalues = network._forward_critic(minibatch_states).flatten()

            ratio = torch.exp(minibatch_newlogproba - minibatch_oldlogproba)
            surr1 = ratio * minibatch_advantages
            surr2 = ratio.clamp(1 - clip_now, 1 + clip_now) * minibatch_advantages
            loss_surr = - torch.mean(torch.min(surr1, surr2))

            # not sure the value loss should be clipped as well
            # clip example: https://github.com/Jiankai-Sun/Proximal-Policy-Optimization-in-Pytorch/blob/master/ppo.py
            # however, it does not make sense to clip score-like value by a dimensionless clipping parameter
            # moreover, original paper does not mention clipped value
            if args.lossvalue_norm:
                minibatch_return_6std = 6 * minibatch_returns.std()
                loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2)) / minibatch_return_6std
            else:
                loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2))

            loss_entropy = torch.mean(torch.exp(minibatch_newlogproba) * minibatch_newlogproba)

            total_loss = loss_surr + args.loss_coeff_value * loss_value + args.loss_coeff_entropy * loss_entropy
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if args.schedule_clip == 'linear':
            ep_ratio = 1 - (i_episode / args.num_episode)
            clip_now = args.clip * ep_ratio

        if args.schedule_adam == 'linear':
            ep_ratio = 1 - (i_episode / args.num_episode)
            lr_now = args.lr * ep_ratio
            # set learning rate
            # ref: https://stackoverflow.com/questions/48324152/

        if i_episode % args.log_num_episode == 0:
            print('Finished episode: {} Reward: {:.4f} total_loss = {:.4f} = {:.4f} + {} * {:.4f} + {} * {:.4f}' \
                  .format(i_episode, reward_record[-1]['meanepreward'], total_loss.data, loss_surr.data,
                          args.loss_coeff_value,
                          loss_value.data, args.loss_coeff_entropy, loss_entropy.data))
            print('-----------------')

    return reward_record


def train(args):
    record_dfs = []
    for i in range(args.num_parallel_run):
        args.seed += 1
        reward_record = pd.DataFrame(ppo(args))
        reward_record['#parallel_run'] = i
        record_dfs.append(reward_record)
    record_dfs = pd.concat(record_dfs, axis=0)
    # record_dfs.to_csv(joindir(RESULT_DIR, 'ppo-record-{}.csv'.format(args.env_name)))


if __name__ == '__main__':
    for env in ['Walker2d-v2', 'Swimmer-v2', 'Hopper-v2', 'Humanoid-v2', 'HalfCheetah-v2', 'Reacher-v2']:
        args.env_name = env
        train(args)

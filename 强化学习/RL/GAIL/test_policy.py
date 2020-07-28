import argparse
import gym
import numpy as np
from GAIL.network_models.PPO_Discrete import Agent


def argparser():
    # tf.keras.backend.set_floatx('float64')

    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--update_interval', type=int, default=5)
    parser.add_argument('--actor_lr', type=float, default=0.0005)
    parser.add_argument('--critic_lr', type=float, default=0.001)
    parser.add_argument('--clip_ratio', type=float, default=0.1)
    parser.add_argument('--lmbda', type=float, default=0.95)
    parser.add_argument('--epochs', type=int, default=3)
    return parser.parse_args()


if __name__ == '__main__':
    args = argparser()

    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    agent = Agent(env, args)

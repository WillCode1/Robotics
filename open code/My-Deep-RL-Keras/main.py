""" Deep RL Algorithms for OpenAI Gym environments
"""

import os
import sys
import gym
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
# from A2C.a2c import A2C
# from A3C.a3c import A3C
from DDQN.ddqn import DDQN
from DDPG.ddpg import DDPG

from utils.atari_environment import AtariEnvironment
from utils.continuous_environments import Environment
from utils.networks import get_session

gym.logger.set_level(40)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(args=None):
    # Environment Initialization
    if args.is_atari:
        # Atari Environment Wrapper
        env = AtariEnvironment(args)
        state_dim = env.get_state_size()
        action_dim = env.get_action_size()
    elif args.type == "DDPG":
        # Continuous Environments Wrapper
        env = Environment(gym.make(args.env), args.consecutive_frames)
        env.reset()
        state_dim = env.get_state_size()
        action_space = gym.make(args.env).action_space
        action_dim = action_space.high.shape[0]
        act_range = action_space.high
    else:
        # Standard Environments
        env = Environment(gym.make(args.env), args.consecutive_frames)
        env.reset()
        state_dim = env.get_state_size()
        action_dim = gym.make(args.env).action_space.n

    # Pick algorithm to train
    if args.type == "DDQN":
        algo = DDQN(action_dim, state_dim, args)
    elif args.type == "DDPG":
        algo = DDPG(action_dim, state_dim, act_range, args.consecutive_frames)

    # Train
    stats = algo.train(env, args, summary_writer)

    # Export results to CSV
    if args.gather_stats:
        df = pd.DataFrame(np.array(stats))
        df.to_csv(args.type + "/logs.csv", header=['Episode', 'Mean', 'Stddev'], float_format='%10.5f')

    # Save weights and close environments
    exp_dir = '{}/models/'.format(args.type)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    export_path = '{}{}_ENV_{}_NB_EP_{}_BS_{}'.format(exp_dir,
                                                      args.type,
                                                      args.env,
                                                      args.nb_episodes,
                                                      args.batch_size)

    algo.save_weights(export_path)
    env.env.close()


if __name__ == "__main__":
    main()

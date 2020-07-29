import argparse
import gym
import numpy as np
import tensorflow as tf
from GAIL.network_models.PPO_Discrete import Agent
from GAIL.network_models.discriminator import Discriminator


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/train/gail')
    parser.add_argument('--savedir', help='save directory', default='trained_models/gail')
    parser.add_argument('--gamma', default=0.95)
    parser.add_argument('--iteration', default=int(1e4))
    return parser.parse_args()


def main(args):
    env = gym.make('CartPole-v1')
    env.seed(42)

    agent = Agent(env, args)
    discriminator = Discriminator(env, lr=0.01)

    # 得到专家的观测和行动
    expert_observations = np.genfromtxt('trajectory/observations.csv')
    expert_actions = np.genfromtxt('trajectory/actions.csv', dtype=np.int32)

    obs = env.reset()
    success_num = 0

    for iteration in range(args.iteration):
        observations = []
        actions = []
        rewards = []
        v_preds = []
        run_policy_steps = 0

        while True:
            run_policy_steps += 1
            obs = np.stack([obs]).astype(dtype=np.float32)
            act, v_pred = Policy.act(obs=obs, stochastic=True)

            act = np.asscalar(act)
            v_pred = np.asscalar(v_pred)

            next_obs, reward, done, info = env.step(act)

            observations.append(obs)
            actions.append(act)
            rewards.append(reward)
            v_preds.append(v_pred)

            if done:
                next_obs = np.stack([next_obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                _, v_pred = Policy.act(obs=next_obs, stochastic=True)
                v_preds_next = v_preds[1:] + [np.asscalar(v_pred)]
                obs = env.reset()
                break
            else:
                obs = next_obs

        if sum(rewards) >= 475:
            success_num += 1
            if success_num >= 100:
                saver.save(sess, args.savedir + '/model.ckpt')
                print('Clear!! Model saved.')
                break
        else:
            success_num = 0

        observations = np.reshape(observations, newshape=[-1] + list(env.observation_space.shape))
        actions = np.array(actions).astype(dtype=np.int32)

        for i in range(2):
            discriminator.train(expert_s=expert_observations, expert_a=expert_actions,
                                agent_s=observations, agent_a=actions)

        d_rewards = discriminator.get_rewards(agent_s=observations, agent_a=actions)
        d_rewards = np.reshape(d_rewards, newshape=[-1]).astype(dtype=np.float32)

        gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)
        gaes = np.array(gaes).astype(dtype=np.float32)
        # gaes = (gaes - gaes.mean()) / gaes.std()
        v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

        # train policy
        inp = [observations, actions, gaes, d_rewards, v_preds_next]
        PPO.assign_policy_parameters()
        for epoch in range(6):
            sample_indices = np.random.randint(low=0, high=observations.shape[0], size=32)
            sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]
            PPO.train(obs=sampled_inp[0], actions=sampled_inp[1], gaes=sampled_inp[2],
                      rewards=sampled_inp[3], v_preds_next=sampled_inp[4])


if __name__ == '__main__':
    args = argparser()
    main(args)

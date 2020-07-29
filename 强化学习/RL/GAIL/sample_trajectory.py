import argparse
import gym
import numpy as np
from GAIL.network_models.PPO_Discrete import Actor


# noinspection PyTypeChecker
def open_file_and_save(file_path, data):
    """
    :param file_path: type==string
    :param data:
    """
    try:
        with open(file_path, 'ab') as f_handle:
            np.savetxt(f_handle, data, fmt='%s')
    except FileNotFoundError:
        with open(file_path, 'wb') as f_handle:
            np.savetxt(f_handle, data, fmt='%s')


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor_lr', type=float, default=0.0005)
    parser.add_argument('--model', help='filename of model to test', default='models/actor.h5')
    parser.add_argument('--iteration', default=10, type=int)

    return parser.parse_args()


def main(args):
    env = gym.make('CartPole-v1')
    env.seed(0)
    ob_space = env.observation_space
    actor = Actor(env.observation_space.shape[0], env.action_space.n, args)
    actor.model.load_weights('models/' + 'actor.h5')

    obs = env.reset()

    for iteration in range(args.iteration):
        observations = []
        actions = []
        run_steps = 0
        while True:
            run_steps += 1
            # prepare to feed placeholder Policy.obs
            obs = np.stack([obs]).astype(dtype=np.float32)
            act = actor.get_action(obs)

            observations.append(obs)
            actions.append(act)

            next_obs, reward, done, info = env.step(act)

            if done:
                print(run_steps)
                obs = env.reset()
                break
            else:
                obs = next_obs

        observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
        actions = np.array(actions).astype(dtype=np.int32)

        open_file_and_save('trajectory/observations.csv', observations)
        open_file_and_save('trajectory/actions.csv', actions)


if __name__ == '__main__':
    args = argparser()
    main(args)
import gym
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from DDPG.network.ddpg import DDPG
from tqdm import tqdm
from utils.stats import gather_stats
from utils.networks import OrnsteinUhlenbeckProcess


def play_and_train(agent, env, batch_size=32, n_episode=1000, load_model=False,
                   if_gather_stats=False, if_debug=False):
    if load_model:
        agent.load_weights(agent.model_path)

    results = []
    mean_reward = -np.inf

    tqdm_e = tqdm(range(n_episode), desc='Score', leave=True, unit=" episodes")
    for episode in tqdm_e:
        time, total_reward, done = 0, 0, False
        state = env.reset()
        noise = OrnsteinUhlenbeckProcess(size=agent.act_dim)

        while not done:
            action = agent.policy_action(state)
            action = np.clip(action + noise.generate(time), -agent.act_range, agent.act_range)
            new_state, reward, done, _ = env.step(action)
            agent.memorize(state, action, reward, new_state, done)
            state = new_state
            total_reward += reward
            time += 1

        if len(agent.buffer) >= batch_size:
            agent.train(batch_size, if_debug=if_debug)

        if episode != 0 and episode % 10 == 0:
            agent.save_weights(agent.model_path)
        if episode != 0 and episode % 1000 == 0:
            agent.set_learning_rate(lr_decay=0.1)

            # Gather stats every episode for plotting
            mean, stdev = gather_stats(agent, env)
            print('episode {0}: mean={1}'.format(episode, mean))
            # if mean > mean_reward:
            #     mean_reward = mean
            #     agent.save_weights(agent.model_path)
            if if_gather_stats:
                results.append([episode, mean, stdev])

        # Display score
        tqdm_e.set_description("Score: " + str(total_reward))
        tqdm_e.refresh()

    return results


if __name__ == "__main__":
    if not os.path.isdir("./models"):
        os.makedirs("./models")

    env = gym.make("Pendulum-v0")
    state = env.reset()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    batch_size = 128
    EPISODES = 10000
    lr = 1e-2
    tau = 0.01
    gamma = 0.95

    soft_update = True
    load_model = True
    if_debug = False

    agent = DDPG(act_dim=action_dim, state_dim=state_dim, model_path=f'models/', hidden_layers=[64, 64],
                 soft_update=soft_update, buffer_size=5000, act_range=2.0, lr=lr, tau=tau, gamma=gamma)

    stats = play_and_train(agent, env, batch_size=batch_size, n_episode=EPISODES, load_model=load_model,
                           if_gather_stats=False, if_debug=if_debug)

    # Export results to CSV
    # df = pd.DataFrame(np.array(stats))
    # df.to_csv("/logs.csv", header=['Episode', 'Mean', 'Stddev'], float_format='%10.5f')

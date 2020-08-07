import gym
import os
import numpy as np
from test_env.Humanoid.DDPG.network.ddpg import DDPG
from tqdm import tqdm
from utils.stats import gather_stats
from utils.networks import OrnsteinUhlenbeckProcess


def play_and_train(agent, env, batch_size=32, n_episode=1000, load_model=False,
                   if_gather_stats=False, if_render=False, if_debug=False):
    if load_model:
        agent.load_weights(agent.model_path)

    history = {'episode': [], 'episode_reward': []}
    # history = {'episode': [], 'episode_reward': [], 'loss': []}

    tqdm_e = tqdm(range(n_episode), desc='Score', leave=True, unit=" episodes")
    for episode in tqdm_e:
        time, total_reward, done = 0, 0, False
        state = env.reset()
        noise = OrnsteinUhlenbeckProcess(size=agent.act_dim)

        while not done:
            if if_render:
                env.render()
            action = agent.policy_action(state)
            action = np.clip(action + noise.generate(time), -agent.act_range, agent.act_range)
            new_state, reward, done, _ = env.step(action)
            agent.memorize(state, action, reward, new_state, done)
            state = new_state
            total_reward += reward
            time += 1

            if if_debug:
                print("action:{}".format(action[0]))

        if len(agent.buffer) >= batch_size:
            agent.train(batch_size, if_debug=if_debug)

        if episode != 0 and episode % 100 == 0:
            agent.save_weights(agent.model_path)

            # Gather stats every episode for plotting
            mean, stdev = gather_stats(agent, env, 20)
            print('episode {0}: mean={1}'.format(episode, mean))

        if if_gather_stats:
            history['episode'].append(episode)
            history['episode_reward'].append(mean)

        # Display score
        tqdm_e.set_description("Score: " + str(total_reward))
        tqdm_e.refresh()

    return history


if __name__ == "__main__":
    if not os.path.isdir("models"):
        os.makedirs("models")

    env = gym.make("Humanoid-v2")
    state = env.reset()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    act_range = env.action_space.high[0]

    batch_size = 128
    EPISODES = 2000
    lr = 1e-1
    tau = 0.01
    gamma = 0.95

    load_model = False
    if_debug = False

    agent = DDPG(act_dim=action_dim, state_dim=state_dim, model_path=f'models/', hidden_layers=[64, 64],
                 buffer_size=5000, act_range=act_range, lr=lr, tau=tau, gamma=gamma)

    history = play_and_train(agent, env, batch_size=batch_size, n_episode=EPISODES, load_model=load_model,
                             if_gather_stats=False, if_render=True, if_debug=if_debug)
    env.close()

    # Export results to CSV
    # df = pd.DataFrame(np.array(stats))
    # df.to_csv("/logs.csv", header=['Episode', 'Mean', 'Stddev'], float_format='%10.5f')

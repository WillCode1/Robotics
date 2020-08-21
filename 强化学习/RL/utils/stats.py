import numpy as np


def gather_stats(agent, env, episodes):
    stats_rewards = []
    for k in range(episodes):
        old_state = env.reset()
        reward_sum, done = 0, False
        while not done:
            a = agent.policy_action(old_state)
            old_state, reward, done, _ = env.step(a)
            reward_sum += reward
        stats_rewards.append(reward_sum)
    return np.mean(np.array(stats_rewards)), np.std(np.array(stats_rewards))


def run_game(agent, env, if_render=True):
    while True:
        reward_sum, done = 0, False
        old_state = env.reset()
        while not done:
            if if_render:
                env.render()
            a = agent.policy_action(old_state)
            old_state, reward, done, _ = env.step(a)
            reward_sum += reward
        print(reward_sum)

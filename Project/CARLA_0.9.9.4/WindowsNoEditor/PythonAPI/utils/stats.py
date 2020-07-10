import numpy as np


def gather_stats(agent, env):
    """ Compute average rewards over 10 episodes
  """
    score = []
    for k in range(10):
        old_state = env.reset()
        total_reward, done = 0, False
        while not done:
            a = agent.policy_action(old_state)
            old_state, reward, done, _ = env.step(a)
            total_reward += reward
        score.append(total_reward)
    return np.mean(np.array(score)), np.std(np.array(score))

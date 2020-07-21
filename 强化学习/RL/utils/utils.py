import numpy as np


def gather_stats(agent, env):
    """ Compute average rewards over 10 episodes
  """
    score = []
    for k in range(100):
        old_state = env.reset()
        total_reward, done = 0, False
        while not done:
            a = agent.policy_action(old_state)
            old_state, reward, done, _ = env.step(a)
            total_reward += reward
        score.append(total_reward)
    return np.mean(np.array(score)), np.std(np.array(score))


class OrnsteinUhlenbeckProcess(object):
    """ Ornstein-Uhlenbeck Noise (original code by @slowbull)
    """

    def __init__(self, theta=0.15, mu=0, sigma=1, x0=0, dt=1e-2, n_steps_annealing=100, size=1):
        self.theta = theta
        self.sigma = sigma
        self.n_steps_annealing = n_steps_annealing
        self.sigma_step = - self.sigma / float(self.n_steps_annealing)
        self.x0 = x0
        self.mu = mu
        self.dt = dt
        self.size = size

    def generate(self, step):
        sigma = max(0, self.sigma_step * step + self.sigma)
        x = self.x0 + self.theta * (self.mu - self.x0) * self.dt + sigma * np.sqrt(self.dt) * np.random.normal(
            size=self.size)
        self.x0 = x
        return x

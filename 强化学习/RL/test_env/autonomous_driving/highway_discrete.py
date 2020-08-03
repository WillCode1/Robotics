import highway_env
import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym
from test_env.autonomous_driving.network.dqn import DQN
from utils.plot import plot_log


def main(agent, batch_size=32, render=False):
    rewards = []
    best_score = 0

    for episode in range(1000):
        obs = env.reset()
        total_reward = 0
        for step in range(500):
            if render:
                env.render()
            epsilon = max(1 - episode / 500, 0.5)
            obs, reward, done, info = agent.play_one_step(env, obs, epsilon)
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
        if total_reward > best_score:
            best_weights = agent.model.get_weights()
            best_score = total_reward
        print("\rEpisode: {}, Rewards: {}, eps: {:.3f}".format(episode, total_reward, epsilon), end="")
        if episode > 50:
            agent.training_step(batch_size)
            agent.update_model()

    agent.model.set_weights(best_weights)
    return rewards


if __name__ == "__main__":
    env = gym.make("highway-v0")

    state_dim = env.observation_space.shape
    action_dim = env.action_space.n

    keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)
    env.seed(42)

    batch_size = 32
    discount_rate = 0.95
    lr = 1e-2
    tau = 0.01

    agent = DQN(state_dim, action_dim, lr=lr, tau=tau, gamma=discount_rate)
    rewards = main(agent, batch_size=batch_size, render=True)

    plot_log(1000, rewards)

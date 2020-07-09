import sys
import numpy as np

from tqdm import tqdm
from rl.DDPG.actor import Actor
from rl.DDPG.critic import Critic
from utils.stats import gather_stats
from utils.networks import tfSummary, OrnsteinUhlenbeckProcess
from utils.memory_buffer import MemoryBuffer
import random
from collections import deque


class DDPG:
    """ Deep Deterministic Policy Gradient (DDPG) Helper Class
    """

    def __init__(self, act_dim, state_dim, act_range, k=1, buffer_size=5000, gamma=0.99, lr=0.00005, tau=0.001):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.act_range = act_range
        self.state_dim = state_dim
        self.gamma = gamma
        self.lr = lr
        # Create actor and critic networks
        self.actor = Actor(self.state_dim, act_dim, act_range, 0.1 * lr, tau)
        self.critic = Critic(self.state_dim, act_dim, lr, tau)
        # self.buffer = MemoryBuffer(buffer_size)
        self.buffer = deque(maxlen=buffer_size)

    def policy_action(self, state):
        """ Use the actor to predict value
        """
        state = (state[0][np.newaxis], state[1][np.newaxis], state[2][np.newaxis])
        return self.actor.predict(state)[0]

    def bellman(self, rewards, q_values, dones):
        """ Use the Bellman Equation to compute the critic target
        """
        critic_target = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                critic_target[i] = rewards[i]
            else:
                critic_target[i] = rewards[i] + self.gamma * q_values[i]
        return critic_target

    def memorize(self, state, action, reward, done, new_state, td_error=None):
        """ Store experience in memory buffer
        """
        # self.buffer.memorize(state, action, reward, done, new_state)
        self.buffer.append((state, action, reward, done, new_state))

    def sample_batch(self, batch_size):
        # return self.buffer.sample_batch(batch_size)
        batch_experiences = random.sample(self.buffer, batch_size)
        states, next_states = [[experience[field_index] for experience in batch_experiences]
                               for field_index in range(4) if field_index in [0, 3]]
        actions, rewards, dones = [np.array([experience[field_index] for experience in batch_experiences])
                                   for field_index in range(5) if field_index in [1, 2, 4]]
        input1, input2, input3 = [np.array([state[field_index] for state in states])
                                  for field_index in range(3)]
        states = (input1, input2, input3)
        input1, input2, input3 = [np.array([next_state[field_index] for next_state in next_states])
                                  for field_index in range(3)]
        next_states = (input1, input2, input3)
        return states, actions, rewards, next_states, dones

    def update_models(self, states, actions, critic_target):
        """ Update actor and critic networks from sampled experience
        """
        # Train critic
        self.critic.train_on_batch(states, actions, critic_target)
        # Q-Value Gradients under Current Policy
        actions = self.actor.model(states)
        action_grads = self.critic.action_gradients(states, actions)
        # Train actor
        self.actor.train(states, np.array(action_grads).reshape((-1, self.act_dim)))
        # Transfer weights to target networks at rate Tau
        self.actor.transfer_weights()
        self.critic.transfer_weights()

    def train(self, batch_size):
        # Sample experience from buffer
        states, actions, rewards, next_states, dones = self.sample_batch(batch_size)
        # Predict target q-values using target networks
        q_values = self.critic.target_predict((next_states, self.actor.target_predict(next_states)))
        # Compute critic target
        critic_target = self.bellman(rewards, q_values, dones)
        # Train both networks on sampled batch, update target networks
        self.update_models(states, actions, critic_target)

    def play_and_train(self, env, batch_size=32, n_episode=1000, if_gather_stats=False):
        results = []

        # First, gather experience
        tqdm_e = tqdm(range(n_episode), desc='Score', leave=True, unit=" episodes")
        for e in tqdm_e:
            # Reset episode
            time, total_reward, done = 0, 0, False
            state = env.reset()
            noise = OrnsteinUhlenbeckProcess(size=self.act_dim)

            while not done:
                # Actor picks an action (following the deterministic policy)
                action = self.policy_action(state)
                # Clip continuous values to be valid w.r.t. environment
                action = np.clip(action + noise.generate(time), -self.act_range, self.act_range)
                # Retrieve new state, reward, and whether the state is terminal
                new_state, reward, done, _ = env.step(action)
                # Add outputs to memory buffer
                self.memorize(state, action, reward, new_state, done)
                # Update current state
                state = new_state
                total_reward += reward
                time += 1

            if len(self.buffer) >= batch_size:
                self.train(batch_size)

            # Gather stats every episode for plotting
            if if_gather_stats:
                mean, stdev = gather_stats(self, env)
                results.append([e, mean, stdev])

            # Display score
            tqdm_e.set_description("Score: " + str(total_reward))
            tqdm_e.refresh()

        return results

    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)


if __name__ == "__main__":
    IM_WIDTH = 800
    IM_HEIGHT = 600
    image_shape = (IM_HEIGHT, IM_WIDTH, 3)
    state_dim = [image_shape, image_shape, 1]
    action_dim = 2  # [throttle_brake, steer]

    algo = DDPG(act_dim=action_dim, state_dim=state_dim, act_range=1.0)

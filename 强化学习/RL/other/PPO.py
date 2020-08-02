import numpy as np
import os
import time
import gym
from collections import deque
import tensorflow as tf
import tensorflow.keras as K
from scipy import signal


class MemoryPPO:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        # a fucntion so that different dimensions state array shapes are all processed corecctly
        def combined_shape(length, shape=None):
            if shape is None:
                return (length,)
            return (length, shape) if np.isscalar(shape) else (length, *shape)

        # just empty arrays with appropriate sizes
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)  # states
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)  # actions
        # actual rwards from state using action
        self.rew_buf = np.zeros(size, dtype=np.float32)
        # predicted values of state
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)  # gae advantewages
        self.ret_buf = np.zeros(size, dtype=np.float32)  # discounted rewards
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def discount_cumsum(self, x, discount):
        """
        magic from rllab for computing discounted cumulative sums of vectors.
        example input: [x0, x1, x2] output: [x0 + discount * x1 + discount^2 * x2,  x1 + discount * x2, x2]
        """
        return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def store(self, obs, act, rew, val):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self.discount_cumsum(deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get_batch(self, batch_size):
        """simply retuns a randomized batch of batch_size from the data in memory
        """
        # make a randlim list with batch_size numbers.
        pos_lst = np.random.randint(self.ptr, size=batch_size)
        return self.obs_buf[pos_lst], self.act_buf[pos_lst], \
               self.adv_buf[pos_lst], self.ret_buf[pos_lst], self.val_buf[pos_lst]

    def clear(self):
        """Set back pointers to the beginning
        """
        self.ptr, self.path_start_idx = 0, 0


class Agent:
    def __init__(self, env, TRAINING_BATCH_SIZE, TRAJECTORY_BUFFER_SIZE):
        self.action_n = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape
        self.action_bound = env.action_space.high[0]
        # CONSTANTS
        self.TRAINING_BATCH_SIZE = TRAINING_BATCH_SIZE
        self.TRAJECTORY_BUFFER_SIZE = TRAJECTORY_BUFFER_SIZE
        self.TARGET_UPDATE_ALPHA = 0.95
        self.GAMMA = 0.99
        self.GAE_LAMBDA = 0.95
        self.CLIPPING_LOSS_RATIO = 0.1
        self.ENTROPY_LOSS_RATIO = 0.001
        self.TARGET_UPDATE_ALPHA = 0.9
        self.NOISE = 1.0  # Exploration noise, for continous action space
        # create actor and critic neural networks
        self.critic_network = self._build_critic_network()
        self.actor_network = self._build_actor_network()
        # for the loss function, additionally "old" predicitons are required from before the last update.
        # therefore create another networtk. Set weights to be identical for now.
        self.actor_old_network = self._build_actor_network()
        self.actor_old_network.set_weights(self.actor_network.get_weights())
        # for getting an action (predict), the model requires it's ususal input, but advantage and old_prediction is only used for loss(training). So create dummys for prediction only
        self.dummy_advantage = np.zeros((1, 1))
        self.dummy_old_prediciton = np.zeros((1, 2 * self.action_n))
        # our transition memory buffer
        self.memory = MemoryPPO(self.state_dim, self.action_n, self.TRAJECTORY_BUFFER_SIZE)

    def ppo_loss(self, advantage, old_prediction):
        def get_log_probability_density(mu_and_sigma, y_true):
            # the actor output contains mu and sigma concatenated. split them. shape is (batches,2xaction_n)
            mu = mu_and_sigma[:, 0:self.action_n]
            sigma = mu_and_sigma[:, self.action_n:]
            variance = K.backend.square(sigma)
            pdf = 1. / K.backend.sqrt(2. * np.pi * variance) * K.backend.exp(
                -K.backend.square(y_true - mu) / (2. * variance))
            log_pdf = K.backend.log(pdf + K.backend.epsilon())
            return log_pdf

        # refer to Keras custom loss function intro to understand why we define a funciton inside a function.
        # here y_true are the actions taken and y_pred are the predicted prob-distribution(mu,sigma) for each n in acion space
        def loss(y_true, y_pred):
            # First the probability density function.
            log_probability_density_new = get_log_probability_density(y_pred, y_true)
            log_probability_density_old = get_log_probability_density(old_prediction, y_true)
            # Calc ratio and the surrogates
            # ratio = prob / (old_prob + K.epsilon()) #ratio new to old
            ratio = K.backend.exp(log_probability_density_new - log_probability_density_old)
            surrogate1 = ratio * advantage
            clip_ratio = K.backend.clip(ratio, min_value=1 - self.CLIPPING_LOSS_RATIO,
                                        max_value=1 + self.CLIPPING_LOSS_RATIO)
            surrogate2 = clip_ratio * advantage
            # loss is the mean of the minimum of either of the surrogates
            loss_actor = - K.backend.mean(K.backend.minimum(surrogate1, surrogate2))
            # entropy bonus in accordance with move37 explanation https://youtu.be/kWHSH2HgbNQ
            sigma = y_pred[:, self.action_n:]
            variance = K.backend.square(sigma)
            loss_entropy = self.ENTROPY_LOSS_RATIO * K.backend.mean(
                -(K.backend.log(2 * np.pi * variance) + 1) / 2)  # see move37 chap 9.5
            # total bonus is all losses combined. Add MSE-value-loss here as well?
            return loss_actor + loss_entropy

        return loss

    def _build_actor_network(self):
        """builds and returns a compiled keras.model for the actor.
        There are 3 inputs. Only the state is for the pass though the neural net.
        The other two inputs are exclusivly used for the custom loss function (ppo_loss).
        """
        state = K.layers.Input(shape=(self.state_dim[0],), name='state_input')
        advantage = K.layers.Input(shape=(1,), name='advantage_input')
        old_prediction = K.layers.Input(shape=(2 * self.action_n,), name='old_prediction_input')
        dense = K.layers.Dense(32, activation='relu', name='dense1')(state)
        dense = K.layers.Dense(32, activation='relu', name='dense2')(dense)
        action = K.layers.Dense(self.action_n, activation='tanh', name="actor_output_mu")(dense)
        mu = K.layers.Lambda(lambda x: x * self.action_bound, name="lambda_mu")(action)
        sigma = K.layers.Dense(self.action_n, activation='softplus', name="actor_output_sigma")(dense)
        mu_and_sigma = K.layers.concatenate([mu, sigma])
        actor_network = K.Model(inputs=[state, advantage, old_prediction], outputs=mu_and_sigma)
        actor_network.compile(optimizer='adam', loss=self.ppo_loss(advantage, old_prediction),
                              experimental_run_tf_function=False)
        return actor_network

    def _build_critic_network(self):
        """builds and returns a compiled keras.model for the critic.
        The critic is a simple scalar prediction on the state value(output) given an state(input)
        Loss is simply mse
        """
        state = K.layers.Input(shape=(self.state_dim[0],), name='state_input')
        dense = K.layers.Dense(32, activation='relu', name='dense1')(state)
        dense = K.layers.Dense(32, activation='relu', name='dense2')(dense)
        V = K.layers.Dense(1, name="actor_output_layer")(dense)
        critic_network = K.Model(inputs=state, outputs=V)
        critic_network.compile(optimizer='Adam', loss='mean_squared_error')
        return critic_network

    def update_target_network(self):
        """Softupdate of the target network.
        In ppo, the updates of the
        """
        alpha = self.TARGET_UPDATE_ALPHA
        actor_weights = np.array(self.actor_network.get_weights())
        actor_tartget_weights = np.array(self.actor_old_network.get_weights())
        new_weights = alpha * actor_weights + (1 - alpha) * actor_tartget_weights
        self.actor_old_network.set_weights(new_weights)

    def choose_action(self, state, optimal=False):
        assert isinstance(state, np.ndarray)
        assert state.shape == self.state_dim
        state = state.reshape(1, -1)
        mu_and_sigma = self.actor_network.predict_on_batch([state, self.dummy_advantage, self.dummy_old_prediciton])
        mu = mu_and_sigma[0, 0:self.action_n]
        sigma = mu_and_sigma[0, self.action_n:]
        if optimal:
            action = mu
        else:
            action = np.random.normal(loc=mu, scale=sigma, size=self.action_n)
        return action

    def train_network(self):
        """Train the actor and critic networks using GAE Algorithm.
        1. Get GAE rewards, s,a,
        2. get "old" precition (of target network)
        3. fit actor and critic network
        4. soft update target "old" network
        """
        states, actions, gae_advantages, discounted_rewards, values = \
            self.memory.get_batch(self.TRAINING_BATCH_SIZE)
        gae_advantages = gae_advantages.reshape(-1, 1)  # batches of shape (1,) required
        gae_advantages = K.utils.normalize(gae_advantages)  # optionally normalize
        # calc old_prediction. Required for actor loss.
        batch_old_prediction = self.get_old_prediction(states)
        # commit training
        self.actor_network.fit(x=[states, gae_advantages, batch_old_prediction], y=actions, verbose=0)
        self.critic_network.fit(x=states, y=discounted_rewards, epochs=1, verbose=0)
        # soft update the target network(aka actor_old).
        self.update_target_network()

    def store_transition(self, s, a, r):
        """Store the experiences transtions into memory object.
        """
        value = self.get_v(s)
        self.memory.store(s, a, r, value)

    def get_v(self, state):
        """Returns the value of the state.
        Basically, just a forward pass though the critic networtk
        """
        v = self.critic_network.predict_on_batch(state[np.newaxis])
        return v

    def get_old_prediction(self, state):
        """Makes an prediction (an action) given a state on the actor_old_network.
        This is for the train_network --> ppo_loss
        """
        return self.actor_old_network.predict_on_batch(
            [state, self.dummy_advantage, self.dummy_old_prediciton])


# ENV_NAME = "MountainCarContinuous-v0"
ENV_NAME = "Pendulum-v0"
EPOCHS = 10000
MAX_EPISODE_STEPS = 2000
# train at the end of each epoch for simplicity. Not necessarily better.
TRAJECTORY_BUFFER_SIZE = MAX_EPISODE_STEPS
BATCH_SIZE = 256
RENDER_EVERY = 10

if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    agent = Agent(env, BATCH_SIZE, TRAJECTORY_BUFFER_SIZE)
    for epoch in range(EPOCHS):
        state = env.reset()
        r_sum = 0
        for t in range(MAX_EPISODE_STEPS):
            # sometimes render
            if epoch % RENDER_EVERY == 0:
                env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward)
            r_sum += reward
            if done or (t == MAX_EPISODE_STEPS - 1):
                # predict critic for s_ (value of s_)
                last_val = reward if done else agent.get_v(next_state)
                # do the discounted_rewards and GAE calucaltions
                agent.memory.finish_path(last_val)
                break
        agent.train_network()
        agent.memory.clear()
        if epoch % 10 == 0:
            print(f"Episode:{epoch}, step:{t}, r_sum:{r_sum}")
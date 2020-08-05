import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from rl.DDPG.actor import Actor
from rl.DDPG.critic import Critic
from utils.stats import gather_stats
from utils.networks import tfSummary, OrnsteinUhlenbeckProcess
from utils.memory_buffer import MemoryBuffer
import random
from collections import deque


class DDPG:
    def __init__(self, act_dim, state_dim, act_range, model_path, k=1, soft_update=True,
                 buffer_size=5000, gamma=0.99, lr=0.00005, tau=0.001):
        self.act_dim = act_dim
        self.act_range = act_range
        self.state_dim = state_dim
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.update_type = soft_update
        self.buffer_size = buffer_size

        self.model_path = model_path
        self.actor = Actor(self.state_dim, self.act_dim, self.act_range, 0.1 * self.lr, self.tau,
                           DDPG.create_conv2d_model)
        self.critic = Critic(self.state_dim, self.act_dim, self.lr, self.tau, DDPG.create_conv2d_model)
        # self.buffer = MemoryBuffer(buffer_size)
        self.buffer = deque(maxlen=self.buffer_size)

    @staticmethod
    def create_conv2d_model(shape):
        inp = keras.layers.Input(shape=shape)
        x = keras.layers.Lambda(lambda image: tf.cast(image, np.float32) / 255)(inp)
        x = keras.layers.SeparableConv2D(64, 7, strides=2, activation="selu", padding="same")(x)
        x = keras.layers.SeparableConv2D(128, 3, strides=2, activation="selu", padding="same")(x)
        x = keras.layers.SeparableConv2D(256, 3, strides=2, activation="selu", padding="same")(x)
        output = keras.layers.SeparableConv2D(128, 3, strides=2, activation="selu", padding="same")(x)
        model = keras.Model(inputs=[inp], outputs=[output])
        # print(model.summary())
        return model

    def policy_action(self, state):
        """ Use the actor to predict value
        """
        state = (state[0][np.newaxis], state[1][np.newaxis])
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
        input1, input2 = [np.array([state[field_index] for state in states])
                          for field_index in range(2)]
        states = (input1, input2)
        input1, input2 = [np.array([next_state[field_index] for next_state in next_states])
                          for field_index in range(2)]
        next_states = (input1, input2)
        return states, actions, rewards, next_states, dones

    def update_models(self, states, actions, q_target):
        """ Update actor and critic networks from sampled experience
        """
        # Train critic
        self.critic.model.train_on_batch([*states, actions], q_target)

        # Train actor
        with tf.GradientTape() as tape:
            with tf.GradientTape() as action_tape:
                actions = self.actor.model(states)
                action_tape.watch(actions)
                q_target = self.critic.model([*states, actions])
            action_grads = action_tape.gradient(q_target, actions)
        # 链式法则
        # 反馈Q值越大损失越小，得到的反馈Q值越小损失越大，因此只要对状态估计网络返回的Q值取负号
        params_grad = tape.gradient(actions, self.actor.model.trainable_variables, -action_grads)
        self.actor.optimizer.apply_gradients(zip(params_grad, self.actor.model.trainable_variables))

        self.actor.transfer_weights(self.update_type)
        self.critic.transfer_weights(self.update_type)

    def train(self, batch_size):
        # Sample experience from buffer
        states, actions, rewards, next_states, dones = self.sample_batch(batch_size)
        # Predict target q-values using target networks
        next_q_values = self.critic.target_predict([next_states, self.actor.target_predict(next_states)])
        # Compute critic target
        q_target = self.bellman(rewards, next_q_values, dones)
        # Train both networks on sampled batch, update target networks
        self.update_models(states, actions, q_target)

    def play_and_train(self, env, batch_size=32, n_episode=1000, load_model=False, if_gather_stats=False):
        if load_model:
            self.load_weights(self.model_path)

        results = []
        mean_reward = -np.inf

        tqdm_e = tqdm(range(n_episode), desc='Score', leave=True, unit=" episodes")
        for episode in tqdm_e:
            # Reset episode
            time, total_reward, done = 0, 0, False
            state = env.reset()
            noise = OrnsteinUhlenbeckProcess(size=self.act_dim)

            while not done:
                action = self.policy_action(state)
                action = np.clip(action + noise.generate(time), -self.act_range, self.act_range)
                new_state, reward, done, _ = env.step(action)
                self.memorize(state, action, reward, new_state, done)
                state = new_state
                total_reward += reward
                time += 1

            if len(self.buffer) >= batch_size:
                self.train(batch_size)

            self.save_weights(self.model_path)

            # Display score
            tqdm_e.set_description("Score: " + str(total_reward))
            tqdm_e.refresh()

        return results

    def imitation_learning(self, env, batch_size=32, n_episode=1000, load_model=False):
        if load_model:
            self.load_weights(self.model_path)

        tqdm_e = tqdm(range(n_episode), desc='Score', leave=True, unit=" episodes")
        for episode in tqdm_e:
            # Reset episode
            total_reward, done = 0, False
            state = env.reset()
            env.vehicle.set_autopilot(True)

            while not done:
                action = np.array([10, 10])
                new_state, reward, done, vehicle_control = env.step(action)
                action = np.array([vehicle_control.throttle - vehicle_control.brake, vehicle_control.steer])

                self.memorize(state, action, reward, new_state, done)
                state = new_state
                total_reward += reward

            env.vehicle.set_autopilot(False)
            env.clear_env()

            if len(self.buffer) >= batch_size:
                self.train(batch_size)

            self.save_weights(self.model_path)

            # Display score
            tqdm_e.set_description("Score: " + str(total_reward))
            tqdm_e.refresh()

    def save_weights(self, path):
        # path += '_LR_{}'.format(self.lr)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path):
        self.critic.load_weights(path)
        self.actor.load_weights(path)


if __name__ == "__main__":
    IM_WIDTH = 400
    IM_HEIGHT = 400
    image_shape = (IM_HEIGHT, IM_WIDTH, 3)
    state_dim = [image_shape, image_shape, 1]
    action_dim = 2  # [throttle_brake, steer]

    algo = DDPG(act_dim=action_dim, state_dim=state_dim, act_range=1.0)
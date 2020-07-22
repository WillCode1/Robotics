import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from DDPG.network.actor import Actor
from DDPG.network.critic import Critic
import random
from collections import deque
K = keras.backend


class DDPG:
    def __init__(self, act_dim, state_dim, act_range, model_path, hidden_layers, k=1,
                 buffer_size=5000, gamma=0.99, lr=0.00005, tau=0.001):
        self.act_dim = act_dim
        self.act_range = act_range
        self.state_dim = state_dim
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.buffer_size = buffer_size

        self.model_path = model_path
        self.actor = Actor(self.state_dim, self.act_dim, self.act_range, self.lr * 0.1, self.tau, hidden_layers)
        self.critic = Critic(self.state_dim, self.act_dim, self.lr, self.tau, hidden_layers)
        # self.buffer = MemoryBuffer(buffer_size)
        self.buffer = deque(maxlen=self.buffer_size)

    def policy_action(self, state):
        return self.actor.predict(state[np.newaxis])[0]

    def bellman(self, rewards, q_values, dones):
        critic_target = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                critic_target[i] = rewards[i]
            else:
                critic_target[i] = rewards[i] + self.gamma * q_values[i]
        return critic_target

    def memorize(self, state, action, reward, done, new_state, td_error=None):
        # self.buffer.memorize(state, action, reward, done, new_state)
        self.buffer.append((state, action, reward, done, new_state))

    def sample_batch(self, batch_size):
        # return self.buffer.sample_batch(batch_size)
        batch_experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch_experiences])
            for field_index in range(5)]
        return states, actions, rewards, next_states, dones

    def update_models(self, states, actions, q_target, if_debug):
        # Train critic
        loss = self.critic.model.train_on_batch([states, actions], q_target)
        if if_debug:
            print("mean_loss:{}".format(loss))

        # Train actor
        with tf.GradientTape() as tape:
            with tf.GradientTape() as action_tape:
                actions = self.actor.model(states)
                action_tape.watch(actions)
                q_target = self.critic.model([states, actions])
            action_grads = action_tape.gradient(q_target, actions)
        # 链式法则
        # 反馈Q值越大损失越小，得到的反馈Q值越小损失越大，因此只要对状态估计网络返回的Q值取负号
        params_grad = tape.gradient(actions, self.actor.model.trainable_variables, -action_grads)
        self.actor.optimizer.apply_gradients(zip(params_grad, self.actor.model.trainable_variables))

        self.actor.transfer_weights()
        self.critic.transfer_weights()

    def train(self, batch_size, if_debug=False):
        states, actions, rewards, next_states, dones = self.sample_batch(batch_size)
        next_q_values = self.critic.target_predict([next_states, self.actor.target_predict(next_states)])
        q_target = self.bellman(rewards, next_q_values, dones)
        self.update_models(states, actions, q_target, if_debug)

    def save_weights(self, path):
        # path += '_LR_{}'.format(self.lr)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path):
        self.critic.load_weights(path)
        self.actor.load_weights(path)


if __name__ == "__main__":
    state_dim = 3
    action_dim = 1

    algo = DDPG(act_dim=action_dim, state_dim=state_dim, act_range=1.0, model_path=f'', hidden_layers=[32, 32])

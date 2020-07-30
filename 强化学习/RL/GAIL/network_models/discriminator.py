import tensorflow as tf
from tensorflow import keras
import numpy as np

K = keras.backend


class Discriminator:
    def __init__(self, env, lr=0.001):
        self.model = self.create_model(env.observation_space.shape[0], env.action_space.n)
        # add noise for stabilise training
        # agent_a_one_hot += tf.random_normal(tf.shape(agent_a_one_hot), mean=0.2, stddev=0.1, dtype=tf.float32) / 1.2
        self.optimizer = keras.optimizers.Adam(lr, clipvalue=1.0)

    def create_model(self, state_dim, action_n):
        state = keras.layers.Input(shape=state_dim)
        action = keras.layers.Input(shape=1, dtype='int32')
        embedding = tf.one_hot(action, depth=action_n)
        # embedding = keras.layers.Embedding(input_dim=1, output_dim=2)(action)
        embedding = keras.layers.Flatten()(embedding)

        x = keras.layers.concatenate([state, embedding])
        x = keras.layers.Dense(20, activation=keras.layers.LeakyReLU(0.2))(x)
        x = keras.layers.Dense(20, activation=keras.layers.LeakyReLU(0.2))(x)
        x = keras.layers.Dense(20, activation=keras.layers.LeakyReLU(0.2))(x)
        prob = keras.layers.Dense(1, activation="sigmoid")(x)

        model = keras.Model(inputs=[state, action], outputs=[prob])
        return model

    def train(self, expert_s, expert_a, agent_s, agent_a):
        with tf.GradientTape() as tape:
            prob_expert = self.model([expert_s, expert_a])
            prob_agent = self.model([agent_s, agent_a])
            loss_expert = -tf.reduce_mean(tf.math.log(tf.clip_by_value(prob_expert, 1e-10, 1)))
            loss_agent = -tf.reduce_mean(tf.math.log(tf.clip_by_value(1 - prob_agent, 1e-10, 1)))
            loss = loss_expert + loss_agent
        grad = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        return loss.numpy()

    def get_rewards(self, states, actions):
        prob_agent = self.model.predict([states, actions])
        reward = np.log(np.clip(prob_agent, 1e-10, 1))
        return reward


if __name__ == "__main__":
    import gym
    env = gym.make('CartPole-v1')

    algo = Discriminator(env, 0.1)
    # print(algo.model.summary())
    expert_s, expert_a = (1, 1, 1, 1), (1,)
    agent_s, agent_a = (1, 1, 1, 1), (1,)

    expert_s = np.array(expert_s, dtype='float32')[np.newaxis]
    expert_a = np.array(expert_a, dtype='int32')[np.newaxis]
    agent_s = np.array(agent_s, dtype='float32')[np.newaxis]
    agent_a = np.array(agent_a, dtype='int32')[np.newaxis]

    reward = algo.get_rewards(agent_s, agent_a)
    print(reward)

    loss = algo.train(expert_s, expert_a, agent_s, agent_a)
    print(loss)

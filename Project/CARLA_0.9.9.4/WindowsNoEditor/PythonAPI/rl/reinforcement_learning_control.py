# reinforcement_learning_control.
"""
    1. 速度的多样性
    2. action的多样性
    3. 输入应该不只是image
"""

import glob
import os
import sys
import random
import numpy as np
import cv2
import time
import math
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

'''
carla.VehicleControl
Manages the basic movement of a vehicle using typical driving controls.

Instance Variables
    - throttle (float)
        A scalar value to control the vehicle throttle [0.0, 1.0]. Default is 0.0.
    - steer (float)
        A scalar value to control the vehicle steering [-1.0, 1.0]. Default is 0.0.
    - brake (float)
        A scalar value to control the vehicle brake [0.0, 1.0]. Default is 0.0.
    - hand_brake (bool)
        Determines whether hand brake will be used. Default is False.
    - reverse (bool)
        Determines whether the vehicle will move backwards. Default is False.
    - manual_gear_shift (bool)
        Determines whether the vehicle will be controlled by changing gears manually. Default is False.
    - gear (int)
        States which gear is the vehicle running on.
'''


class CarEnv:
    def __init__(self, img_height, img_width, show_camera=True, run_seconds_per_episode=100):
        self.show_camera = show_camera
        self.img_width = img_width
        self.img_height = img_height
        self.run_seconds_per_episode = run_seconds_per_episode

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.front_camera = None

    def process_img(self, image):
        # RGBA
        img = np.array(image.raw_data).reshape((self.img_height, self.img_width, 4))
        img = img[:, :, :3]
        if self.show_camera:
            cv2.namedWindow('camera_image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow("camera_image", img)
            cv2.waitKey(30)
        self.front_camera = img

    def collision_data(self, event):
        self.collision_hist.append(event)

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())

        # 解决出现在障碍物处问题
        loop = True
        while loop:
            loop = False
            random_seed = random.randint(1, 100)
            random.seed(random_seed)
            np.random.seed(random_seed)
            tf.random.set_seed(random_seed)
            try:
                self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
            except RuntimeError:
                print("spawn_actor at collision.")
                loop = True

        self.actor_list.append(self.vehicle)

        self.rgb_camera = self.blueprint_library.find("sensor.camera.rgb")
        self.rgb_camera.set_attribute("image_size_x", f"{self.img_width}")
        self.rgb_camera.set_attribute("image_size_y", f"{self.img_height}")
        self.rgb_camera.set_attribute("fov", "110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_camera, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        # 碰撞检测
        collision_sensor = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_sensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.1)

        self.episode_start = time.time()
        return self.front_camera

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + self.run_seconds_per_episode < time.time():
            done = True

        return self.front_camera, reward, done, None


class DQNAgent:
    def __init__(self, model, discount_rate=0.99, deque_maxlen=5000, update_frequency=50):
        self.model = model
        self.discount_rate = discount_rate
        self.update_frequency = update_frequency

        self.target_model = keras.models.clone_model(model)
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = Adam(lr=0.01)
        self.loss_fn = keras.losses.Huber()

        self.replay_memory = deque(maxlen=deque_maxlen)
        self.target_update_counter = 0

    def training_step(self, min_replay_memory_size=1000, batch_size=32, soft_update=False):
        if len(self.replay_memory) < min_replay_memory_size:
            return

        batch_experiences = random.sample(self.replay_memory, batch_size)
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch_experiences])
            for field_index in range(5)]
        next_Q_values = self.model.predict(next_states / 255)
        best_next_actions = np.argmax(next_Q_values, axis=1)
        next_mask = tf.one_hot(best_next_actions, self.model.output.shape[1]).numpy()
        next_best_Q_values = (self.target_model.predict(next_states / 255) * next_mask).sum(axis=1)
        target_Q_values = (rewards + (1 - dones) * self.discount_rate * next_best_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1)
        mask = tf.one_hot(actions, self.model.output.shape[1])
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states / 255)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.target_update_counter += 1
        if self.target_update_counter > self.update_frequency:
            # 更新目标模型
            if not soft_update:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0
            else:   # 软更新方式
                target_weights = self.target_model.get_weights()
                online_weights = model.get_weights()
                for index in range(len(target_weights)):
                    target_weights[index] = 0.99 * target_weights[index] + 0.01 * online_weights[index]
                self.target_model.set_weights(target_weights)

    def epsilon_greedy_policy(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.model.output.shape[1])  # 动作个数
        else:
            Q_values = self.model.predict(state[np.newaxis])
            return np.argmax(Q_values[0])


if __name__ == "__main__":
    # MEMORY_FRACTION = 0.8
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    # backend.set_session(tf.Session(config=tf.ConfigProto(gpu_option=gpu_options)))

    MODEL_NAME = "Xception"

    if not os.path.isdir("./models"):
        os.makedirs("./models")

    IM_WIDTH = 640
    IM_HEIGHT = 640

    base_model = Xception(weights=None, include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH, 3))
    x = base_model.output
    x = keras.layers.GlobalAvgPool2D()(x)
    output = keras.layers.Dense(3)(x)
    model = keras.Model(inputs=[base_model.input], outputs=[output])
    agent = DQNAgent(model, discount_rate=0.99, deque_maxlen=5000)
    env = CarEnv(IM_HEIGHT, IM_WIDTH, show_camera=False, run_seconds_per_episode=100)

    EPISODES = 601
    best_score = -np.inf
    total_rewards_list = []

    for episode in tqdm(range(EPISODES), ascii=True, unit="episodes"):
        state = env.reset()
        episode_reward = 0
        done = False

        while True:
            epsilon = max(1 - episode / 500, 0.01)
            action = agent.epsilon_greedy_policy(state / 255, epsilon)
            new_state, reward, done, _ = env.step(action)
            agent.replay_memory.append((state, action, reward, new_state, done))
            state = new_state
            episode_reward += reward

            if done:
                break

        for actor in env.actor_list:
            actor.destroy()

        total_rewards_list.append(episode_reward)
        if episode % 20 == 0:
            average_reward = sum(total_rewards_list) / len(total_rewards_list)

            if average_reward > best_score:
                best_score = average_reward
                min_reward = min(total_rewards_list)
                max_reward = max(total_rewards_list)

                agent.model.save(f'models/{MODEL_NAME}.h5')
                print(f"Save model by {MODEL_NAME}__{average_reward:_>7.2f}avg__{max_reward:_>7.2f}max"
                      f"__{min_reward:_>7.2f}min__{int(time.time())}")

            total_rewards_list = []

        print("Episode: {}, epsilon: {:.3f}".format(episode, epsilon))
        agent.training_step(batch_size=4)
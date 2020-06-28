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
        self.sem_camera_input = None
        self.depth_camera_input = None
        self.actor_list = []
        self.collision_hist = []
        self.lane_invasion = []
        self.acceleration = None
        self.angular_velocity = None

        random.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)

    def __del__(self):
        self.clear_env()

    def clear_env(self):
        for actor in env.actor_list:
            actor.destroy()
        self.sem_camera_input = None
        self.depth_camera_input = None
        self.actor_list = []
        self.collision_hist = []
        self.lane_invasion = []
        self.acceleration = None
        self.angular_velocity = None

    def init_vehicle(self):
        vehicle_transform = None
        # 解决出现在障碍物处问题
        loop = True
        while loop:
            loop = False
            try:
                vehicle_transform = random.choice(self.world.get_map().get_spawn_points())
                self.vehicle = self.world.spawn_actor(self.model_3, vehicle_transform)
            except RuntimeError:
                loop = True

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        self.actor_list.append(self.vehicle)
        return vehicle_transform

    def rgb_camera_callback(self, image):
        # RGBA
        img = np.array(image.raw_data).reshape((self.img_height, self.img_width, 4))
        img = img[:, :, :3]
        cv2.namedWindow('rgb_camera', cv2.WINDOW_AUTOSIZE)
        cv2.imshow("rgb_camera", img)
        cv2.waitKey(15)

    def sem_camera_callback(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        img = np.array(image.raw_data).reshape((self.img_height, self.img_width, 4))
        img = img[:, :, :3]
        if self.show_camera:
            cv2.namedWindow('semantic_segmentation', cv2.WINDOW_AUTOSIZE)
            cv2.imshow("semantic_segmentation", img)
            cv2.waitKey(15)
        self.sem_camera_input = img

    def depth_camera_callback(self, image):
        # image.convert(carla.ColorConverter.Depth)
        image.convert(carla.ColorConverter.LogarithmicDepth)
        img = np.array(image.raw_data).reshape((self.img_height, self.img_width, 4))
        img = img[:, :, :3]
        if self.show_camera:
            cv2.namedWindow('depth_camera', cv2.WINDOW_AUTOSIZE)
            cv2.imshow("depth_camera", img)
            cv2.waitKey(15)
        self.depth_camera_input = img

    def collision_callback(self, event):
        self.collision_hist.append(event)

    def lane_callback(self, lane):
        self.lane_invasion = lane.crossed_lane_markings

    def imu_callback(self, imu):
        self.acceleration = np.array([imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z])
        self.angular_velocity = np.array([imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z])

    def reset(self):
        self.clear_env()

        vehicle_transform = self.init_vehicle()
        camera_transform = carla.Transform(carla.Location(x=2, y=0, z=1), carla.Rotation(0, 180, 0))
        other_transform = carla.Transform(carla.Location(0, 0, 0), carla.Rotation(0, 0, 0))

        # 语义分割相机
        sem_camera = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        sem_camera.set_attribute("image_size_x", f"{self.img_width}")
        sem_camera.set_attribute("image_size_y", f"{self.img_height}")
        sem_camera.set_attribute("fov", "110")

        sem_camera = self.world.spawn_actor(sem_camera, camera_transform, attach_to=self.vehicle,
                                            attachment_type=carla.AttachmentType.SpringArm)
        sem_camera.listen(lambda data: self.sem_camera_callback(data))
        self.actor_list.append(sem_camera)

        # 深度相机
        depth_camera = self.blueprint_library.find("sensor.camera.depth")
        depth_camera.set_attribute("image_size_x", f"{self.img_width}")
        depth_camera.set_attribute("image_size_y", f"{self.img_height}")
        depth_camera.set_attribute("fov", "110")

        depth_camera = self.world.spawn_actor(depth_camera, camera_transform, attach_to=self.vehicle,
                                              attachment_type=carla.AttachmentType.SpringArm)
        depth_camera.listen(lambda data: self.depth_camera_callback(data))
        self.actor_list.append(depth_camera)

        # RGB相机
        if self.show_camera:
            rgb_camera = self.blueprint_library.find("sensor.camera.rgb")
            rgb_camera.set_attribute("image_size_x", f"{self.img_width}")
            rgb_camera.set_attribute("image_size_y", f"{self.img_height}")
            rgb_camera.set_attribute("fov", "110")

            rgb_camera = self.world.spawn_actor(rgb_camera, camera_transform, attach_to=self.vehicle,
                                                attachment_type=carla.AttachmentType.SpringArm)
            rgb_camera.listen(lambda data: self.rgb_camera_callback(data))
            self.actor_list.append(rgb_camera)

        """
        雷达传感器
        """
        # Add IMU sensor to ego vehicle.
        imu_bp = self.blueprint_library.find('sensor.other.imu')
        # imu_bp.set_attribute("sensor_tick", str(3.0))
        imu = self.world.spawn_actor(imu_bp, other_transform, attach_to=self.vehicle,
                                     attachment_type=carla.AttachmentType.Rigid)
        imu.listen(lambda imu: self.imu_callback(imu))
        self.actor_list.append(imu)

        # Add Lane invasion sensor to ego vehicle.
        lane_bp = self.blueprint_library.find('sensor.other.lane_invasion')
        lane_invasion = self.world.spawn_actor(lane_bp, other_transform, attach_to=self.vehicle,
                                               attachment_type=carla.AttachmentType.Rigid)
        lane_invasion.listen(lambda lane: self.lane_callback(lane))
        self.actor_list.append(lane_invasion)

        # 碰撞检测
        collision_sensor = self.blueprint_library.find("sensor.other.collision")
        collision_sensor = self.world.spawn_actor(collision_sensor, vehicle_transform, attach_to=self.vehicle)
        collision_sensor.listen(lambda event: self.collision_callback(event))
        self.actor_list.append(collision_sensor)

        while self.sem_camera_input is None or self.depth_camera_input is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        return (self.sem_camera_input, self.depth_camera_input, self.acceleration, self.angular_velocity)
        # return self.sem_camera_input

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1))
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

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

        if len(self.lane_invasion) != 0:
            reward -= 80
        self.lane_invasion = []

        if self.episode_start + self.run_seconds_per_episode < time.time():
            done = True

        return (self.sem_camera_input, self.depth_camera_input, self.acceleration, self.angular_velocity), \
               reward, done, None
        # return self.sem_camera_input, reward, done, None


class DQNAgent:
    def __init__(self, model, discount_rate=0.99, deque_maxlen=5000, update_frequency=50):
        self.discount_rate = discount_rate
        self.update_frequency = update_frequency

        self.model = model
        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = Adam(lr=0.01)
        self.loss_fn = keras.losses.Huber()

        self.replay_memory = deque(maxlen=deque_maxlen)
        self.target_update_counter = 0

    def training_step(self, min_replay_memory_size=1000, batch_size=32, soft_update=False):
        if len(self.replay_memory) < min_replay_memory_size:
            return

        batch_experiences = random.sample(self.replay_memory, batch_size)
        states, next_states = [[experience[field_index] for experience in batch_experiences]
                               for field_index in range(4) if field_index in [0, 3]]
        actions, rewards, dones = [np.array([experience[field_index] for experience in batch_experiences])
                                   for field_index in range(5) if field_index in [1, 2, 4]]
        input1, input2, input3, input4 = [np.array([state[field_index] for state in states])
                                          for field_index in range(4)]
        states = (input1, input2, input3, input4)
        input1, input2, input3, input4 = [np.array([next_state[field_index] for next_state in next_states])
                                          for field_index in range(4)]
        next_states = (input1, input2, input3, input4)

        next_Q_values = self.model.predict(next_states)
        best_next_actions = np.argmax(next_Q_values, axis=1)
        next_mask = tf.one_hot(best_next_actions, self.model.output.shape[1]).numpy()
        next_best_Q_values = (self.target_model.predict(next_states) * next_mask).sum(axis=1)
        target_Q_values = (rewards + (1 - dones) * self.discount_rate * next_best_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1)
        mask = tf.one_hot(actions, self.model.output.shape[1])
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
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
            else:  # 软更新方式
                target_weights = self.target_model.get_weights()
                online_weights = model.get_weights()
                for index in range(len(target_weights)):
                    target_weights[index] = 0.99 * target_weights[index] + 0.01 * online_weights[index]
                self.target_model.set_weights(target_weights)

    def epsilon_greedy_policy(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.model.output.shape[1])  # 动作个数
        else:
            Q_values = self.model.predict((state[0][np.newaxis], state[1][np.newaxis],
                                           state[2][np.newaxis], state[3][np.newaxis]))
            return np.argmax(Q_values[0])


def create_model(input_shape, action_num, include_velocity=True):
    # model = keras.models.Sequential([
    #     keras.layers.Lambda(lambda image: tf.cast(image, np.float32) / 255, input_shape=(IM_HEIGHT, IM_WIDTH, 3)),
    #     keras.layers.Conv2D(64, 7, activation="relu", padding="same"),
    #     keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'),
    #     keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
    #     keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'),
    #     keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
    #     keras.layers.GlobalAvgPool2D(),
    #     keras.layers.Dense(action_num)
    # ])

    input_1 = keras.layers.Input(shape=input_shape, name="sem_camera_input")
    input_2 = keras.layers.Input(shape=input_shape, name="depth_camera_input")
    input_3 = keras.layers.Input(shape=[3], name="acceleration_input")
    input_4 = keras.layers.Input(shape=[3], name="angular_velocity_input")

    x = keras.layers.concatenate([input_1, input_2])
    x = keras.layers.Lambda(lambda image: tf.cast(image, np.float32) / 255)(x)

    x = keras.layers.Conv2D(64, 7, activation="relu", padding="same")(x)
    x = keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(x)
    x = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(x)
    x = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = keras.layers.GlobalAvgPool2D()(x)

    if include_velocity:
        x = keras.layers.concatenate([x, input_3, input_4])
        x = keras.layers.Dense(20, activation="relu")(x)

    output = keras.layers.Dense(action_num)(x)

    model = keras.Model(inputs=[input_1, input_2, input_3, input_4], outputs=[output])
    return model


if __name__ == "__main__":
    # MEMORY_FRACTION = 0.8
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    # backend.set_session(tf.Session(config=tf.ConfigProto(gpu_option=gpu_options)))

    MODEL_NAME = "Xception"

    if not os.path.isdir("./models"):
        os.makedirs("./models")

    IM_WIDTH = 800
    IM_HEIGHT = 600
    action_num = 4
    batch_size = 4  # ??

    model = create_model(input_shape=(IM_HEIGHT, IM_WIDTH, 3), action_num=action_num)
    # print(model.summary())
    # model.load_weights(f'models/{MODEL_NAME}.h5')

    agent = DQNAgent(model, discount_rate=0.99, deque_maxlen=5000)
    env = CarEnv(IM_HEIGHT, IM_WIDTH, show_camera=False, run_seconds_per_episode=20)

    EPISODES = 601
    best_score = -np.inf
    total_rewards_list = []

    for episode in tqdm(range(EPISODES), ascii=True, unit="episodes"):
        state = env.reset()
        episode_reward = 0
        done = False

        while True:
            epsilon = max(1 - episode / 500, 0.05)
            action = agent.epsilon_greedy_policy(state, epsilon)
            new_state, reward, done, _ = env.step(action)
            agent.replay_memory.append((state, action, reward, new_state, done))
            state = new_state
            episode_reward += reward

            if done:
                break

        total_rewards_list.append(episode_reward)
        if episode % 10 == 0:
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

        agent.training_step(batch_size=batch_size)

# reinforcement_learning_control.
"""
    1. 速度的多样性
    2. 是否设置目标点
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
    def __init__(self, img_height, img_width, show_rgb_camera=False, show_sem_camera=False,
                 show_depth_camera=False, run_seconds_per_episode=None):
        self.show_rgb_camera = show_rgb_camera
        self.show_sem_camera = show_sem_camera
        self.show_depth_camera = show_depth_camera

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

        # vehicle.set_autopilot(True)
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
        if self.show_sem_camera:
            cv2.namedWindow('semantic_segmentation', cv2.WINDOW_AUTOSIZE)
            cv2.imshow("semantic_segmentation", img)
            cv2.waitKey(15)
        self.sem_camera_input = img

    def depth_camera_callback(self, image):
        # image.convert(carla.ColorConverter.Depth)
        image.convert(carla.ColorConverter.LogarithmicDepth)
        img = np.array(image.raw_data).reshape((self.img_height, self.img_width, 4))
        img = img[:, :, :3]
        if self.show_depth_camera:
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
        if self.show_rgb_camera:
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
        # Add IMU sensor to vehicle.
        imu_bp = self.blueprint_library.find('sensor.other.imu')
        # imu_bp.set_attribute("sensor_tick", str(3.0))
        imu = self.world.spawn_actor(imu_bp, other_transform, attach_to=self.vehicle,
                                     attachment_type=carla.AttachmentType.Rigid)
        imu.listen(lambda imu: self.imu_callback(imu))
        self.actor_list.append(imu)

        # Add Lane invasion sensor to vehicle.
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
        return (self.sem_camera_input, self.depth_camera_input,
                self.acceleration, self.angular_velocity, np.array([0.0]))

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1))
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        elif action == 4:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0))

        velocity = self.vehicle.get_velocity()
        velocity = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        kmh = 3.6 * velocity

        if len(self.collision_hist) != 0:
            done = True
            reward = -10000
        elif kmh <= 30:
            done = False
            reward = -1
        elif kmh <= 50:
            done = False
            reward = 10
        else:
            done = False
            reward = -10

        if len(self.lane_invasion) != 0:
            for lane in self.lane_invasion:
                if lane.type == carla.LaneMarkingType.Solid:
                    reward -= 10
                elif lane.type == carla.LaneMarkingType.SolidSolid:
                    reward -= 30

            self.lane_invasion = []

        if self.run_seconds_per_episode is not None:
            if self.episode_start + self.run_seconds_per_episode < time.time():
                done = True

        return (self.sem_camera_input, self.depth_camera_input,
                self.acceleration, self.angular_velocity, np.array([velocity])), reward, done, None


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

    def training_step(self, min_replay_memory_size=1000, batch_size=32, soft_update=True):
        if len(self.replay_memory) < min_replay_memory_size:
            return

        batch_experiences = random.sample(self.replay_memory, batch_size)
        states, next_states = [[experience[field_index] for experience in batch_experiences]
                               for field_index in range(4) if field_index in [0, 3]]
        actions, rewards, dones = [np.array([experience[field_index] for experience in batch_experiences])
                                   for field_index in range(5) if field_index in [1, 2, 4]]
        input1, input2, input3, input4, input5 = [np.array([state[field_index] for state in states])
                                                  for field_index in range(5)]
        states = (input1, input2, input3, input4, input5)
        input1, input2, input3, input4, input5 = [np.array([next_state[field_index] for next_state in next_states])
                                                  for field_index in range(5)]
        next_states = (input1, input2, input3, input4, input5)

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

        if soft_update:
            target_weights = self.target_model.get_weights()
            online_weights = self.model.get_weights()
            for index in range(len(target_weights)):
                target_weights[index] = 0.99 * target_weights[index] + 0.01 * online_weights[index]
            self.target_model.set_weights(target_weights)
        else:
            self.target_update_counter += 1
            if self.target_update_counter > self.update_frequency:
                self.target_update_counter = 0
                self.target_model.set_weights(self.model.get_weights())

    def epsilon_greedy_policy(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.model.output.shape[1])  # 动作个数
        else:
            Q_values = self.model.predict((state[0][np.newaxis], state[1][np.newaxis], state[2][np.newaxis],
                                           state[3][np.newaxis], state[4][np.newaxis]))
            return np.argmax(Q_values[0])


def create_model(input_shape, action_num, include_velocity=True):
    input_1 = keras.layers.Input(shape=input_shape, name="sem_camera_input")
    input_2 = keras.layers.Input(shape=input_shape, name="depth_camera_input")
    input_3 = keras.layers.Input(shape=[3], name="acceleration_input")
    input_4 = keras.layers.Input(shape=[3], name="angular_velocity_input")
    input_5 = keras.layers.Input(shape=[1], name="velocity_input")

    x = keras.layers.concatenate([input_1, input_2])
    x = keras.layers.Lambda(lambda image: tf.cast(image, np.float32) / 255)(x)

    x = keras.layers.Conv2D(64, 7, strides=3, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
    x = keras.layers.GlobalAvgPool2D()(x)

    if include_velocity:
        x = keras.layers.concatenate([x, input_3, input_4, input_5])
        x = keras.layers.Dense(30, activation="selu")(x)
        x = keras.layers.Dense(10, activation="selu")(x)

    output = keras.layers.Dense(action_num)(x)

    model = keras.Model(inputs=[input_1, input_2, input_3, input_4, input_5], outputs=[output])
    return model


if __name__ == "__main__":
    # MEMORY_FRACTION = 0.8
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    # backend.set_session(tf.Session(config=tf.ConfigProto(gpu_option=gpu_options)))

    if not os.path.isdir("./models"):
        os.makedirs("./models")

    IM_WIDTH = 800
    IM_HEIGHT = 600
    action_num = 5
    batch_size = 8
    run_seconds_per_episode = 50

    model = create_model(input_shape=(IM_HEIGHT, IM_WIDTH, 3), action_num=action_num)
    # print(model.summary())
    model.load_weights(f'models/-100186.00min_-40440.60avg_0.53epsilon_50s run_seconds.h5')

    agent = DQNAgent(model, discount_rate=0.99, deque_maxlen=5000)
    env = CarEnv(IM_HEIGHT, IM_WIDTH, show_sem_camera=True, run_seconds_per_episode=run_seconds_per_episode)

    EPISODES = 1000
    best_min = -np.inf
    best_average = -np.inf
    max_epsilon = 0.4
    total_rewards_list = []

    for episode in tqdm(range(EPISODES), ascii=True, unit="episodes"):
        state = env.reset()
        episode_reward = 0
        done = False

        while True:
            epsilon = max(max_epsilon - episode / EPISODES, 0.01)
            action = agent.epsilon_greedy_policy(state, epsilon)
            new_state, reward, done, _ = env.step(action)
            agent.replay_memory.append((state, action, reward, new_state, done))
            state = new_state
            episode_reward += reward

            if done:
                break

        total_rewards_list.append(episode_reward)
        if episode != 0 and episode % 10 == 0:
            average_reward = sum(total_rewards_list) / len(total_rewards_list)
            min_reward = min(total_rewards_list)
            max_reward = max(total_rewards_list)

            if min_reward > best_min or average_reward > best_average:
                if min_reward > best_min:
                    best_min = min_reward
                elif average_reward > best_average:
                    best_average = average_reward

                agent.model.save(f'models/{min_reward:.2f}min_{average_reward:.2f}avg_{epsilon:.2f}epsilon'
                                 f'_{run_seconds_per_episode}s run_seconds.h5')
                print(f"Save model.{average_reward:_>7.2f}avg__{max_reward:_>7.2f}max__{min_reward:_>7.2f}min")
            else:
                print(f"{average_reward:_>7.2f}avg__{max_reward:_>7.2f}max__{min_reward:_>7.2f}min")

            total_rewards_list = []

        agent.training_step(batch_size=batch_size)
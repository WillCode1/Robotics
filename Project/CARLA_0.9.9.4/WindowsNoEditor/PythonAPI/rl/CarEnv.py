import glob
import os
import sys
import random
import numpy as np
import cv2
import time
import math
import tensorflow as tf

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla_api.misc import get_speed
from carla_api.misc import distance_vehicle
from carla_api.misc import GlobalRouteAgent
from carla_api.misc import draw_waypoints
from carla_api.misc import spawn_car
from carla_api.misc import compute_cos_about_waypoint

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
                 show_depth_camera=False, run_seconds_per_episode=None, no_rendering_mode=True, debug=False):
        self.show_rgb_camera = show_rgb_camera
        self.show_sem_camera = show_sem_camera
        self.show_depth_camera = show_depth_camera

        self.img_width = img_width
        self.img_height = img_height
        self.run_seconds_per_episode = run_seconds_per_episode

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)
        try:
            world = self.client.load_world('Town04')
        except RuntimeError:
            pass
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        self.debug = debug
        settings = self.world.get_settings()
        settings.no_rendering_mode = no_rendering_mode
        self.world.apply_settings(settings)

        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.sem_camera_input = None
        self.depth_camera_input = None
        self.actor_list = []
        self.collision_hist = []
        self.lane_invasion = []
        self.acceleration = None
        self.angular_velocity = None

        self.global_route = None

        random.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)

    def clear_env(self):
        for actor in self.actor_list:
            actor.destroy()
        self.sem_camera_input = None
        self.depth_camera_input = None
        self.actor_list = []
        self.collision_hist = []
        self.lane_invasion = []

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
        waypoint = self.map.get_waypoint(self.vehicle.get_location())
        forward_vector = waypoint.transform.rotation.get_forward_vector()*5
        self.vehicle.set_velocity(forward_vector)
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
        if self.debug:
            # print('acceleration: x={0}, y={1}, z={2}'.
            #       format(imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z))
            print('gyroscope: x={0}, y={1}, z={2}'.format(imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z))

    def reset(self):
        self.clear_env()

        vehicle_transform = self.init_vehicle()
        camera_transform = carla.Transform(carla.Location(x=2, y=0, z=1), carla.Rotation(0, 180, 0))
        other_transform = carla.Transform(carla.Location(0, 0, 0), carla.Rotation(0, 0, 0))

        # 初始化全局路由代理
        if 0:
            self.global_route = GlobalRouteAgent(self.vehicle, target_speed=20)
            location = random.choice(self.world.get_map().get_spawn_points()).location
            start_waypoint = self.map.get_waypoint(self.vehicle.get_location())
            end_waypoint = self.map.get_waypoint(carla.Location(location.x, location.y, location.z))
            route = self.global_route.trace_route(start_waypoint, end_waypoint, sampling_distance=2)
            # for index, temp in enumerate(route):
            #     print(temp[0].transform.location, temp[0].transform.rotation)
            # draw_waypoints(self.world, route, z=100)
            spawn_car(self.world, route, self.model_3)

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
        # depth_camera = self.blueprint_library.find("sensor.camera.depth")
        # depth_camera.set_attribute("image_size_x", f"{self.img_width}")
        # depth_camera.set_attribute("image_size_y", f"{self.img_height}")
        # depth_camera.set_attribute("fov", "110")
        #
        # depth_camera = self.world.spawn_actor(depth_camera, camera_transform, attach_to=self.vehicle,
        #                                       attachment_type=carla.AttachmentType.SpringArm)
        # depth_camera.listen(lambda data: self.depth_camera_callback(data))
        # self.actor_list.append(depth_camera)

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
        雷达传感器、sensor.other.obstacle
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

        while self.sem_camera_input is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        return (self.sem_camera_input, np.array([0.0]))

    def step(self, action):
        action = action.astype(np.float64)
        throttle_brake, steer = action
        info = None

        if throttle_brake == 10.0 and steer == 10.0:
            # self.vehicle.apply_control(carla.VehicleControl(throttle=0.7, steer=0))
            self.vehicle.set_autopilot(True)
            info = self.vehicle.get_control()
        elif throttle_brake >= 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=throttle_brake, steer=steer))
        else:
            self.vehicle.apply_control(carla.VehicleControl(brake=-throttle_brake, steer=steer))

        velocity = get_speed(self.vehicle)
        interval_time = time.time() - self.episode_start

        current_waypoint = self.map.get_waypoint(self.vehicle.get_location())
        cos = compute_cos_about_waypoint(current_waypoint, self.vehicle)
        kmh = velocity[0] * cos * 3.6

        if kmh < -0.1:
            print("kmh:{}, normal:{}".format(velocity[0] * 3.6, kmh))

        if kmh >= 40:
            done = False
            reward = 1
        elif kmh < 0:
            done = False
            reward = kmh
        else:
            done = False
            reward = kmh / 40

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif current_waypoint.lane_type != carla.LaneType.Driving:
            done = True
            reward = -100

        if len(self.lane_invasion) != 0:
        #     for lane in self.lane_invasion:
        #         if lane.type == carla.LaneMarkingType.Solid:
        #             done = True
        #             reward -= 30
        #         elif lane.type == carla.LaneMarkingType.SolidSolid:
        #             done = True
        #             reward -= 50
            self.lane_invasion = []

        if self.run_seconds_per_episode is not None and interval_time > self.run_seconds_per_episode:
            done = True

        return (self.sem_camera_input, velocity), reward, done, info


if __name__ == "__main__":
    IM_WIDTH = 800
    IM_HEIGHT = 600
    image_shape = (IM_HEIGHT, IM_WIDTH, 3)
    state_dim = [image_shape, image_shape, 1]
    action_dim = 2  # [throttle_brake, steer]

    env = CarEnv(IM_HEIGHT, IM_WIDTH, show_sem_camera=True, no_rendering_mode=False)
    old_state = env.reset(), 0
    action = np.array([10, 1])
    while True:
        new_state, reward, done, _ = env.step(action)

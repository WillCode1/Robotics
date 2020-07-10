# reinforcement_learning_control.
"""
    1. sensor.other.obstacle 用于保持车距
    2. LaneType
    3. 是否设置目标点
    5. 传入多个点，拐角处给出3个以上点（3~5个点）
"""

import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from rl.CarEnv import CarEnv
from rl.DDPG.ddpg import DDPG


if __name__ == "__main__":
    if not os.path.isdir("./models"):
        os.makedirs("./models")

    IM_WIDTH = 800
    IM_HEIGHT = 600
    batch_size = 8
    EPISODES = 1000
    run_seconds_per_episode = 50
    image_shape = (IM_HEIGHT, IM_WIDTH, 3)
    state_dim = [image_shape, image_shape, 1]
    action_dim = 2  # [throttle_brake, steer]
    # model.load_weights(f'models/-10234.00min_-3670.20avg_0.37epsilon_50s run_seconds.h5')

    algo = DDPG(act_dim=action_dim, state_dim=state_dim, act_range=1.0)
    env = CarEnv(IM_HEIGHT, IM_WIDTH, show_sem_camera=True, run_seconds_per_episode=run_seconds_per_episode,
                 no_rendering_mode=False)

    stats = algo.play_and_train(env, path=f'models/', batch_size=batch_size, n_episode=EPISODES)

    # Export results to CSV
    df = pd.DataFrame(np.array(stats))
    df.to_csv("/logs.csv", header=['Episode', 'Mean', 'Stddev'], float_format='%10.5f')

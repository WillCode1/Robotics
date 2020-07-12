# reinforcement_learning_control.
"""
    1. sensor.other.obstacle 用于保持车距
    3. 是否设置目标点
    5. 传入多个点，拐角处给出3个以上点（3~5个点）
"""

import os
import numpy as np
import pandas as pd
from rl.CarEnv import CarEnv
from rl.DDPG.ddpg import DDPG


if __name__ == "__main__":
    if not os.path.isdir("./models"):
        os.makedirs("./models")

    IM_WIDTH = 400
    IM_HEIGHT = 400
    batch_size = 8
    EPISODES = 1000
    image_shape = (IM_HEIGHT, IM_WIDTH, 3)
    state_dim = [image_shape, image_shape, 1]
    action_dim = 2  # [throttle_brake, steer]

    lr = 0.005
    tau = 0.01

    soft_update = False
    load_model = False
    debug = False
    # model.load_weights(f'models/-10234.00min_-3670.20avg_0.37epsilon_50s run_seconds.h5')

    env = CarEnv(IM_HEIGHT, IM_WIDTH, show_sem_camera=False, run_seconds_per_episode=50,
                 no_rendering_mode=True, debug=debug)

    algo = DDPG(act_dim=action_dim, state_dim=state_dim, model_path=f'models/', soft_update=soft_update,
                buffer_size=5000, act_range=1.0, lr=lr, tau=tau)
    stats = algo.play_and_train(env, batch_size=batch_size, n_episode=EPISODES, load_model=load_model)

    # Export results to CSV
    df = pd.DataFrame(np.array(stats))
    df.to_csv("/logs.csv", header=['Episode', 'Mean', 'Stddev'], float_format='%10.5f')

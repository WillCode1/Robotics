import gym
import time

# 声明使用的环境
env = gym.make("Breakout-v0")
obs = env.reset()

for _ in range(1000):
    env.render()
    time.sleep(0.05)
    # 随机选择一个动作
    env.step(env.action_space.sample())

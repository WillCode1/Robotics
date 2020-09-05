import numpy as np


class RewardScale:
    def __init__(self):
        self.reward_list = np.array([], dtype=np.float32)

    def add_delay_reward(self, delay_reward):
        self.reward_list = np.append(self.reward_list, delay_reward)

    def std(self):
        return self.reward_list.std()


if __name__ == "__main__":
    rs = RewardScale()
    rs.add_delay_reward(12)
    print(rs.reward_list)
    print(rs.std())

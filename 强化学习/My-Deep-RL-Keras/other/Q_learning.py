# Q学习
import numpy as np

transition_probabilities = [  # shape=[s, a, s']
    [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
    [[0.0, 1.0, 0.0], None, [0.0, 0.0, 1.0]],
    [None, [0.8, 0.1, 0.1], None]]
rewards = [  # shape=[s, a, s']
    [[+10, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, -50]],
    [[0, 0, 0], [+40, 0, 0], [0, 0, 0]]]
possible_actions = [[0, 1, 2], [0, 2], [1]]

# 初始化
Q_values = np.full((3, 3), -np.inf)  # -np.inf for impossible actions
for state, actions in enumerate(possible_actions):
    Q_values[state, actions] = 0.0  # for all possible actions


def step(state, action):
    probas = transition_probabilities[state][action]
    next_state = np.random.choice([0, 1, 2], p=probas)
    reward = rewards[state][action][next_state]
    return next_state, reward


# 实现智能体的探索策略
def exploration_policy(state):
    return np.random.choice(possible_actions[state])


l_r0 = 0.05  # initial learning rate
decay = 0.005  # learning rate decay
gamma = 0.95  # discount factor
state = 0  # initial state

# 公式18-5 Q学习算法
for iteration in range(10000):
    action = exploration_policy(state)
    next_state, reward = step(state, action)
    next_value = np.max(Q_values[next_state])
    l_r = l_r0 / (1 + iteration * decay)
    Q_values[state, action] *= 1 - l_r
    Q_values[state, action] += l_r * (reward + gamma * next_value)
    state = next_state

print(Q_values)

# 对于每个状态，查询拥有最高Q-值的动作，最优动作
print(np.argmax(Q_values, axis=1))  # [0 0 1]
# 这样就得到了衰减因子等于0.95时，这个MDP的最佳策略是什么：
# 状态s0时选择动作a0；在状态s1时选择动作a0；在状态s2时选择动作a1。

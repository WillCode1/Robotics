import numpy as np
import sys
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# HelloGrid 环境
MAPS = {'4x4': ["SOOO", "OXOX", "OOOX", "XOOG"]}


class HelloGridEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None):
        # 环境地图Grid
        self.desc = desc
        # 获取MAPS的形状(4, 4)
        self.shape = desc.shape

        # 动作集个数
        nA = 4

        # 状态集个数
        nS = np.prod(self.desc.shape)

        # 设置最大的行号和最大的列号方便索引
        MAX_Y = desc.shape[0]
        MAX_X = desc.shape[1]

        # 初始化状态分布[1. 0. 0. ...], 并从格子S开始执行
        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        # 动作-状态转换概率字典
        P = {}

        # 使用numpy的nditer对状态grid进行遍历
        state_grid = np.arange(nS).reshape(desc.shape)
        it = np.nditer(state_grid, flags=['multi_index'])

        # 通常it.finish, it.iternext()连在一起使用
        while not it.finished:
            # 获取当前的状态state
            s = it.iterindex
            # 获取当前状态所在grid格子中的值
            y, x = it.multi_index

            # P[s][a] == [(probability, nextstate, reward, done)*4]
            P[s] = {a: [] for a in range(nA)}

            s_letter = desc[y][x]
            # 使用lambda表达式代替函数
            is_done = lambda letter: letter in b'GX'
            # 只有到达位置G奖励才为1
            reward = 1.0 if s_letter in b'G' else -1.0

            if is_done(s_letter):
                # 如果达到状态G, 直接更新动作-状态转换概率
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            else:
                # 如果还没有到达状态G
                # 新状态位置的索引
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1

                # 新状态位置的索引对应的字母
                sl_up = desc[ns_up // MAX_X][ns_up % MAX_X]
                sl_right = desc[ns_right // MAX_X][ns_right % MAX_X]
                sl_down = desc[ns_down // MAX_X][ns_down % MAX_X]
                sl_left = desc[ns_left // MAX_X][ns_left % MAX_X]

                P[s][UP] = [(1.0, ns_up, reward, is_done(sl_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(sl_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(sl_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(sl_left))]
            # 准备更新下一个状态
            it.iternext()

        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)

    def render(self, mode='human', close=False):
        # 判断程序是否已经结束
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        # 格式转换
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]

        state_grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(state_grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            # 对于当前状态用红色标注
            if self.s == s:
                desc[y][x] = utils.colorize(desc[y][x], "red", highlight=True)
            it.iternext()

        outfile.write("\n".join(' '.join(line) for line in desc) + "\n")

        if mode != 'human':
            return outfile
        print('')


desc = np.asarray(MAPS["4x4"], dtype='c')
env = HelloGridEnv(desc)
state = env.reset()
env.render()

for _ in range(5):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    print("action:{}({})".format(action, ["Up", "Right", "Down", "Left"][action]))
    print("done:{}, observation:{}, reward:{}".format(done, state, reward))
    env.render()

    if done:
        print("Episode finished after {} timesteps".format(_ + 1))
        break

import numpy as np

# 游戏环境设置
game_board = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, -1, 1]
])
start, end = (0, 0), (3, 3)

# Q-Learning参数
alpha = 0.1# 学习率
gamma = 0.9# 折扣因子
epsilon = 0.1# 探索概率

# 初始化Q表
Q = np.zeros((4, 4, 4))  # 4x4游戏板，4个动作

# 动作索引
actions = {
    0: (-1, 0),  # 上
    1: (1, 0),   # 下
    2: (0, -1),  # 左
    3: (0, 1)    # 右
}

# 训练过程
for episode in range(1000):
    state = start
    while state != end:
        if np.random.rand() < epsilon:
            action_index = np.random.choice([0, 1, 2, 3])
        else:
            action_index = np.argmax(Q[state[0], state[1]])
        
        action = actions[action_index]
        next_state = (state[0] + action[0], state[1] + action[1])
        
        # 检查边界
        if 0 <= next_state[0] < 4 and 0 <= next_state[1] < 4:
            reward = game_board[next_state[0], next_state[1]]
            # 更新Q表
            Q[state[0], state[1], action_index] += alpha * (reward + gamma * np.max(Q[next_state[0], next_state[1]]) - Q[state[0], state[1], action_index])
            state = next_state
        else:
            # 边界外，惩罚
            Q[state[0], state[1], action_index] += alpha * (-1 - Q[state[0], state[1], action_index])

# 输出最终的Q表
print("Final Q-Table:")
print(Q)

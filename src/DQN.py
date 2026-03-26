import gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

print(gym.__version__)

# 超参数
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 10
MEMORY_SIZE = 10000

# 定义Q网络
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, action_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# 初始化环境和网络
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

online_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(online_net.state_dict())
optimizer = optim.Adam(online_net.parameters(), lr=0.001)
buffer = ReplayBuffer(MEMORY_SIZE)

epsilon = EPS_START

# 训练循环
for episode in range(100):
    state,_ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # ε-greedy 选择动作
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = online_net(state_tensor)
                action = q_values.argmax().item()
        
        # 执行动作
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  # 合并终止标志
        total_reward += reward
        
        # 存储经验
        # 修改经验存储部分
        buffer.push(np.array(state), action, reward, np.array(next_state), done)
        state = next_state
        
        # 训练步骤
        if len(buffer) >= BATCH_SIZE:
            batch = buffer.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)
            
            # 计算当前Q值
            current_q = online_net(states).gather(1, actions.unsqueeze(1))
            
            # 计算目标Q值（使用目标网络）
            with torch.no_grad():
                next_q = target_net(next_states).max(1)[0]
                target_q = rewards + GAMMA * next_q * (1 - dones)
            
            # 计算损失并更新
            loss = nn.MSELoss()(current_q.squeeze(), target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # 更新目标网络
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(online_net.state_dict())
    
    # ε衰减
    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    
    print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()
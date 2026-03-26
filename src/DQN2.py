import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# 定义DQN网络
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 40)
        self.fc2 = nn.Linear(40, 80)
        self.fc3 = nn.Linear(80, action_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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

# 超参数
EPISODES = 500
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 1

# 初始化环境
env = gym.make('CartPole-v1', render_mode='human')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# 初始化网络和优化器
policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=0.005)
buffer = ReplayBuffer(10000)
epsilon = EPS_START

# 训练循环
for episode in range(EPISODES):
    state = env.reset()[0]
    total_reward = 0
    
    while True:
        # 渲染环境
        env.render()
        
        # 选择动作（epsilon贪婪策略）
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()
        
        # 执行动作
        next_state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        
        # 存储经验
        buffer.push(np.array(state), action, reward, np.array(next_state), done)
        state = next_state
        
        # 经验回放
        if len(buffer) >= BATCH_SIZE:
            batch = buffer.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # 转换为张量
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)
            
            # 计算Q值
            current_q = policy_net(states).gather(1, actions)
            next_q = target_net(next_states).max(1)[0].detach()
            target_q = rewards + (1 - dones) * GAMMA * next_q
            
            # 计算损失
            loss = nn.MSELoss()(current_q.squeeze(), target_q)
            
            # 优化网络
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if done or truncated:
            break
    
    # 更新epsilon
    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    
    # 更新目标网络
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    print(f"Episode: {episode+1}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

env.close()
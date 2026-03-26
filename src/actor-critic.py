import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        # 共享特征层
        self.shared = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
        # Actor分支
        self.actor = nn.Linear(256, action_size)
        
        # Critic分支（带目标网络）
        self.critic = nn.Linear(256, 1)
        self.target_critic = nn.Linear(256, 1)
        self._initialize_target()
        
    def _initialize_target(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        
    def forward(self, x, use_target=False):
        x = self.shared(x)
        policy = F.softmax(self.actor(x), dim=-1)
        critic = self.target_critic(x) if use_target else self.critic(x)
        return policy, critic

# 超参数配置
config = {
    "gamma": 0.99,          # 折扣因子
    "actor_lr": 1e-4,       # Actor学习率
    "critic_lr": 3e-4,      # Critic学习率
    "tau": 0.005,           # 目标网络混合系数
    "max_episodes": 500,    # 最大训练回合数
    "batch_size": 128,      # 批次大小
    "buffer_size": 10000,   # 经验回放容量
    "entropy_coef": 0.01,   # 熵正则化系数
    "gae_lambda": 0.95,     # GAE系数
    "clip_grad": 1.0,       # 梯度裁剪
    "update_freq": 3        # 网络更新频率
}

# 初始化环境
env = gym.make('CartPole-v1', render_mode='human')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# 初始化网络
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActorCritic(state_size, action_size).to(device)
optimizer = optim.Adam([
    {'params': model.actor.parameters(), 'lr': config['actor_lr']},
    {'params': model.critic.parameters(), 'lr': config['critic_lr']}
])
replay_buffer = deque(maxlen=config['buffer_size'])

# 训练循环
episode_rewards = []
for episode in range(config['max_episodes']):
    state = env.reset()[0]
    episode_data = []
    total_reward = 0
    
    # 收集轨迹数据
    while True:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            policy, value = model(state_tensor)
            dist = torch.distributions.Categorical(policy)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
        next_state, reward, done, truncated, _ = env.step(action.item())
        total_reward += reward
        
        episode_data.append({
            'state': state,
            'action': action.item(),
            'reward': reward,
            'log_prob': log_prob,
            'value': value,
            'entropy': entropy,
            'done': done or truncated
        })
        
        state = next_state
        if done or truncated:
            break
    
    # 计算GAE和returns
    states = torch.FloatTensor(np.array([t['state'] for t in episode_data])).to(device)
    with torch.no_grad():
        _, next_values = model(states, use_target=True)
    
    returns = []
    advantages = []
    gae = 0
    next_value = 0
    
    for t in reversed(range(len(episode_data))):
        delta = episode_data[t]['reward'] + config['gamma'] * next_value * (1 - episode_data[t]['done']) - episode_data[t]['value']
        gae = delta + config['gamma'] * config['gae_lambda'] * (1 - episode_data[t]['done']) * gae
        advantages.insert(0, gae)
        next_value = episode_data[t]['value']
    
    advantages = torch.tensor(advantages, device=device)
    returns = advantages + torch.cat([t['value'] for t in episode_data])
    
    # 标准化优势
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # 存储经验
    for t, data in enumerate(episode_data):
        replay_buffer.append((
            data['state'],
            data['action'],
            returns[t].item(),
            advantages[t].item(),
            data['log_prob'].item(),
            data['entropy'].item()
        ))
    
    # 网络更新
    if len(replay_buffer) >= config['batch_size'] and episode % config['update_freq'] == 0:
        batch = random.sample(replay_buffer, config['batch_size'])
        
        # 解包批次数据
        states_b = torch.FloatTensor(np.array([t[0] for t in batch])).to(device)
        actions_b = torch.LongTensor([t[1] for t in batch]).to(device)
        returns_b = torch.FloatTensor([t[2] for t in batch]).to(device)
        advantages_b = torch.FloatTensor([t[3] for t in batch]).to(device)
        old_log_probs_b = torch.FloatTensor([t[4] for t in batch]).to(device)
        entropies_b = torch.FloatTensor([t[5] for t in batch]).to(device)
        
        # 计算新策略
        new_policies, new_values = model(states_b)
        new_dist = torch.distributions.Categorical(new_policies)
        new_log_probs = new_dist.log_prob(actions_b)
        entropy = new_dist.entropy()
        
        # 计算损失
        ratio = (new_log_probs - old_log_probs_b).exp()
        actor_loss = -(ratio * advantages_b).mean() - config['entropy_coef'] * entropy.mean()
        
        critic_loss = F.mse_loss(new_values.squeeze(), returns_b)
        
        # 梯度更新
        optimizer.zero_grad()
        total_loss = actor_loss + critic_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'])
        optimizer.step()
        
        # 更新目标网络
        for target_param, param in zip(model.target_critic.parameters(), model.critic.parameters()):
            target_param.data.copy_(
                config['tau'] * param.data + (1 - config['tau']) * target_param.data
            )
    
    episode_rewards.append(total_reward)
    
    # 打印训练信息
    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        print(f"Episode: {episode+1:4d} | Reward: {total_reward:5.1f} | Avg10: {avg_reward:5.1f}")

env.close()
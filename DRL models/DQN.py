import gym
import math
import random
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 如果CUDA可用，启用CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建 gym 环境
env = gym.make('CartPole-v1')
env.seed(1)  # 为了复现结果

# 获取状态空间和动作空间的大小
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# 定义超参数
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# 定义策略网络
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 实例化策略网络和目标网络
policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # 不启用 BatchNormalization 和 Dropout

# 设置优化器
optimizer = optim.Adam(policy_net.parameters())

# 定义存储转移的命名元组
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# 定义经验回放的类
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """保存转移"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# 实例化经验回放
memory = ReplayMemory(10000)

# 选择动作的函数
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(action_size)]], device=device, dtype=torch.long)

# 优化模型的函数
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # 计算非最终状态的掩码并连接batch元素
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # 计算Q(s_t, a) - 模型计算 Q(s_t)，然后选择所采取动作的列。
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 计算所有下一个状态的 V(s_{t+1})
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # 计算期望的 Q 值
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # 计算 Huber 损失
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # 优化模型
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# 主训练循环
num_episodes = 50
for i_episode in range(num_episodes):
    # 初始化环境和状态
    state = env.reset()
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    for t in count():
        # 选择并执行一个动作
        action = select_action(state)
        next_state, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # 观察新的状态
        if not done:
            next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
        else:
            next_state = None

        # 在记忆中存储转移
        memory.push(state, action, next_state, reward)

        # 移动到下一个状态
        state = next_state

        # 进行一步优化 (在目标网络上)
        optimize_model()
        if done:
            break

    # 更新目标网络
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# 超参数
EPISODES = 1000
GAMMA = 0.99
LEARNING_RATE = 0.001
MEMORY_SIZE = 5000
BATCH_SIZE = 64
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.exploration_rate = EXPLORATION_MAX

        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            reward = torch.FloatTensor([reward])
            if done:
                target = reward
            else:
                target = reward + GAMMA * torch.max(self.model(next_state))
            q_values = self.model(state)
            target_f = q_values.clone().detach()
            target_f[0][action] = target
            loss = self.criterion(q_values, target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

if __name__ == "__main__":
    env = gym.make('RedistributionEnv-v0')
    # 计算 state_size，确保计算正确
    state_size = sum([space.n if isinstance(space, gym.spaces.Discrete) else np.prod(space.shape) for space in env.observation_space.spaces.values()])
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    rewards = []

    for episode in range(EPISODES):
        state_dict, _ = env.reset()  # 获取字典形式的状态
        state = np.concatenate([np.array([v]) if np.isscalar(v) else v for v in state_dict.values()])
        state = np.reshape(state, [1, state_size])
        done = False
        total_reward = 0

        while not done:
            env.render()
            action = agent.act(state)
            next_state_dict, reward, done, _ = env.step(action)
            next_state = np.concatenate([np.array([v]) if np.isscalar(v) else v for v in next_state_dict.values()])
            next_state = np.reshape(next_state, [1, state_size])
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.experience_replay()

        print(f"Episode: {episode}/{EPISODES}, Total Reward: {total_reward}")
        rewards.append(total_reward)

        if agent.exploration_rate > EXPLORATION_MIN:
            agent.exploration_rate *= EXPLORATION_DECAY

    env.close()

    # 绘制奖励曲线
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Training Rewards in RedistributionEnv')
    plt.savefig('training_rewards_redistributionenv.png')
    plt.show()

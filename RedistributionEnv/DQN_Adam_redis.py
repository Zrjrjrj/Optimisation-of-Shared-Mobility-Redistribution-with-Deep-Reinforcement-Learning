import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import time
from redistribution_env import RedistributionEnv

# Ensure GPU usage if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=200000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_min = 0.1  # minimum exploration rate
        self.epsilon_decay = 0.999  # slower decay rate for exploration
        self.learning_rate = 0.0005  # learning rate
        self.model = QNetwork(state_size, action_size, hidden_size).to(device)
        self.target_model = QNetwork(state_size, action_size, hidden_size).to(device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.99)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        """Update weights of the target network using the main network."""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # 探索
        if np.random.rand() <= self.epsilon:  # 生成的随机数小于等于epsilon的时候，agent选择随机动作
            return random.randrange(self.action_size)  # 动作空间中随机挑一个动作
        # 利用
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # 状态转换成floattensor类型，并添加一个批次的维度
        act_values = self.model(state)  # 转换类型后的状态输入到主函数，输出各个动作的Q值
        return np.argmax(act_values.detach().cpu().numpy()[0])  # 将预测的 Q 值从计算图中分离出来，并转换为 numpy 数组，选择 Q 值最大的动作

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return  # 样本不足不进行训练

        minibatch = random.sample(self.memory, batch_size)  # 随机采用一个小批量的经验。
        states, targets = [], []

        for state, action, reward, next_state, done in minibatch:  # 计算目标Q值
            target = reward  # 如果是done的，是终止状态，目标Q值就是当前的奖励
            if not done:  # 不是终止状态
                next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                target = reward + self.gamma * torch.max(self.target_model(next_state)[0]).item()  # 目标Q值：当前reward加上折扣后的未来Q值（未来Q值是用目标网络来计算的）
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            target_f = self.model(state).detach().cpu().numpy()  # 预测当前状态的Q值
            target_f[0][action] = target  # 预测Q值更新为目标Q值
            states.append(state)  # 更新后的状态和目标存储在列表中
            targets.append(torch.FloatTensor(target_f).to(device))

        states = torch.cat(states)  # 将列表中的状态和目标拼接成一个大批次
        targets = torch.cat(targets)
        # 训练模型
        self.model.train()  # 模型设置为训练模式，计算预测值和目标值之间的损失。
        output = self.model(states)
        loss = self.criterion(output, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # 更新epsilon
        self.scheduler.step()  # 调整学习率

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)


def preprocess_state(state):
    truck_position = state['truck_position']
    bike_states = state['bike_states']
    bikes_on_truck = state['bikes_on_truck']
    state_vector = np.concatenate(([truck_position], bike_states, [bikes_on_truck]))
    return state_vector


def smooth_rewards(scores, window=100):
    smoothed_scores = []
    for i in range(len(scores)):
        start = max(0, i - window)
        smoothed_scores.append(np.mean(scores[start:i + 1]))
    return smoothed_scores


if __name__ == "__main__":
    env = RedistributionEnv()
    state, _ = env.reset()
    state_size = len(preprocess_state(state))
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, hidden_size=128)
    batch_size = 128
    episodes = 50000
    scores = []
    target_update = 1000  # update target network every 1000 episodes

    model_dir = "DQN_Adam-50000_models"
    plot_dir = "DQN_Adam-50000_plots"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Start time
    start_time = time.time()

    for e in range(episodes):
        episode_start_time = time.time()

        state, _ = env.reset()
        state = preprocess_state(state)
        total_reward = 0
        for time_step in range(env.max_steps_per_episode):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            next_state = preprocess_state(next_state)
            agent.remember(state, action, reward, next_state, done or truncated)
            state = next_state
            if done or truncated:
                break
        scores.append(total_reward)
        episode_duration = (time.time() - episode_start_time) * 1000  # Convert to milliseconds
        print(
            f"episode: {e}/{episodes}, score: {total_reward}, e: {agent.epsilon:.2}, duration: {episode_duration:.2f} ms")
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if (e + 1) % target_update == 0:
            agent.update_target_model()
        if (e + 1) % 2500 == 0 or e == episodes - 1:
            agent.save(os.path.join(model_dir, f"dqn_redistribution_{e + 1}.pth"))

            # Save smoothed scores plot
            plt.figure(figsize=(10, 8))
            smoothed_scores = smooth_rewards(scores)
            plt.plot(smoothed_scores)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Smoothed DQN Training Rewards')
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir, f'smoothed_training_rewards_{e + 1}.png'))
            plt.close()

            # Save raw scores plot
            plt.figure(figsize=(10, 8))
            plt.plot(scores)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Raw DQN Training Rewards')
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir, f'raw_training_rewards_{e + 1}.png'))
            plt.close()

    total_training_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    print(f"Total training time: {total_training_time:.2f} ms")

    # Final smoothed scores plot
    plt.figure(figsize=(10, 8))
    smoothed_scores = smooth_rewards(scores)
    plt.plot(smoothed_scores)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Final Smoothed DQN Adam Training Rewards')
    plt.text(0.5, 0.01, f'Total training time: {total_training_time / 1000:.2f} seconds',
             horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'final_smoothed_training_rewards.png'))
    plt.show()

    # Final raw scores plot
    plt.figure(figsize=(10, 8))
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Final Raw DQN Adam Training Rewards')
    plt.text(0.5, 0.01, f'Total training time: {total_training_time / 1000:.2f} seconds',
             horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'final_raw_training_rewards.png'))
    plt.show()

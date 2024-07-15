import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from redistribution_env import RedistributionEnv

# Ensure GPU usage if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PolicyNet
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)


# ValueNet
class Critic(nn.Module):
    def __init__(self, state_size, hidden_size=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class A2C:
    def __init__(self, state_size, action_size, hidden_size,
                 actor_lr, critic_lr, gamma):
        # 实例化策略网络
        self.actor = Actor(state_size, action_size, hidden_size).to(device)
        # 实例化价值网络
        self.critic = Critic(state_size, hidden_size).to(device)
        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma  # 折扣因子

    # 动作选择
    def take_action(self, state):
        # 维度变换 [n_state] --> tensor[1, n_states]
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float).to(device)
        # 当前状态下，每个动作的概率分布 [1, n_actions]
        probs = self.actor(state)
        # 创建以probs为标准的概率分布
        action_dist = torch.distributions.Categorical(probs)
        # 依据其概率随机挑选一个动作
        action = action_dist.sample().item()
        return action

    # 训练
    def learn(self, transition_dict):
        # 提取数据集
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(device)
        actions = torch.tensor(transition_dict['actions']).to(device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(device).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(device).view(-1, 1)

        # 计算当前状态的state_value和优势函数
        td_value = self.critic(states)
        next_td_value = self.critic(next_states)
        td_target = rewards + self.gamma * next_td_value * (1 - dones)
        advantage = (td_target - td_value).detach()

        # 策略网络给出每个动作的概率，根据action得到当前时刻下该动作的概率
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = -torch.mean(log_probs * advantage)

        critic_loss = torch.mean(F.mse_loss(td_value, td_target.detach()))

        # 梯度清0
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        # 反向传播
        actor_loss.backward()
        critic_loss.backward()
        # 梯度更新
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def save(self, path):
        torch.save(self.actor.state_dict(), path + "_actor.pth")
        torch.save(self.critic.state_dict(), path + "_critic.pth")

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + "_actor.pth"))
        self.critic.load_state_dict(torch.load(path + "_critic.pth"))


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
    episodes = 20000
    gamma = 0.99
    actor_lr = 1e-4
    critic_lr = 2e-4
    hidden_size = 256

    env = RedistributionEnv()
    state, _ = env.reset()  # state字典
    state_size = len(preprocess_state(state))
    action_size = env.action_space.n

    agent = A2C(state_size, action_size, hidden_size, actor_lr, critic_lr, gamma)

    scores = []

    model_dir = "A2C_models"
    plot_dir = "A2C_plots"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Start time
    start_time = time.time()

    for e in range(episodes):
        episode_start_time = time.time()

        state, _ = env.reset()
        state = preprocess_state(state)  # 状态字典变一维数组（向量）才投入计算。
        total_reward = 0

        # 和DQN不同，不定义在循环外来积累整个训练过程的转移数据，而是只积累每个episode内的数据
        transition_dict = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': [],
        }

        for time_step in range(env.max_steps_per_episode):
            action = agent.take_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(preprocess_state(next_state))
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)

            total_reward += reward
            state = preprocess_state(next_state)
            if done or truncated:
                break

        scores.append(total_reward)
        agent.learn(transition_dict)

        episode_duration = (time.time() - episode_start_time) * 1000  # Convert to milliseconds
        print(
            f"episode: {e}/{episodes}, score: {total_reward}, duration: {episode_duration:.2f} ms")

        if (e + 1) % 2500 == 0 or e == episodes - 1:
            agent.save(os.path.join(model_dir, f'a2c_redistribution_{e + 1}'))

            # Save smoothed scores plot
            plt.figure(figsize=(10, 8))
            smoothed_scores = smooth_rewards(scores)
            plt.plot(smoothed_scores)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Smoothed A2C Training Rewards')
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir, f'smoothed_training_rewards_{e + 1}.png'))
            plt.close()

            # Save raw scores plot
            plt.figure(figsize=(10, 8))
            plt.plot(scores)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Raw A2C Training Rewards')
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
    plt.title('Final Smoothed A2C Training Rewards')
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
    plt.title('Final Raw A2C Training Rewards')
    plt.text(0.5, 0.01, f'Total training time: {total_training_time / 1000:.2f} seconds',
             horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'final_raw_training_rewards.png'))
    plt.show()

import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import yaml
import time

current_dir = os.path.dirname(os.path.abspath(__file__))

config_path = os.path.join(current_dir, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

batch_size = config['training']['batch_size']
lr = config['training']['lr']
loss_function = config['training']['loss']
num_episodes = config['training']['num_episodes']
warmup_episode = config['training']['warmup_episode']
save_freq = config['training']['save_freq']

optimizer_name = config['optimizer']['name']
lr_min = config['optimizer']['lr_min']
lr_decay = config['optimizer']['lr_decay']

gamma = config['rl']['gamma']
max_steps_per_episode = config['rl']['max_steps_per_episode']
target_model_update_episodes = config['rl']['target_model_update_episodes']
max_queue_length = config['rl']['max_queue_length']

max_epsilon = config['epsilon']['max_epsilon']
min_epsilon = config['epsilon']['min_epsilon']
decay_epsilon = config['epsilon']['decay_epsilon']

input_dim = config['model_for_cartpole']['input_dim']
output_dim = config['model_for_cartpole']['output_dim']
hidden_layers = config['model_for_cartpole']['hidden_layers']

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=max_queue_length)
        self.exploration_rate = max_epsilon

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_decay, gamma=0.99)
        self.criterion = nn.MSELoss() if loss_function == 'mse' else nn.SmoothL1Loss()

        self.update_target_model()
        self.training_step = 0

    def _build_model(self):
        layers = [nn.Linear(self.state_size, hidden_layers[0]), nn.ReLU()]
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layers[-1], self.action_size))
        return nn.Sequential(*layers)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def experience_replay(self):
        if len(self.memory) < batch_size:
            return 0
        batch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        total_loss = 0
        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            reward = torch.FloatTensor([reward])
            target = reward if done else reward + gamma * torch.max(self.target_model(next_state))
            q_values = self.model(state)
            target_f = q_values.clone().detach()
            target_f[0][action] = target
            states.append(state)
            targets_f.append(target_f)
        
        states = torch.cat(states)
        targets_f = torch.cat(targets_f)
        loss = self.criterion(self.model(states), targets_f)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        total_loss += loss.item()

        self.exploration_rate = max(min_epsilon, self.exploration_rate * decay_epsilon)
        return total_loss / batch_size

    def save_model(self, filepath, exploration_rate):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'exploration_rate': exploration_rate
        }, filepath)

    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.exploration_rate = checkpoint['exploration_rate']
        self.update_target_model()

if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode='human')
    state_size = input_dim
    action_size = output_dim
    agent = DQNAgent(state_size, action_size)
    rewards = []
    losses = []
    episode_times = []

    model_save_dir = os.path.join(current_dir, 'models_cartpole')
    os.makedirs(model_save_dir, exist_ok=True)

    latest_model_path = os.path.join(model_save_dir, 'dqn_cartpole_model_1000.pth')
    start_episode = 0
    if os.path.exists(latest_model_path):
        agent.load_model(latest_model_path)
        start_episode = int(latest_model_path.split('_')[-1].split('.')[0])
        print("Loaded model from", latest_model_path)

    start_time = time.time()
    consecutive_successes = 0
    success_threshold = 1000
    consecutive_goal = 10
    evaluation_window = 100

    for episode in range(start_episode, start_episode + num_episodes):
        episode_start_time = time.time()
        state, _ = env.reset(seed=42)
        state = np.reshape(state, [1, state_size])
        done = False
        total_reward = 0
        total_loss = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            loss = agent.experience_replay()
            total_loss += loss

        episode_end_time = time.time()
        episode_time = episode_end_time - episode_start_time
        episode_times.append(episode_time)

        rewards.append(total_reward)
        losses.append(total_loss)

        if episode % target_model_update_episodes == 0:
            agent.update_target_model()

        if (episode + 1) % save_freq == 0:
            model_save_path = os.path.join(model_save_dir, f'dqn_cartpole_model_{episode + 1}.pth')
            agent.save_model(model_save_path, agent.exploration_rate)
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(rewards)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('DQN Training Rewards in CartPole')
            plt.subplot(2, 1, 2)
            plt.plot(losses)
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.title(f'DQN Training Loss in CartPole (Total Training Time: {sum(episode_times):.2f} seconds)')
            plt.tight_layout()
            plt.savefig('dqn_cartpole_training.png')
            plt.close()

        print(f"Episode: {episode + 1}/{start_episode + num_episodes}, Total Reward: {total_reward}, Loss: {total_loss}, Time: {episode_time:.2f} seconds")

        if episode >= evaluation_window:
            recent_rewards = rewards[-evaluation_window:]
            if np.mean(recent_rewards) >= success_threshold:
                consecutive_successes += 1
                if consecutive_successes >= consecutive_goal:
                    print(f"Solved after {episode + 1} episodes!")
                    break
            else:
                consecutive_successes = 0

    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    env.close()

    agent.save_model(latest_model_path, agent.exploration_rate)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Training Rewards in CartPole')

    plt.subplot(2, 1, 2)
    plt.plot(losses)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title(f'DQN Training Loss in CartPole (Total Training Time: {total_training_time:.2f} seconds)')

    plt.tight_layout()
    plt.savefig('dqn_cartpole_training.png')
    plt.show()

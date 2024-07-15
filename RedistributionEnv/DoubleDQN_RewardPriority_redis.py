import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import time
from redistribution_env import RedistributionEnv

# Ensure GPU usage if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'final_reward'))
PrioritizedExperience = namedtuple('PrioritizedExperience', ('transition', 'priority'))


class RewardBasedPrioritizedReplayBuffer:
    def __init__(self, memory, alpha=0.6, beta=0.4):
        self.max_memory = memory
        self.buffer = []
        self.pos = 0
        self.alpha = alpha
        self.beta = beta

    def add(self, transition, final_reward):
        if final_reward > 400:
            priority = 1.0  # Highest priority
        # elif final_reward > 200:
        #     priority = 0.6
        elif final_reward > 0:
            priority = 0.6
        else:
            priority = 0.4  # Lowest priority

        experience = PrioritizedExperience(transition, priority)

        if len(self.buffer) < self.max_memory:
            self.buffer.append(experience)  # 未满，添加新的经验
        else:
            self.buffer[self.pos] = experience  # 满，从头替换旧经验
        self.pos = (self.pos + 1) % self.max_memory

    def sample(self, batch_size):
        priorities = np.array([experience.priority for experience in self.buffer])
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        batch = Transition(*zip(*[s.transition for s in samples]))
        return batch, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        pass  # No need to update priorities as they are based on rewards


class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = RewardBasedPrioritizedReplayBuffer(200000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0005
        self.model = QNetwork(state_size, action_size, hidden_size).to(device)
        self.target_model = QNetwork(state_size, action_size, hidden_size).to(device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.99)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done, final_reward):
        self.memory.add(Transition(state, action, reward, next_state, done, final_reward), final_reward)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        act_values = self.model(state)
        return np.argmax(act_values.detach().cpu().numpy()[0])

    def replay(self, batch_size):
        if len(self.memory.buffer) < batch_size:
            return

        transitions, indices, weights = self.memory.sample(batch_size)
        batch = Transition(*transitions)

        states = torch.FloatTensor(batch.state).to(device)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(batch.reward).to(device)
        next_states = torch.FloatTensor(batch.next_state).to(device)
        dones = torch.FloatTensor(batch.done).to(device)
        weights = weights.unsqueeze(1).to(device)

        # Double DQN update
        state_action_values = self.model(states).gather(1, actions)
        next_action_values = self.model(next_states).max(1)[1].unsqueeze(1)
        next_state_values = self.target_model(next_states).gather(1, next_action_values).detach()
        expected_state_action_values = (next_state_values * self.gamma * (1 - dones)) + rewards

        loss = (weights * (state_action_values - expected_state_action_values.unsqueeze(1)).pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.scheduler.step()

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
    episodes = 20000
    scores = []
    target_update = 1000

    model_dir = "DoubleDQN_RewardPriority_models"
    plot_dir = "DoubleDQN_RewardPriority_plots"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

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
            agent.remember(state, action, reward, next_state, done or truncated, total_reward)
            state = next_state
            if done or truncated:
                break
        scores.append(total_reward)
        episode_duration = (time.time() - episode_start_time) * 1000
        print(
            f"episode: {e}/{episodes}, score: {total_reward}, e: {agent.epsilon:.2}, duration: {episode_duration:.2f} ms")
        if len(agent.memory.buffer) > batch_size:
            agent.replay(batch_size)
        if (e + 1) % target_update == 0:
            agent.update_target_model()
        if (e + 1) % 2500 == 0 or e == episodes - 1:
            agent.save(os.path.join(model_dir, f"double dqn_RewardPriority_redistribution_{e + 1}.pth"))

            plt.figure(figsize=(10, 8))
            smoothed_scores = smooth_rewards(scores)
            plt.plot(smoothed_scores)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Smoothed double DQN RewardPriority Training Rewards')
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir, f'smoothed_training_rewards_{e + 1}.png'))
            plt.close()

            plt.figure(figsize=(10, 8))
            plt.plot(scores)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Raw double DQN RewardPriority Training Rewards')
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir, f'raw_training_rewards_{e + 1}.png'))
            plt.close()

    total_training_time = (time.time() - start_time) * 1000
    print(f"Total training time: {total_training_time:.2f} ms")

    plt.figure(figsize=(10, 8))
    smoothed_scores = smooth_rewards(scores)
    plt.plot(smoothed_scores)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Final Smoothed double DQN RewardPriority Training Rewards')
    plt.text(0.5, 0.01, f'Total training time: {total_training_time / 1000:.2f} seconds', horizontalalignment='center',
             verticalalignment='center', transform=plt.gca().transAxes)
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'final_smoothed_training_rewards.png'))
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Final Raw double DQN RewardPriority Training Rewards')
    plt.text(0.5, 0.01, f'Total training time: {total_training_time / 1000:.2f} seconds', horizontalalignment='center',
             verticalalignment='center', transform=plt.gca().transAxes)
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'final_raw_training_rewards.png'))
    plt.show()

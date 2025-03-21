"""
This script implements a reinforcement learning agent using Deep Q-Networks (DQN).
"""

import random
from collections import deque

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# Νευρωνικό Δίκτυο DQN
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# Replay Buffer
class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Επιλογή δράσης
def select_action(state, epsilon, policy_net, action_dim, device):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    else:
        with torch.no_grad():
            state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
            return policy_net(state).argmax().item()

# Εκπαίδευση
def train(policy_net, target_net, memory, optimizer, batch_size, gamma, device):
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    state, action, reward, next_state, done = zip(*transitions)

    state = torch.FloatTensor(np.array(state)).to(device)
    next_state = torch.FloatTensor(np.array(next_state)).to(device)
    action = torch.LongTensor(action).unsqueeze(1).to(device)
    reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done = torch.FloatTensor(done).unsqueeze(1).to(device)

    q_values = policy_net(state).gather(1, action)
    next_q_values = target_net(next_state).max(1)[0].detach().unsqueeze(1)
    target = reward + (1 - done) * gamma * next_q_values

    loss = nn.functional.mse_loss(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Αξιολόγηση πράκτορα χωρίς εξερεύνηση
def evaluate_agent(model_path, episodes=5, render=False):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(state_dim, action_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        while True:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state

            if done or truncated:
                break

        rewards.append(total_reward)
        print(f"Evaluation Episode {ep+1}: Total Reward = {total_reward}")

    env.close()
    return rewards

# Κύριο πρόγραμμα
if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    memory = ReplayMemory(10000)

    episodes = 300
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    min_epsilon = 0.01
    epsilon_decay = 0.995
    target_update_freq = 10
    all_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for t in range(500):
            action = select_action(state, epsilon, policy_net, action_dim, device)
            next_state, reward, done, truncated, _ = env.step(action)
            memory.push(state, action, reward, next_state, done or truncated)

            state = next_state
            total_reward += reward

            train(policy_net, target_net, memory, optimizer, batch_size, gamma, device)

            if done or truncated:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        all_rewards.append(total_reward)

        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()
    torch.save(policy_net.state_dict(), "model.pth")

    plt.plot(all_rewards)
    plt.xlabel("Επεισόδιο")
    plt.ylabel("Συνολική Ανταμοιβή")
    plt.title("Εκπαίδευση Πράκτορα DQN στο CartPole-v1")
    plt.grid()
    plt.savefig("reward_plot.png")

    # (Προαιρετικό) Εκτέλεση αξιολόγησης χωρίς εξερεύνηση
    evaluate_agent("model.pth", episodes=5, render=False)

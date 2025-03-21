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
    """ Class implementing a Deep Q-Network (DQN)."""
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
        """ Forward pass through the network."""
        return self.net(x)


# Replay Buffer
class ReplayMemory:
    """ Class implementing a Replay Buffer. """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """ Adds a new experience to memory."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """ Samples a batch of experiences from memory."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# Επιλογή δράσης
def select_action(current_state, epsilon_value, policy_net, action_dim, device):
    """ Selects an action based on epsilon-greedy policy."""
    if random.random() < epsilon_value:
        return random.randint(0, action_dim - 1)
    else:
        with torch.no_grad():
            state_tensor = torch.from_numpy(np.array(current_state)).float().unsqueeze(0).to(device)
            return policy_net(state_tensor).argmax().item()


# Εκπαίδευση
def train(policy_net, target_net, memory, optimizer, batch_size, gamma_value, device):
    """ Trains the network."""
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*transitions)

    states = torch.FloatTensor(np.array(states)).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

    q_values = policy_net(states).gather(1, actions)
    next_q_values = target_net(next_states).max(1)[0].detach().unsqueeze(1)
    target_values = rewards + (1 - dones) * gamma_value * next_q_values

    loss = nn.functional.mse_loss(q_values, target_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Αξιολόγηση πράκτορα χωρίς εξερεύνηση
def evaluate_agent(model_path, num_episodes=5, render_mode=False):
    """ Evaluates the agent on a specified number of episodes."""
    env = gym.make("CartPole-v1", render_mode="human" if render_mode else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(state_dim, action_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    episode_rewards = []

    for ep in range(num_episodes):
        eval_state, _ = env.reset()
        eval_total_reward = 0

        while True:
            state_tensor = torch.from_numpy(np.array(eval_state)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                eval_action = model(state_tensor).argmax().item()

            next_eval_state, eval_reward, eval_done, eval_truncated, _ = env.step(eval_action)
            eval_total_reward += eval_reward
            eval_state = next_eval_state

            if eval_done or eval_truncated:
                break

        episode_rewards.append(eval_total_reward)
        print(f"Evaluation Episode {ep + 1}: Total Reward = {eval_total_reward}")

    env.close()
    return episode_rewards


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

    num_episodes = 300
    batch_size = 64
    gamma_value = 0.99
    epsilon_value = 1.0
    min_epsilon_value = 0.01
    epsilon_decay_rate = 0.995
    target_update_freq = 10
    episode_rewards = []

    for episode_idx in range(num_episodes):
        episode_state, _ = env.reset()
        total_episode_reward = 0

        for timestep in range(500):
            selected_action = select_action(episode_state, epsilon_value, policy_net, action_dim, device)
            new_state, reward_obtained, is_done, is_truncated, _ = env.step(selected_action)
            memory.push(episode_state, selected_action, reward_obtained, new_state, is_done or is_truncated)

            episode_state = new_state
            total_episode_reward += reward_obtained

            train(policy_net, target_net, memory, optimizer, batch_size, gamma_value, device)

            if is_done or is_truncated:
                break

        epsilon_value = max(min_epsilon_value, epsilon_value * epsilon_decay_rate)
        episode_rewards.append(total_episode_reward)

        if episode_idx % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode_idx + 1}: Total Reward = {total_episode_reward}")

    env.close()
    torch.save(policy_net.state_dict(), "model.pth")

    plt.plot(episode_rewards)
    plt.xlabel("Επεισόδιο")
    plt.ylabel("Συνολική Ανταμοιβή")
    plt.title("Εκπαίδευση Πράκτορα DQN στο CartPole-v1")
    plt.grid()
    plt.savefig("reward_plot.png")

    # (Προαιρετικό) Εκτέλεση αξιολόγησης χωρίς εξερεύνηση
    evaluate_agent("model.pth", num_episodes=5, render_mode=False)

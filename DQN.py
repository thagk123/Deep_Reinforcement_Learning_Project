""" Module to train and evaluate a DQN and a Double DQN agent on the LunarLander-v3 environment. """

import os
import random
import shutil
import time
from collections import deque
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ------------------------------
# Hyperparameters
# ------------------------------
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 128
TARGET_UPDATE_FREQ = 50
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.997
NUM_EPISODES = 1000


class DQN(nn.Module):
    """ A simple feedforward neural network for DQN. """
    def __init__(self):
        """ Initialize the DQN model with 3 fully connected layers. """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        """ Forward pass through the network. """
        return self.net(x)


class ReplayBuffer:
    """ A simple replay buffer to store transitions. """
    def __init__(self, capacity):
        """ Initialize the replay buffer with a maximum capacity. """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """ Store a transition in the replay buffer. """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """ Sample a batch of transitions from the replay buffer. """
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.int64).unsqueeze(1),
            torch.tensor(reward, dtype=torch.float32).unsqueeze(1),
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        """ Return the current size of the replay buffer. """
        return len(self.buffer)


def train_dqn(use_double=False):
    """ Train the DQN or Double DQN agent on the LunarLander-v3 environment.

    Args:
        use_double (bool): If True, use Double DQN. Otherwise, use standard DQN.
    """
    env = gym.make("LunarLander-v3")
    policy_net = DQN()
    target_net = DQN()
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(100000)

    epsilon = EPS_START
    step_count = 0
    start = time.time()

    best_reward = float('-inf')

    # Προσθήκη για συλλογή rewards
    all_rewards = []
    average_rewards = []

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        total_reward = 0

        while True:
            step_count += 1

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action = policy_net(state_tensor).argmax(dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            if len(replay_buffer) >= BATCH_SIZE:
                s, a, r, s_next, d = replay_buffer.sample(BATCH_SIZE)

                q_all = policy_net(s)
                q_values = q_all.gather(1, a)

                if use_double:
                    next_actions = policy_net(s_next).argmax(1).unsqueeze(1)
                    next_q_values = target_net(s_next).gather(1, next_actions).detach()
                else:
                    next_q_values = target_net(s_next).max(1)[0].detach().unsqueeze(1)

                expected_q_values = r + GAMMA * next_q_values * (1 - d)

                loss = nn.MSELoss()(q_values, expected_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if step_count % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        # Calculating reward
        all_rewards.append(total_reward)
        if (episode + 1) % 100 == 0:
            avg_reward = sum(all_rewards[-100:]) / 100
            average_rewards.append({'Episode': episode + 1, 'Average Reward': avg_reward})

        if total_reward > best_reward:
            best_reward = total_reward
            save_path = "best_model_double.pth" if use_double else "best_model.pth"
            torch.save(policy_net.state_dict(), save_path)
            print(f"New best model saved, with reward: {best_reward:.2f}")

        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        agent_type = "Double DQN" if use_double else "DQN"
        print(
            f"[{agent_type}] Episode {episode + 1}, "
            f"Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}"
        )

    env.close()
    end = time.time()
    print(f"Training time: {(end - start)/60:.2f} minutes")

    # Making the DataFrame
    df = pd.DataFrame(average_rewards)
    print(df)

    # Making the plot
    plt.plot(df['Episode'], df['Average Reward'], marker='o')
    plt.title('Μέσο Reward ανά 100 επεισόδια')
    plt.xlabel('Επεισόδιο')
    plt.ylabel('Μέσο Reward')
    plt.grid(True)
    plt.show()


def test_agent(model_path, save_dir):
    """ Test the trained agent on the LunarLander-v3 environment."""

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=save_dir,
        name_prefix="lunar_test",
        episode_trigger=lambda x: True
    )

    model = DQN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    rewards = []

    print(f"Testing agent from '{model_path}'")
    for episode in range(10):
        state, _ = env.reset()
        total_reward = 0

        while True:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = model(state_tensor)
                action = q_values.argmax().item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
            if terminated or truncated:
                break

        rewards.append(total_reward)
        print(f"  Episode {episode + 1}, Reward: {total_reward:.2f}")

    env.close()

    best_ep = int(np.argmax(rewards))
    print(f"Best episode: {best_ep + 1}, Reward: {rewards[best_ep]:.2f}")
    print(f"Videos saved to '{save_dir}'")

def main():
    """ Main function to train and test the DQN and Double DQN agents. """
    #train_dqn(use_double=False)
    #test_agent("best_model.pth", save_dir="test_results")

    train_dqn(use_double=True)
    test_agent("best_model_double.pth", "test_results_double")

if __name__ == "__main__":
    main()

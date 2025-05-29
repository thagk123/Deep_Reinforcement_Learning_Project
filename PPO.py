"""
Module to train and evaluate a PPO agent on the LunarLander-v2 environment.
"""

import os
import time
import shutil
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import RecordVideo

# ------------------------------
# Hyperparameters
# ------------------------------
GAMMA = 0.99
LR = 3e-4
BATCH_SIZE = 4096
CLIP_EPSILON = 0.2
UPDATE_EPOCHS = 10
NUM_EPISODES = 5000


class PPOActorCritic(nn.Module):
    """A simple actor-critic network for PPO."""

    def __init__(self):
        """Initialize the actor-critic network."""
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, 4)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        """Forward pass through the network."""
        x = self.shared(x)
        return self.actor(x), self.critic(x)


class RolloutBuffer:
    """Buffer to collect experiences for PPO updates."""

    def __init__(self):
        """Initialize the buffer."""
        self.clear()

    def clear(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []


def compute_returns(rewards, dones, gamma):
    """Compute cumulative returns for each timestep."""
    returns = []
    G = 0
    for r, d in zip(reversed(rewards), reversed(dones)):
        if d:
            G = 0
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)

def final_evaluation(env, model, best_reward, update_index, average_rewards):
    """Evaluate final trained model with a rollout of 4096 steps."""
    buffer = RolloutBuffer()
    total_steps = 0
    rewards_per_episode = []

    while total_steps < BATCH_SIZE:
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, value = model(state_tensor)
                probs = torch.softmax(logits, dim=1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
                logprob = dist.log_prob(torch.tensor(action)).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.states.append(state)
            buffer.actions.append(action)
            buffer.logprobs.append(logprob)
            buffer.rewards.append(reward)
            buffer.dones.append(done)
            buffer.values.append(value.item())

            state = next_state
            total_reward += reward
            total_steps += 1

            if done:
                rewards_per_episode.append(total_reward)
                break

    avg_reward = np.mean(rewards_per_episode)
    average_rewards.append({'Update': update_index, 'Average Reward': avg_reward})

    if avg_reward > best_reward:
        torch.save(model.state_dict(), "best_model_ppo.pth")
        print(f"[Final Eval] Final model was best with Avg Reward: {avg_reward:.2f}")

    return average_rewards

def train_ppo():
    """Train PPO agent on LunarLander-v2."""
    env = gym.make("LunarLander-v2")
    model = PPOActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
    buffer = RolloutBuffer()

    best_reward = float('-inf')
    episode = 0
    update_index = 0
    start_time = time.time()
    average_rewards = []

    while episode < NUM_EPISODES:
        buffer.clear()
        total_steps = 0
        rewards_per_episode = []

        while total_steps < BATCH_SIZE:
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                logits, value = model(state_tensor)
                probs = torch.softmax(logits, dim=1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
                logprob = dist.log_prob(torch.tensor(action)).item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                buffer.states.append(state)
                buffer.actions.append(action)
                buffer.logprobs.append(logprob)
                buffer.rewards.append(reward)
                buffer.dones.append(done)
                buffer.values.append(value.item())

                state = next_state
                total_reward += reward
                total_steps += 1

                if done:
                    rewards_per_episode.append(total_reward)
                    episode += 1
                    break

        avg_reward = np.mean(rewards_per_episode)
        average_rewards.append({'Update': update_index, 'Average Reward': avg_reward})
        update_index += 1

        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(model.state_dict(), "best_model_ppo.pth")
            print(f"New best model saved, with avg reward: {best_reward:.2f}")

        print(f"[PPO] Episode {episode}, Avg Reward: {avg_reward:.2f}")

        returns = compute_returns(buffer.rewards, buffer.dones, GAMMA)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        states = torch.tensor(np.array(buffer.states), dtype=torch.float32)
        actions = torch.tensor(buffer.actions)
        old_logprobs = torch.tensor(buffer.logprobs, dtype=torch.float32)
        values = torch.tensor(buffer.values, dtype=torch.float32)

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(UPDATE_EPOCHS):
            logits, new_values = model(states)
            probs = torch.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)

            new_logprobs = dist.log_prob(actions)
            ratio = torch.exp(new_logprobs - old_logprobs)

            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * advantages

            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            value_loss = nn.MSELoss()(new_values.squeeze(), returns)

            entropy = dist.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

    # --- Final Evaluation for the last policy ---
    average_rewards = final_evaluation(env, model, best_reward, update_index, average_rewards)

    env.close()
    end_time = time.time()
    print(f"Training time: {(end_time - start_time) / 60:.2f} minutes")

    # Grouped updates diagram
    df = pd.DataFrame(average_rewards)
    group_size = 20
    num_groups = len(df) // group_size

    smoothed = []
    for i in range(num_groups):
        start = i * group_size
        end = start + group_size
        group = df.iloc[start:end]
        avg = group['Average Reward'].mean()
        update = group['Update'].iloc[-1]
        smoothed.append({'Update': update, 'Average Reward': avg})

    df_smooth = pd.DataFrame(smoothed)
    plt.plot(df_smooth['Update'], df_smooth['Average Reward'], marker='o')
    plt.title('Μέσο Reward ανά PPO Update (Grouped)')
    plt.xlabel('PPO Update')
    plt.ylabel('Μέσο Reward')
    plt.grid(True)
    plt.show()

def test_agent(model_path, save_dir):
    """Test a trained PPO agent and record videos."""
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=save_dir,
        name_prefix="lunar_test",
        episode_trigger=lambda x: True
    )

    model = PPOActorCritic()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    rewards = []

    print(f"Testing agent from '{model_path}'")
    for episode in range(10):
        state, _ = env.reset()
        total_reward = 0

        while True:
            with torch.no_grad():
                logits, _ = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                probs = torch.softmax(logits, dim=1)
                action = torch.argmax(probs, dim=1).item()

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
    """Main function to train and test the PPO agent."""
    train_ppo()
    test_agent("best_model_ppo.pth", "test_results_ppo")


if __name__ == "__main__":
    main()

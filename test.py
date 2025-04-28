import os
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from collections import deque
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
    """Actor-Critic network for PPO."""
    def __init__(self):
        """Initialize the network."""
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
    """Buffer to collect trajectories."""
    def __init__(self):
        """Initialize the buffer."""
        self.clear()

    def clear(self):
        """ Clear the buffer."""
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []


def compute_returns(rewards, dones, gamma):
    """Compute returns for each timestep."""
    returns = []
    G = 0
    for r, d in zip(reversed(rewards), reversed(dones)):
        if d:
            G = 0
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)


def collect_trajectories(env, model, buffer, episode_counter):
    """Collect BATCH_SIZE steps of experience."""
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
                episode_counter[0] += 1
                break

    return rewards_per_episode


def ppo_update(model, optimizer, states, actions, old_logprobs, returns, advantages):
    """Perform PPO updates."""
    for _ in range(UPDATE_EPOCHS):
        logits, new_values = model(states)
        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)

        new_logprobs = dist.log_prob(actions)
        ratio = torch.exp(new_logprobs - old_logprobs)

        clipped_ratio = []
        for r in ratio:
            if r.item() < 1.0 - CLIP_EPSILON:
                clipped_ratio.append(1.0 - CLIP_EPSILON)
            elif r.item() > 1.0 + CLIP_EPSILON:
                clipped_ratio.append(1.0 + CLIP_EPSILON)
            else:
                clipped_ratio.append(r.item())
        clipped_ratio = torch.tensor(clipped_ratio)

        surrogate1 = ratio * advantages
        surrogate2 = clipped_ratio * advantages

        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        value_loss = nn.MSELoss()(new_values.squeeze(), returns)
        entropy = dist.entropy().mean()

        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_ppo():
    """ Train the PPO agent."""
    env = gym.make("LunarLander-v2")
    model = PPOActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
    buffer = RolloutBuffer()

    best_reward = float('-inf')
    start_time = time.time()
    episode_counter = [0]

    while episode_counter[0] < NUM_EPISODES:
        rewards_per_episode = collect_trajectories(env, model, buffer, episode_counter)

        returns = compute_returns(buffer.rewards, buffer.dones, GAMMA)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        states = torch.tensor(buffer.states, dtype=torch.float32)
        actions = torch.tensor(buffer.actions)
        old_logprobs = torch.tensor(buffer.logprobs, dtype=torch.float32)
        values = torch.tensor(buffer.values, dtype=torch.float32)

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ppo_update(model, optimizer, states, actions, old_logprobs, returns, advantages)

        scheduler.step()

        avg_reward = np.mean(rewards_per_episode)
        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(model.state_dict(), "best_model_ppo.pth")
            print(f"New best model saved, with avg reward: {best_reward:.2f}")

        print(f"[PPO] Episode {episode_counter[0]}, Avg Reward: {avg_reward:.2f}")

    env.close()
    end_time = time.time()
    print(f"Training time: {(end_time - start_time)/60:.2f} minutes")


def test_agent(model_path, save_dir):
    """Test the trained agent and save videos."""
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    env = RecordVideo(env, video_folder=save_dir, name_prefix="lunar_test", episode_trigger=lambda x: True)

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
    """Run PPO training and testing."""
    train_ppo()
    test_agent("best_model_ppo.pth", "test_results_ppo")


if __name__ == "__main__":
    main()

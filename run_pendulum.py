# import gym
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import random

# Orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hiddnen_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hiddnen_dim)
        self.fc2 = nn.Linear(hiddnen_dim, hiddnen_dim)
        self.mu_head = nn.Linear(hiddnen_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        # self.log_std = nn.Linear(64, action_dim)
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.mu_head, gain=0.01)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = 2.0 * torch.tanh(self.mu_head(x))  # Output in [-2, 2]
        # mu = self.mu_head(x)
        # std = torch.exp(self.log_std(x))
        std = torch.exp(self.log_std)
        return mu, std

    def get_dist(self, s):
        mean, std = self.forward(s)
        return torch.distributions.Normal(mean, std)

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, hiddnen_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hiddnen_dim)
        self.fc2 = nn.Linear(hiddnen_dim, hiddnen_dim)
        self.fc3 = nn.Linear(hiddnen_dim, 1)
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, clip_epsilon=0.2, update_steps=10):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        # self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=5e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.K_epochs = update_steps
        self.entropy_coef = 0.01

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            mu, std = self.actor(state)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            # action = torch.clamp(action, -2, 2)
            return torch.clamp(action, -2, 2).cpu().numpy().flatten(), dist.log_prob(action).sum().item()

    def update(self, buffer):
        s, a, logprob_old, r, s_, dw, done = buffer.numpy_to_tensor()
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            adv, gae = [], 0
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * 0.95 * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs

        for _ in range(self.K_epochs):
            dist_now = self.actor.get_dist(s)
            entropy = dist_now.entropy().sum(1, keepdim=True)
            logprob_now = dist_now.log_prob(a).sum(1, keepdim=True)
            ratios = torch.exp(logprob_now - logprob_old)
            surr1 = ratios * adv
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv
            actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * entropy
            self.actor_optimizer.zero_grad()
            actor_loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
            self.actor_optimizer.step()

            critic_loss = F.mse_loss(v_target, self.critic(s))
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
            self.critic_optimizer.step()

# Replay Buffer
class ReplayBuffer:
    def __init__(self, batch_size, state_dim, action_dim):
        self.s = np.zeros((batch_size, state_dim))
        self.a = np.zeros((batch_size, action_dim))
        self.logprob = np.zeros((batch_size, 1))
        self.r = np.zeros((batch_size, 1))
        self.s_ = np.zeros((batch_size, state_dim))
        self.dw = np.zeros((batch_size, 1))
        self.done = np.zeros((batch_size, 1))
        self.count = 0

    def store(self, s, a, logprob, r, s_, dw, done):
        idx = self.count
        self.s[idx], self.a[idx], self.logprob[idx] = s, a, logprob
        self.r[idx], self.s_[idx], self.dw[idx], self.done[idx] = r, s_, dw, done
        self.count += 1

    def numpy_to_tensor(self):
        return tuple(torch.tensor(arr, dtype=torch.float32) for arr in
                     [self.s, self.a, self.logprob, self.r, self.s_, self.dw, self.done])

# State Normalization
class Normalizer:
    def __init__(self, num_inputs):
        self.n = torch.zeros(num_inputs)
        self.mean = torch.zeros(num_inputs)
        self.mean_diff = torch.zeros(num_inputs)
        self.var = torch.zeros(num_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.clone()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = torch.clamp(self.mean_diff / self.n, min=1e-2)

    def normalize(self, inputs):
        obs_std = torch.sqrt(self.var)
        return (inputs - self.mean) / obs_std


# Training Loop
# env = gym.make('Pendulum-v1')
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
buffer_size = 128
env = gym.make("Pendulum-v1", g=10)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = PPOAgent(state_dim, action_dim)
buffer = ReplayBuffer(buffer_size, state_dim, action_dim)
# state_normalizer = Normalizer(state_dim)

for episode in range(1500):
    state, _ = env.reset()
    # state_normalizer.observe(torch.from_numpy(state))
    # state = state_normalizer.normalize(torch.from_numpy(state)).numpy()
    episode_reward = 0
    for step in range(200):
        action, logprob = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)

        # state_normalizer.observe(torch.from_numpy(next_state))
        # next_state = state_normalizer.normalize(torch.from_numpy(next_state)).numpy()

        buffer.store(state, action, logprob, reward, next_state, done, done)
        state = next_state
        episode_reward += reward
        if buffer.count == buffer_size:
            agent.update(buffer)
            buffer.count = 0
        if done:
            break
    print(f"Episode {episode}, Reward: {episode_reward:.2f}")

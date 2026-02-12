# import gym
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import datetime

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
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2, update_steps=10):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=5e-3)
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

def create_discrete_reward_function(num_bins=3, reward_type='binary'):
    """Create discrete reward functions with different granularities"""
    def discrete_reward_fn(continuous_reward):
        if reward_type == 'binary':
            return 1.0 if continuous_reward > -200 else -1.0
        
        elif reward_type == 'ordinal':
            # Better binning for pendulum rewards  
            bins = np.linspace(-1600, 0, num_bins + 1)
            bin_values = np.linspace(-1.0, 1.0, num_bins)
            
            for i in range(len(bins) - 1):
                if bins[i] <= continuous_reward < bins[i + 1]:
                    return bin_values[i]
            return bin_values[-1]
        
        elif reward_type == 'threshold':
            if continuous_reward > -100:
                return 1.0
            elif continuous_reward > -400:
                return 0.5
            elif continuous_reward > -800:
                return 0.0
            else:
                return -0.5
    
    return discrete_reward_fn

def train_agent_with_reward_function(env, agent, buffer, reward_fn, episodes=500, max_steps=200, buffer_size=128):
    """Train agent with specified reward function"""
    episode_rewards = []
    smoothed_rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action, logprob = agent.select_action(state)
            next_state, continuous_reward, done, _, _ = env.step(action)
            
            # Apply reward transformation if specified
            if reward_fn is not None:
                transformed_reward = reward_fn(continuous_reward)
            else:
                transformed_reward = continuous_reward
            
            buffer.store(state, action, logprob, transformed_reward, next_state, done, done)
            state = next_state
            episode_reward += continuous_reward  # Track original reward for comparison
            
            if buffer.count == buffer_size:
                agent.update(buffer)
                buffer.count = 0
            if done:
                break
        
        episode_rewards.append(episode_reward)
        smoothed_rewards.append(np.mean(episode_rewards[-100:]))
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}: Average Reward = {avg_reward:.2f}")
    
    return episode_rewards, smoothed_rewards

def run_comparison_experiment():
    """Run comparison between continuous and discrete rewards"""
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    # Environment setup
    env = gym.make("Pendulum-v1", g=10)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    buffer_size = 128
    
    print("Starting Continuous vs Discrete Reward Comparison on Pendulum-v1")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print("-" * 60)
    
    # Experiment configurations
    configs = [
        ("Continuous", None),
        ("Binary", create_discrete_reward_function(reward_type='binary')),
        ("Ordinal-3", create_discrete_reward_function(num_bins=3, reward_type='ordinal')),
        ("Ordinal-5", create_discrete_reward_function(num_bins=5, reward_type='ordinal')),
        ("Threshold", create_discrete_reward_function(reward_type='threshold'))
    ]
    
    results = {}
    
    # Run experiments
    for config_name, reward_fn in configs:
        print(f"\nTraining with {config_name} rewards...")
        
        # Reset random seeds for fair comparison
        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)
        
        # Create fresh environment and agent
        env = gym.make("Pendulum-v1", g=10)
        agent = PPOAgent(state_dim, action_dim)
        buffer = ReplayBuffer(buffer_size, state_dim, action_dim)
        
        episode_rewards, smoothed_rewards = train_agent_with_reward_function(
            env, agent, buffer, reward_fn, episodes=800, max_steps=200, buffer_size=buffer_size
        )
        
        results[config_name] = {
            'episode_rewards': episode_rewards,
            'smoothed_rewards': smoothed_rewards,
            'final_performance': np.mean(episode_rewards[-100:])
        }
        
        print(f"Final 100-episode average: {results[config_name]['final_performance']:.2f}")
        env.close()
    
    # Plot results
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Learning curves (smoothed)
    plt.subplot(2, 2, 1)
    for config_name in results.keys():
        plt.plot(results[config_name]['smoothed_rewards'], 
                label=f"{config_name}", linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward (100-episode avg)')
    plt.title('Learning Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Raw episode rewards (last 200 episodes)
    plt.subplot(2, 2, 2)
    for config_name in results.keys():
        recent_rewards = results[config_name]['episode_rewards'][-200:]
        plt.plot(range(len(recent_rewards)), recent_rewards, 
                label=f"{config_name}", alpha=0.7)
    plt.xlabel('Recent Episodes (Last 200)')
    plt.ylabel('Episode Reward')
    plt.title('Recent Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Final performance comparison (bar chart)
    plt.subplot(2, 2, 3)
    config_names = list(results.keys())
    final_perfs = [results[name]['final_performance'] for name in config_names]
    colors = ['green', 'red', 'orange', 'blue', 'purple']
    
    bars = plt.bar(config_names, final_perfs, color=colors, alpha=0.7)
    plt.ylabel('Final Average Reward')
    plt.title('Final Performance Comparison')
    plt.xticks(rotation=45)
    
    for bar, perf in zip(bars, final_perfs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{perf:.1f}', ha='center', va='bottom')
    
    # Plot 4: Convergence analysis
    plt.subplot(2, 2, 4)
    convergence_window = 50
    for config_name in results.keys():
        rewards = results[config_name]['smoothed_rewards']
        final_variance = np.var(rewards[-convergence_window:])
        plt.scatter(final_variance, results[config_name]['final_performance'], 
                   s=100, label=config_name)
    
    plt.xlabel('Final Variance (Stability)')
    plt.ylabel('Final Performance')
    plt.title('Performance vs Stability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("plots", f"pendulum_comparison_{timestamp}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    best_config = max(results.keys(), key=lambda x: results[x]['final_performance'])
    print(f"Best performing configuration: {best_config}")
    print(f"Best final performance: {results[best_config]['final_performance']:.2f}")
    
    print("\nPerformance ranking:")
    sorted_configs = sorted(results.keys(), 
                          key=lambda x: results[x]['final_performance'], 
                          reverse=True)
    
    for i, config in enumerate(sorted_configs, 1):
        perf = results[config]['final_performance']
        print(f"{i}. {config}: {perf:.2f}")
    
    # Calculate performance gap
    continuous_perf = results['Continuous']['final_performance']
    print(f"\nPerformance gaps relative to continuous reward:")
    for config_name in results.keys():
        if config_name != 'Continuous':
            gap = results[config_name]['final_performance'] - continuous_perf
            gap_pct = (gap / abs(continuous_perf)) * 100 if continuous_perf != 0 else 0
            print(f"{config_name}: {gap:.2f} ({gap_pct:.1f}%)")
    
    return results

# Original Training Loop (kept for backwards compatibility)
def run_original_training():
    """Run the original training loop with continuous rewards"""
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    buffer_size = 128
    env = gym.make("Pendulum-v1", g=10)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(state_dim, action_dim)
    buffer = ReplayBuffer(buffer_size, state_dim, action_dim)

    for episode in range(1500):
        state, _ = env.reset()
        episode_reward = 0
        for step in range(200):
            action, logprob = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            buffer.store(state, action, logprob, reward, next_state, done, done)
            state = next_state
            episode_reward += reward
            if buffer.count == buffer_size:
                agent.update(buffer)
                buffer.count = 0
            if done:
                break
        print(f"Episode {episode}, Reward: {episode_reward:.2f}")

if __name__ == "__main__":
    # Choose which training to run
    mode = input("Choose mode: (1) Original training, (2) Comparison experiment: ")
    
    if mode == "1":
        run_original_training()
    elif mode == "2":
        run_comparison_experiment()
    else:
        print("Invalid choice, running comparison experiment by default...")
        run_comparison_experiment()
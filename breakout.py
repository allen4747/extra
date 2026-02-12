import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from PIL import Image
import cv2

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class FrameProcessor:
    """Preprocess Atari frames for training"""
    def __init__(self, frame_size=(84, 84), frame_stack=4):
        self.frame_size = frame_size
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
        
    def reset(self):
        self.frames.clear()
        
    def process_frame(self, frame):
        # Convert to grayscale and resize
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32) / 255.0
        return frame
        
    def get_state(self, frame):
        processed_frame = self.process_frame(frame)
        self.frames.append(processed_frame)
        
        # Pad with zeros if not enough frames
        while len(self.frames) < self.frame_stack:
            self.frames.append(np.zeros(self.frame_size, dtype=np.float32))
            
        return np.stack(self.frames, axis=0)

class DQN(nn.Module):
    """Deep Q-Network for Atari games"""
    def __init__(self, input_channels=4, num_actions=4):
        super(DQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate conv output size
        conv_out_size = self._get_conv_out_size(input_channels)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
    def _get_conv_out_size(self, input_channels):
        # Helper to calculate output size after conv layers
        dummy_input = torch.zeros(1, input_channels, 84, 84)
        with torch.no_grad():
            dummy_output = self.conv(dummy_input)
        return dummy_output.numel()
    
    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size(0), -1)  # Flatten
        return self.fc(conv_out)

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.BoolTensor(dones).to(device)
        )
        
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent with Double DQN and target network"""
    def __init__(self, state_channels, num_actions, lr=1e-4, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=100000):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        
        # Networks
        self.q_network = DQN(state_channels, num_actions).to(device)
        self.target_network = DQN(state_channels, num_actions).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Initialize target network
        self.update_target_network()
        
        # Replay buffer
        self.memory = ReplayBuffer()
        
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training:
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                     np.exp(-self.steps_done / self.epsilon_decay)
            self.steps_done += 1
            
            if random.random() < epsilon:
                return random.randrange(self.num_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network(state_tensor)
            return q_values.max(1)[1].item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def update(self, batch_size=32):
        """Update the Q-network"""
        if len(self.memory) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use main network to select actions, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).max(1)[1]
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * (~dones).unsqueeze(1))
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

def create_discrete_reward_functions():
    """Create different discrete reward functions for Breakout"""
    
    def binary_reward(score_change):
        """Binary: positive/negative score change"""
        return 1.0 if score_change > 0 else 0.0
    
    def binary_threshold_reward(score_change):
        """Binary with threshold: significant score change vs minor/none"""
        return 1.0 if score_change >= 4 else 0.0  # Breakout: 1-7 points per brick
    
    def ordinal_3_reward(score_change):
        """3-level ordinal: no gain, small gain, big gain"""
        if score_change == 0:
            return 0.0
        elif score_change <= 4:
            return 0.5
        else:
            return 1.0
    
    def ordinal_5_reward(score_change):
        """5-level ordinal: more granular scoring"""
        if score_change == 0:
            return 0.0
        elif score_change == 1:
            return 0.25
        elif score_change <= 4:
            return 0.5
        elif score_change <= 7:
            return 0.75
        else:
            return 1.0
    
    def milestone_reward(score_change, cumulative_score):
        """Milestone-based: rewards for reaching score thresholds"""
        milestones = [10, 50, 100, 200, 300, 400]
        prev_score = cumulative_score - score_change
        
        # Check if we crossed a milestone
        for milestone in milestones:
            if prev_score < milestone <= cumulative_score:
                return 2.0  # Big reward for milestone
        
        return 1.0 if score_change > 0 else 0.0
    
    return {
        'continuous': None,
        'binary': binary_reward,
        'binary_threshold': binary_threshold_reward,
        'ordinal_3': ordinal_3_reward,
        'ordinal_5': ordinal_5_reward,
        'milestone': milestone_reward
    }

def train_agent(env, agent, frame_processor, reward_fn, reward_name, 
                episodes=2000, max_steps=10000, update_freq=4, target_update_freq=1000):
    """Train DQN agent with specified reward function"""
    
    episode_rewards = []
    episode_scores = []  # Actual game scores
    smoothed_rewards = []
    losses = []
    
    print(f"Training with {reward_name} rewards...")
    
    for episode in range(episodes):
        frame, _ = env.reset()
        frame_processor.reset()
        state = frame_processor.get_state(frame)
        
        episode_reward = 0
        episode_score = 0
        total_loss = 0
        loss_count = 0
        step = 0
        
        while step < max_steps:
            # Select action
            action = agent.select_action(state)
            
            # Execute action
            next_frame, score_change, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Process next state
            next_state = frame_processor.get_state(next_frame)
            
            # Calculate reward based on function type
            if reward_fn is None:
                # Continuous: use raw score change
                reward = float(score_change)
            elif reward_name == 'milestone':
                # Special case: needs cumulative score
                episode_score += score_change
                reward = reward_fn(score_change, episode_score)
                episode_score -= score_change  # Reset for next calculation
            else:
                reward = reward_fn(score_change)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Update agent
            if step % update_freq == 0:
                loss = agent.update()
                if loss is not None:
                    total_loss += loss
                    loss_count += 1
            
            # Update target network
            if step % target_update_freq == 0:
                agent.update_target_network()
            
            # Track metrics
            episode_reward += score_change  # Always track actual score for comparison
            episode_score += score_change
            state = next_state
            step += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_scores.append(episode_score)
        smoothed_rewards.append(np.mean(episode_rewards[-100:]))
        
        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        losses.append(avg_loss)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_score = np.mean(episode_scores[-100:])
            current_epsilon = agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * \
                            np.exp(-agent.steps_done / agent.epsilon_decay)
            
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Score: {avg_score:.2f}")
            print(f"  Epsilon: {current_epsilon:.3f}")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Steps: {agent.steps_done}")
            print()
    
    return {
        'episode_rewards': episode_rewards,
        'episode_scores': episode_scores,
        'smoothed_rewards': smoothed_rewards,
        'losses': losses,
        'final_performance': np.mean(episode_rewards[-100:]),
        'final_score': np.mean(episode_scores[-100:])
    }

def evaluate_agent(env, agent, frame_processor, num_episodes=10):
    """Evaluate trained agent"""
    eval_scores = []
    
    for episode in range(num_episodes):
        frame, _ = env.reset()
        frame_processor.reset()
        state = frame_processor.get_state(frame)
        
        episode_score = 0
        step = 0
        
        while step < 10000:  # Max steps per episode
            action = agent.select_action(state, training=False)
            next_frame, score_change, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_state = frame_processor.get_state(next_frame)
            episode_score += score_change
            state = next_state
            step += 1
            
            if done:
                break
        
        eval_scores.append(episode_score)
    
    return eval_scores

def run_breakout_comparison():
    """Run comprehensive comparison on Breakout"""
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    # Environment setup
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array", frameskip=1)
    frame_processor = FrameProcessor()
    
    # Get action space size
    num_actions = env.action_space.n
    print(f"Breakout environment setup complete")
    print(f"Action space: {num_actions}")
    print(f"Observation space: {env.observation_space.shape}")
    print("-" * 60)
    
    # Get reward functions
    reward_functions = create_discrete_reward_functions()
    results = {}
    
    # Training parameters
    episodes = 1500  # Reduced for faster execution
    
    for reward_name, reward_fn in reward_functions.items():
        print(f"\n{'='*60}")
        print(f"Training with {reward_name.upper()} rewards")
        print(f"{'='*60}")
        
        # Reset random seeds for fair comparison
        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)
        
        # Create fresh agent and frame processor
        agent = DQNAgent(state_channels=4, num_actions=num_actions)
        frame_processor = FrameProcessor()
        
        # Train agent
        result = train_agent(
            env, agent, frame_processor, reward_fn, reward_name, 
            episodes=episodes, max_steps=5000
        )
        
        # Evaluate agent
        print("Evaluating trained agent...")
        eval_scores = evaluate_agent(env, agent, frame_processor, num_episodes=20)
        result['eval_scores'] = eval_scores
        result['eval_mean'] = np.mean(eval_scores)
        result['eval_std'] = np.std(eval_scores)
        
        results[reward_name] = result
        
        print(f"Final training performance: {result['final_performance']:.2f}")
        print(f"Evaluation performance: {result['eval_mean']:.2f} ± {result['eval_std']:.2f}")
    
    # Plot comprehensive results
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Breakout: Continuous vs Discrete Rewards Comparison', fontsize=16)
    
    # Color scheme
    colors = ['green', 'red', 'orange', 'blue', 'purple', 'brown']
    color_map = {name: colors[i] for i, name in enumerate(results.keys())}
    
    # Plot 1: Learning curves (smoothed rewards)
    ax1 = axes[0, 0]
    for reward_name, result in results.items():
        ax1.plot(result['smoothed_rewards'], 
                label=reward_name.replace('_', ' ').title(),
                color=color_map[reward_name], linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Smoothed Score (100-ep avg)')
    ax1.set_title('Learning Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training loss curves
    ax2 = axes[0, 1]
    for reward_name, result in results.items():
        # Smooth losses for visualization
        losses = result['losses']
        if len(losses) > 100:
            smoothed_losses = [np.mean(losses[max(0, i-50):i+50]) for i in range(len(losses))]
        else:
            smoothed_losses = losses
        ax2.plot(smoothed_losses[:len(result['smoothed_rewards'])], 
                label=reward_name.replace('_', ' ').title(),
                color=color_map[reward_name], alpha=0.7)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Training Loss (Smoothed)')
    ax2.set_title('Training Loss Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final performance comparison
    ax3 = axes[1, 0]
    reward_names = list(results.keys())
    final_performances = [results[name]['final_performance'] for name in reward_names]
    bars = ax3.bar(range(len(reward_names)), final_performances,
                  color=[color_map[name] for name in reward_names], alpha=0.7)
    ax3.set_xlabel('Reward Function')
    ax3.set_ylabel('Final Training Score')
    ax3.set_title('Final Training Performance')
    ax3.set_xticks(range(len(reward_names)))
    ax3.set_xticklabels([name.replace('_', ' ').title() for name in reward_names], rotation=45)
    
    # Add value labels on bars
    for bar, perf in zip(bars, final_performances):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{perf:.1f}', ha='center', va='bottom')
    
    # Plot 4: Evaluation performance with error bars
    ax4 = axes[1, 1]
    eval_means = [results[name]['eval_mean'] for name in reward_names]
    eval_stds = [results[name]['eval_std'] for name in reward_names]
    bars = ax4.bar(range(len(reward_names)), eval_means, yerr=eval_stds,
                  color=[color_map[name] for name in reward_names], alpha=0.7, capsize=5)
    ax4.set_xlabel('Reward Function')
    ax4.set_ylabel('Evaluation Score')
    ax4.set_title('Evaluation Performance (20 episodes)')
    ax4.set_xticks(range(len(reward_names)))
    ax4.set_xticklabels([name.replace('_', ' ').title() for name in reward_names], rotation=45)
    
    # Plot 5: Sample efficiency (episodes to reach threshold)
    ax5 = axes[2, 0]
    threshold_score = 50  # Reasonable threshold for Breakout
    episodes_to_threshold = []
    
    for reward_name in reward_names:
        smoothed = results[reward_name]['smoothed_rewards']
        episodes_to_reach = len(smoothed)  # Default if never reached
        for i, score in enumerate(smoothed):
            if score >= threshold_score:
                episodes_to_reach = i
                break
        episodes_to_threshold.append(episodes_to_reach)
    
    bars = ax5.bar(range(len(reward_names)), episodes_to_threshold,
                  color=[color_map[name] for name in reward_names], alpha=0.7)
    ax5.set_xlabel('Reward Function')
    ax5.set_ylabel(f'Episodes to Reach Score {threshold_score}')
    ax5.set_title('Sample Efficiency')
    ax5.set_xticks(range(len(reward_names)))
    ax5.set_xticklabels([name.replace('_', ' ').title() for name in reward_names], rotation=45)
    
    # Plot 6: Score distribution comparison (recent episodes)
    ax6 = axes[2, 1]
    recent_episodes = 200
    for reward_name, result in results.items():
        recent_scores = result['episode_rewards'][-recent_episodes:]
        ax6.hist(recent_scores, bins=20, alpha=0.5, 
                label=reward_name.replace('_', ' ').title(),
                color=color_map[reward_name])
    ax6.set_xlabel('Episode Score')
    ax6.set_ylabel('Frequency')
    ax6.set_title(f'Score Distribution (Last {recent_episodes} Episodes)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print("\n" + "="*80)
    print("BREAKOUT EXPERIMENT ANALYSIS")
    print("="*80)
    
    # Performance ranking
    sorted_by_eval = sorted(results.items(), key=lambda x: x[1]['eval_mean'], reverse=True)
    sorted_by_training = sorted(results.items(), key=lambda x: x[1]['final_performance'], reverse=True)
    
    print("\nEVALUATION PERFORMANCE RANKING:")
    print("-" * 40)
    for i, (name, result) in enumerate(sorted_by_eval, 1):
        print(f"{i}. {name.replace('_', ' ').title()}: {result['eval_mean']:.2f} ± {result['eval_std']:.2f}")
    
    print("\nTRAINING PERFORMANCE RANKING:")
    print("-" * 40)
    for i, (name, result) in enumerate(sorted_by_training, 1):
        print(f"{i}. {name.replace('_', ' ').title()}: {result['final_performance']:.2f}")
    
    # Statistical significance analysis
    print(f"\nSAMPLE EFFICIENCY (Episodes to reach score {threshold_score}):")
    print("-" * 40)
    for name in reward_names:
        smoothed = results[name]['smoothed_rewards']
        episodes_to_reach = len(smoothed)
        for i, score in enumerate(smoothed):
            if score >= threshold_score:
                episodes_to_reach = i
                break
        status = "✓" if episodes_to_reach < len(smoothed) else "✗"
        print(f"{name.replace('_', ' ').title()}: {episodes_to_reach} episodes {status}")
    
    # Hypothesis validation
    continuous_eval = results['continuous']['eval_mean']
    print(f"\nHYPOTHESIS VALIDATION:")
    print("-" * 40)
    print(f"Continuous reward performance: {continuous_eval:.2f}")
    print("Performance gaps:")
    
    for name, result in results.items():
        if name != 'continuous':
            gap = result['eval_mean'] - continuous_eval
            gap_pct = (gap / continuous_eval) * 100 if continuous_eval != 0 else 0
            print(f"  {name.replace('_', ' ').title()}: {gap:+.2f} ({gap_pct:+.1f}%)")
    
    env.close()
    return results

if __name__ == "__main__":
    print("Starting Breakout Continuous vs Discrete Reward Comparison")
    print("This may take several hours to complete...")
    print("Consider reducing episodes if you want faster results")
    
    results = run_breakout_comparison()
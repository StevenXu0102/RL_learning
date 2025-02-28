import argparse
from itertools import count
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=2):
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.6)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

        self.log_probs_saved = []
        self.rewards = []

    
    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.layer2(x)
        return F.softmax(x, dim=1)

def select_action(state, policy):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action_probs = policy(state)
    dist = Categorical(action_probs)
    action = dist.sample()
    policy.log_probs_saved.append(dist.log_prob(action))
    return action.item()

def update_policy(policy, optimizer, gamma):
    eps = np.finfo(np.float32).eps.item()
    discounted_rewards = []
    R = 0
    
    # Compute discounted rewards in reverse order
    for reward in reversed(policy.rewards):
        R = reward + gamma * R
        discounted_rewards.insert(0, R)
    
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + eps)
    policy_loss = []
    for log_prob, R in zip(policy.log_probs_saved, discounted_rewards):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

    del policy.rewards[:]
    del policy.log_probs_saved[:]

def train(env, policy, optimizer, args):
    running_reward = 10
    for episode in count(1):
        state, _ = env.reset(seed=args.seed + episode)
        episode_reward = 0
        for t in range(1, 10000):
            action = select_action(state, policy)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if args.render:
                env.render()
            policy.rewards.append(reward)
            episode_reward += reward
            if done:
                break
        running_reward = 0.05 * episode_reward + 0.95 * running_reward
        update_policy(policy, optimizer, args.gamma)

        if episode % args.log_interval == 0:
            print(f'Episode {episode}	Last reward: {episode_reward:.2f}	Average reward: {running_reward:.2f}')
        
        if running_reward > env.spec.reward_threshold:
            print(f"Solved! Running reward is now {running_reward:.2f} and the last episode runs for {t} time steps!")
            break

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    env = gym.make('CartPole-v1', render_mode='human' if args.render else None)
    torch.manual_seed(args.seed)
    
    policy = PolicyNetwork()
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    
    train(env, policy, optimizer, args)
    
    env.close()


if __name__ == '__main__':
    main()

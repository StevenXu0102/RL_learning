import gymnasium as gym
import argparse
from itertools import count
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch policy gradient for OpenAI Gym Pong')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=20, help='episodes per parameter update')
    parser.add_argument('--seed', type=int, default=87, help='random seed (default: 87)')
    parser.add_argument('--test', action='store_true', help='test mode (no training)')
    return parser.parse_args()

# Preprocess game image
def preprocess_image(I):
    """ Preprocess 210x160x3 image into 6400 1D array """
    I = I[35:195]  # Crop
    I = I[::2, ::2, 0]  # Downsample by factor of 2
    I[(I == 144) | (I == 109)] = 0  # Erase background (various shades of brown)
    I[I != 0] = 1  # Set paddles and ball to 1
    return I.astype(np.float32).ravel()

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim = 6400, hidden_dim = 200, num_actions = 2):
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, num_actions)
        self.log_probs_saved = []
        self.rewards = []
    
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        action_scores = self.layer2(x)
        return F.softmax(action_scores, dim=1)
    
    def select_action(self, x):
        x = Variable(torch.from_numpy(x).float().unsqueeze(0))
        probs = self.forward(x)
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs_saved.append(dist.log_prob(action))
        return action.item()
    
def update_policy(policy:PolicyNetwork, optimizer, gamma):
    discounted_return = []
    step_return = 0
    for step_reward in reversed(policy.rewards):
        if (step_reward != 0.0):
            step_return = 0.0
        step_return = step_reward + gamma * step_return
        discounted_return.insert(0, step_return)
    
    discounted_return = torch.Tensor(discounted_return)
    eps = np.finfo(np.float32).eps.item()
    discounted_return = (discounted_return - discounted_return.mean()) / (discounted_return.std() + eps)
    log_probs = torch.concatenate(policy.log_probs_saved)
    policy_loss = - (log_probs * discounted_return).mean()
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    policy.rewards.clear()
    policy.log_probs_saved.clear()

def train(env, policy, optimizer, args):
    running_reward = None
    for i_episode in count(1):
        state, _ = env.reset(seed=args.seed + i_episode)
        pre_x = None
        reward_sum = 0

        for t in range(10000):
            cur_x = preprocess_image(state)
            x = cur_x - pre_x if pre_x is not None else np.zeros(80 * 80)
            pre_x = cur_x
            # Adjust to Pong action space
            action = policy.select_action(x) + 2
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward_sum += reward
            policy.rewards.append(reward)
            
            if done:
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print(f'EP {i_episode:03d} done. Reward: {reward_sum}. Running mean: {running_reward}')
                break
        
        if i_episode % args.batch_size == 0:
            update_policy(policy, optimizer, args.gamma)
        
        if i_episode % 50 == 0:
            print(f'Saving model at episode {i_episode}...')
            torch.save(policy.state_dict(), 'pg_params.pkl')



# Main entry point
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    env = gym.make("ale_py:ALE/Pong-v5", render_mode='human' if args.test else None)
    policy = PolicyNetwork()
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)
    
    if os.path.exists('pg_params.pkl'):
        print('Loading pre-trained model...')
        policy.load_state_dict(torch.load('pg_params.pkl'))
    
    train(env, policy, optimizer, args)

if __name__ == '__main__':
    main()

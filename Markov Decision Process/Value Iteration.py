import gymnasium as gym
import numpy as np

def run_episode(env, policy, gamma=1.0):
    obs, _ = env.reset()
    total_reward = 0
    step_count = 0
    while True:
        action = int(policy[obs])
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += (gamma ** step_count) * reward
        step_count += 1
        if done:
            break
    return total_reward

def get_policy(env, value_function, gamma=1.0):
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        state_action_values = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            transitions = env.P[state][action]
            state_action_values[action] = sum(
                prob * (reward + gamma * value_function[next_state])
                for prob, next_state, reward, _ in transitions
            )
        policy[state] = np.argmax(state_action_values)
    return policy

def evaluate_policy(env, policy, gamma=1.0, n=200):
    """Evaluate the policy by running it for n episodes and returning the average reward."""
    scores = [run_episode(env, policy, gamma) for _ in range(n)]
    return np.mean(scores)

def value_iteration(env, gamma=1.0):
    value_function = np.zeros(env.observation_space.n)
    max_iterations = 200000
    eps = 1e-20
    for i in range(max_iterations):
        prev_value_function = np.copy(value_function)
        for state in range(env.observation_space.n):
            state_action_values = [
                sum([prob * (reward + gamma * prev_value_function[next_state]) 
                     for prob, next_state, reward, _ in env.P[state][action]]) 
                     for action in range(env.action_space.n)
            ]
            value_function[state] = max(state_action_values)
        if (np.sum(np.fabs(prev_value_function - value_function)) <= eps):
            print(f'Value Iteration converged at step {i + 1}.')
            break
    return value_function

if __name__ == '__main__':
    render = False
    env_name = 'FrozenLake8x8-v1' 
    if render:
        env = gym.make(env_name, render_mode='human')
    else:
        env = gym.make(env_name)
    env = env.unwrapped
    gamma = 1.0
    optimal_value = value_iteration(env, gamma)
    optimal_policy = get_policy(env, optimal_value, gamma)
    optimal_scores = evaluate_policy(env, optimal_policy, gamma, 1000)
    print('Average scores = ', np.mean(optimal_scores))
    env.close()
            
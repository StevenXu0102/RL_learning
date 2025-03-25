import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

def Q_learning(env, num_train_epoch, num_iter, learning_rate, discount, eps):
    Q_learning_table = np.zeros([env.observation_space.n, env.action_space.n])
    for epoch in range(num_train_epoch):
        state, _ = env.reset()
        for iter in range(num_iter):
            # Epsilon greedy
            if (np.random.uniform(0, 1) < eps):
                action = np.random.choice(env.action_space.n)
            else:
                action = np.argmax(Q_learning_table[state, :])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            Q_learning_table[state, action] = Q_learning_table[state, action] + learning_rate * (reward +  discount * np.max(Q_learning_table[next_state, :]) - Q_learning_table[state, action])
            state = next_state
            if done:
                break
    return Q_learning_table


        
# how to use 2 nns, one for game state and one for decision based on first nn (not actor-critic?

import gymnasium as gym
import numpy as np

episodes = 1
learning_rate = 0.05

class Policy:
    def __init__(self, theta):
        self.theta = theta
    def choose_action(self, observation):
        return env.action_space.sample()
    
    
env = gym.make("ALE/Pong-v5", render_mode="human")

observation, info = env.reset()

done = False
episode_number = 0
policy = Policy(0)

while episode_number < episodes:
    action = policy.choose_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)

    print(reward)
    
    if terminated or truncated:
        env.reset()
        episode_number += 1

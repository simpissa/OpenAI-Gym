import ale_py
import shimmy
import gymnasium as gym
import numpy as np
import keyboard
import pickle
import matplotlib.pyplot as plt

from pong import MLP, preprocess

with ('model.pickle', 'rb') as file:
    policy = pickle.load(file)
env = gym.make("ALE/Pong-v5", render_mode="human", obs_type="grayscale")
observation, info = env.reset()
episode_number = 0
frame_number = 0
while episode_number < 1:
    observation = preprocess(observation) if frame_number > 1 else 0
    if frame_number < 2:
        action = env.action_space.sample()
    else:
        x = observation-prev_observation
        action, y = policy.choose_action(x)
        
    prev_observation = observation
    observation, reward, terminated, truncated, info = env.step(action)
    frame_number += 1
    
    if reward != 0:
        frame_number = 0

    if terminated or truncated:
        env.reset()
        episode_number += 1
        
env.close()
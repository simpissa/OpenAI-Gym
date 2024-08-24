# how to use 2 nns, one for game state and one for decision based on first nn (not actor-critic?

import gymnasium as gym
import numpy as np
import keyboard
import pickle
np.set_printoptions(threshold=np.inf)
episodes = 100000
learning_rate = 0.00002
# input_length = 33600 # 210 * 160
input_length = 25440
hidden_layer = 200

def save():
    with open('model.pickle', 'wb') as file:
        pickle.dump(policy, file)

keyboard.add_hotkey('p', lambda: save())

class Policy:
    def __init__(self):
        self.w1 = np.random.randn(hidden_layer, input_length) / np.sqrt(input_length)
        self.w2 = np.random.randn(hidden_layer) / np.sqrt(hidden_layer)
        self.b = np.random.randn(hidden_layer) / np.sqrt(hidden_layer)


    def forward(self, x):
        h = np.matmul(self.w1, x) + self.b
        h[h<0] = 0 # relu
        y = np.matmul(self.w2, h) # log probability

        return y, h

    def backprop(self, A, h, w1, w2, b, x):
        dw2 = A * h
        gradient = A * self.w2
        gradient[h<=0] = 0
        db = gradient
        dw1 = np.outer(gradient, x)

        w1 += dw1 
        w2 += dw2
        b += db

    def update_weights(self, dw1, dw2, db):
        self.w1 += dw1
        self.w2 += dw2
        self.b += db

        # print(np.max(self.w2))

    def choose_action(self, observation):
        y, h = self.forward(observation)
        x = 1.0 / (1.0 + np.exp(-y)) # probability of moving down
        print(str(x))
        if np.random.uniform() > x:
            return 2, h, x # up 
        else:
            return 3, h, x # down
    
def preprocess(frame):
    # erase everything except ball and paddles
    frame = frame[34:193]
    frame[frame==87] = 0
    frame[frame!=0] = 1
    # if frame_number == 20:
    #     file = open("test.txt", "w")
    #     file.write(str(frame))
    return np.ravel(frame)

def init_dw():
    dw1 = np.zeros((hidden_layer, input_length))
    dw2 = np.zeros(hidden_layer)
    db = np.zeros(hidden_layer)
    return dw1, dw2, db

env = gym.make("ALE/Pong-v5", render_mode="human", obs_type="grayscale")

observation, info = env.reset()

done = False
episode_number = 0

# policy = Policy()
with open('newlr0.00002.pickle', 'rb') as file:
    policy = pickle.load(file)

frame_number = 1
observation = 0
dw1, dw2, db = init_dw()

while episode_number < episodes:
    observation = preprocess(observation) if frame_number > 1 else 0
    # random action for first 2 frames
    if frame_number <= 2:
        action = env.action_space.sample()
    else:
        # forward pass
        action, h, y = policy.choose_action(observation-prev_observation)

    prev_observation = observation
    
    observation, reward, terminated, truncated, info = env.step(action)

    if frame_number > 2:
        dy = 1 if action == 3 else -1 # encourages this action
        # backprop
        policy.backprop(dy, h, dw1, dw2, db, prev_observation)

        if reward != 0:
            # point made
            print("Point")
            if reward == 1: # win
                if action == 2: # up
                    advantage = y
                else: # down
                    advantage = 1-y
            else: # loss
                if action == 2: # up
                    advantage = 1-y
                else: # down
                    advantage = y
            policy.update_weights(reward * advantage * dw1 * learning_rate, reward * advantage * dw2 * learning_rate, reward * advantage * db * learning_rate)
            dw1, dw2, db = init_dw()

    frame_number += 1
    if terminated or truncated:
        env.reset()
        frame_number = 1
        episode_number += 1

print(policy.w1)
print(policy.w2)
print(policy.b)
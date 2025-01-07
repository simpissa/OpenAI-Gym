import ale_py
import shimmy
import gymnasium as gym
import numpy as np
import keyboard
import pickle
import matplotlib.pyplot as plt

lr = 1e-4
discount_factor = 0.95
episodes = 500
beta_1 = 0.9
beta_2 = 0.99
epsilon = 1e-6
rng = np.random.default_rng(5)

class MLP:
    def __init__(self, input_len, hidden_layer):
        self.layers = {}
        self.layers['W1'] = rng.normal(size=(input_len, hidden_layer)) / (input_len**0.5) * 5/3
        self.layers['B1'] = rng.normal(size=(hidden_layer)) * 0.01
        self.layers['W2'] = rng.normal(size=(hidden_layer, 1)) / (hidden_layer**0.5)
        # self.B2 = np.random.randn(1) * 0.01
        self.dlayers = {}

        # tracking
        self.h = []
        # self.h2 = []
        # self.temp = []

        # adam
        self.num_updates = 0
        self.m = {k : np.zeros_like(v) for k, v in self.layers.items()} # momentum
        self.v = {k : np.zeros_like(v) for k, v in self.layers.items()} # running mean of gradient^2


    def forward(self, x):
        hidden2 = x @ self.layers['W1'] + self.layers['B1']
        
        hidden = np.tanh(hidden2)
        a = hidden @ self.layers['W2']

        # self.temp.append(hidden2)
        self.h.append(hidden)
        # self.h2.append(a.copy())

        return 1.0 / (1.0 + np.exp(-a)[0].astype(np.float32)) # sigmoid
    
    def backprop(self, x, probs, gradients):
        self.num_updates += 1

        self.h = np.array(self.h)
        # self.h2 = np.array(self.h2)
        
        gradients *= probs * (1-probs) # (batch, 1)
        # self.dB2 = gradients.sum(0) # (batch, 1).sum(0) = (1)
        self.dlayers['W2'] = self.h.transpose() @ gradients # (200, batch) @ (batch, 1) = (200, 1)
        gradients = gradients @ self.layers['W2'].transpose() * (1-self.h**2)# (batch, 1) @ (1, 200) = (batch, 200)
        self.dlayers['B1'] = gradients.sum(0) # (batch, 200).sum(0) = (200)
        self.dlayers['W1'] = x.transpose() @ gradients # (6400, batch) @ (batch, 200)

        # adam
        for k, _ in self.layers.items():
            self.m[k] = beta_1 * self.m[k] + (1-beta_1) * self.dlayers[k]
            self.v[k] = beta_2 * self.v[k] + (1-beta_2) * self.dlayers[k]**2
            mhat = self.m[k] / (1-beta_1**self.num_updates) # unbiased version
            vhat = self.v[k] / (1-beta_2**self.num_updates) # unbiased version
            self.layers[k] -= lr / (epsilon + np.sqrt(vhat)) * mhat

    def choose_action(self, observation):
        prob = self.forward(np.array(observation)) # probability of moving down
        if rng.uniform() > prob:
            return 2, prob # up 
        else:
            return 3, prob # down

    def reset_grad(self):
        self.h = []
        # self.h2 = []
        # self.temp = []

def preprocess(frame):
    # erase everything except ball and paddles
    frame = frame[34:193]
    frame = frame[::2, ::2]
    frame[frame==87] = 0
    frame[frame!=0] = 1

    return np.ravel(frame).astype(np.float32)

def discount_gradients(gradients):
    x = 1.0
    for i in range(len(gradients)-2, -1, -1):
        x *= discount_factor
        gradients[i][0] *= x

def run():
    env = gym.make("ALE/Pong-v5", obs_type="grayscale")
    observation, _ = env.reset()

    policy = MLP(6400, 200)
    # with open('model.pickle', 'rb') as file:
    #     policy = pickle.load(file)

    episode_number = 0
    game_number = 0
    frame_number = 0

    score = 0
    total = 0.0
    while episode_number < episodes:
        observation = preprocess(np.array(observation))
        if frame_number < 2:
            action = env.action_space.sample()
        else:
            x = observation-prev_observation
            if frame_number == 2 and episode_number % 10 == 0 and game_number == 0:
                policy.reset_grad()
                tempx = x.copy()
                tempx -= policy.running_mean
                tempx /= policy.running_std
                action, y = policy.choose_action(x)
                input = [x]
                output = [[y]]
                actions = [[1.0 if action==2 else -1.0]] # 1 gradient for up, -1 gradient for down
            else:
                action, y = policy.choose_action(x)
                input.append(x)
                output.append([y])
                if frame_number == 2:
                    actions = [[1.0 if action==2 else -1.0]]
                else:
                    actions.append([1.0 if action==2 else -1.0])


        prev_observation = observation
        observation, reward, terminated, truncated, _ = env.step(action)
        frame_number += 1
        
        if reward != 0:
            frame_number = 0
            actions = np.array(actions)
            rewards = np.vstack([reward]*actions.shape[0])
            discount_gradients(rewards)
            rewards -= rewards.mean()
            rewards /= rewards.std()
            actions *= rewards
            if game_number == 0 and episode_number % 10 == 0:
                gradients = actions.copy()
            else:
                gradients = np.vstack((gradients, actions))
            game_number += 1
            
            if reward == 1:
                score += 1

        if terminated or truncated:
            episode_number += 1

            if episode_number % 10 == 0 and episode_number > 0:
                input = np.array(input)
                policy.backprop(input, np.array(output), gradients)

            observation, _ = env.reset()
            print(f'Episode {episode_number}, Score {score}')
            
            total += score
            score = 0
            game_number = 0
    env.close()
    with open('model.pickle', 'wb') as file:
        pickle.dump(policy, file)
    print(total / episodes)

if __name__ == "__main__":
    run()
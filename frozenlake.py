import gymnasium as gym
import numpy as np

rng =  np.random.default_rng()

# hyperparameters

gamma = 0.95
learning_rate = 0.8
epsilon = 0.1

class MonteCarlo:
    def __init__(self, gamma, state_size, action_size):
        self.gamma = gamma
        self.state_size = state_size
        self.action_size = action_size
        self.reset_state_action_values()



    def update(self):
        print()
        
    def reset_state_action_values(self):
        self.state_action_values = np.zeros((self.state_size, self.action_size))

class QLearning: 
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.reset_table()
    
    def update(self, state, action, reward, new_state):
        self.qtable[state, action] = self.qtable[state, action] + learning_rate * (reward + gamma * np.max(self.qtable[new_state, :]) - self.qtable[state, action])

    def reset_table(self):
        self.qtable = np.zeros((self.state_size, self.action_size))


class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose_action(self, action_space, state, state_action_values):
        explore_exploit_tradeoff = rng.random()

        # exploration
        if explore_exploit_tradeoff < self.epsilon:
            action = action_space.sample()
        
        # exploitation
        else:
            # settle ties randomly
            tiebreaker = np.random.random(state_action_values.shape)
            state_action_values = state_action_values + tiebreaker

            action = np.argmax(state_action_values[state, :])

            state_action_values = state_action_values - tiebreaker

        return action
        
        

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")
observation, info = env.reset()

# for _ in range(1000):
    # action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated: 
#         observation, info = env.reset()

learner = QLearning(16, 4)
explorer = EpsilonGreedy(epsilon)

seed = 123
runs = 1000

for i in range(runs):
    state, info = env.reset(seed=seed)
    done = False

    while not done:

        action = explorer.choose_action(env.action_space, state, learner.qtable)

        new_state, reward, terminated, truncated, info = env.step(action)

        learner.update(state, action, reward, new_state)

        done = terminated or truncated

        state = new_state


env.close()

print(learner.qtable)

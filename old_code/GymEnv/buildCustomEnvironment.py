# adjust temperature
# optimal temperature: between 37 and 39 degrees
# shower length: 60 seconds
# actions: turn down, leave, turn up
# task: build a modol that keeps us in the optimal range for as long as possible 
from gymnasium import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

# class ShowerEnv(Env):
#     def __init__(self):
#         self.action_spac = Discrete()
#         self.observation_space = Box()
#         self.state = 0
#     def step(self, action):
#         return self.state, reward, done, info
#     def render(self):
#         pass
#     def reset(self):
#         return self.state

# env = ShowerEnv()
# env.action_spac.sample()
# env.observation_space.sample()

class ShowerEnv(Env):
    def __init__(self):
        self.action_spac = Discrete(3)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.state = 38 + random.randint(-3, 3)
        self.shower_length = 60
    def step(self, action):
        self.state += action -1
        self.shower_length -= 1
        if self.state >= 37 and self.state <=39:
            reward = 1
        else: 
            reward = -1
        if self.shower_length <= 0:
            done = True
        else:
            done = False
        # apply temperature noise
        self.state += random.randint(-1, 1)
        info = {}
        return self.state, reward, done, info

    def render(self):
        pass
    def reset(self):
        self.state = 38 + random.randint(-3, 3)
        self.shower_length = 60
        return self.state

env = ShowerEnv()
env.action_spac.sample()
env.observation_space.sample()
episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0
    while not done:
        env.render()
        action = env.action_spac.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    print(f'Episode: {episode}, Score: {score}')


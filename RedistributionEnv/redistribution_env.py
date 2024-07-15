import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from typing import Tuple


class RedistributionEnv(gym.Env):
    metadata = {'render.modes': ['ansi']}

    def __init__(self, num_docks=5, dock_max_bikes=20, max_steps_per_episode=30, max_bikes_on_truck=5):
        self.num_docks = num_docks
        self.dock_max_bikes = dock_max_bikes
        self.max_steps_per_episode = max_steps_per_episode
        self.max_bikes_on_truck = max_bikes_on_truck
        self.action_space = spaces.Discrete(num_docks + 2)
        self.actions = {f"move_to_{i}": i for i in range(num_docks)}
        self.actions["pickup"] = num_docks
        self.actions["dropoff"] = num_docks + 1
        self.observation_space = spaces.Dict({
            'truck_position': spaces.Discrete(num_docks),
            'bike_states': spaces.MultiDiscrete([dock_max_bikes + 1] * num_docks),
            'bikes_on_truck': spaces.Discrete(max_bikes_on_truck + 1)
        })
        self.distance_matrix = np.array([
            [0, 2, 5, 9, 4],
            [2, 0, 3, 7, 2],
            [5, 3, 0, 4, 3],
            [9, 7, 4, 0, 1],
            [4, 2, 3, 1, 0]
        ])
        self.lastaction = None
        self.reward_range = (-10, 80)
        self.state = None
        self.timestep = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.state = {
            'truck_position': random.randint(0, self.num_docks - 1),
            'bike_states': np.random.randint(0, self.dock_max_bikes + 1, size=self.num_docks),
            'bikes_on_truck': 0
        }  # state是字典格式，三个键值
        total_bikes = np.sum(self.state['bike_states'])
        # print(f"total_bikes is {total_bikes}")
        self.balanced_bikes = int(total_bikes / self.num_docks)
        # print(f"self.balanced_bikes is {self.balanced_bikes}")
        self.timestep = 0
        return self.state, {}

    def step(self, action) -> Tuple[dict, float, bool, bool, dict]:
        assert self.action_space.contains(action), "Invalid action"
        reward = -3  # 每个时间步的固定负奖励
        terminated = False
        truncated = False
        info = {}
        if action < self.num_docks:
            to_station = action
            from_station = self.state['truck_position']
            if from_station == to_station:
                reward -= 30  # 给一个小惩罚，鼓励卡车移动
            else:
                travel_cost = self.distance_matrix[from_station][to_station]
                reward -= travel_cost
                self.state['truck_position'] = to_station

        elif action == self.actions["pickup"]:  # Pickup bikes
            dock_index = self.state['truck_position']
            bikes_in_dock = self.state['bike_states'][dock_index]
            if bikes_in_dock > self.balanced_bikes and self.state['bikes_on_truck'] < self.max_bikes_on_truck:
                bikes_out = min(bikes_in_dock - self.balanced_bikes,
                                self.max_bikes_on_truck - self.state['bikes_on_truck'])
                self.state['bike_states'][dock_index] -= bikes_out
                self.state['bikes_on_truck'] += bikes_out
                reward += 10  # fixed reward
                # reward += 10 - bikes_out  # Encourage efficient redistribution
            else:
                reward -= 15

        elif action == self.actions["dropoff"]:  # Dropoff bikes
            dock_index = self.state['truck_position']
            bikes_in_dock = self.state['bike_states'][dock_index]
            if bikes_in_dock < self.balanced_bikes and self.state['bikes_on_truck'] > 0:
                bikes_needed = self.balanced_bikes - bikes_in_dock
                bikes_out = min(bikes_needed, self.state['bikes_on_truck'])
                self.state['bike_states'][dock_index] += bikes_out
                self.state['bikes_on_truck'] -= bikes_out
                reward += 10  # fixed reward
                # reward += 10 - bikes_out  # Encourage efficient redistribution
            else:
                reward -= 15

        # 经过测试这一步对获得最终奖励很重要。
        # Balance reward for individual stations. 每一步中检查所有站点状态并给予接近平衡状态的小额奖励。
        for bikes in self.state['bike_states']:
            if abs(bikes - self.balanced_bikes) <= 1:
                reward += 2  # Small reward for being close to balanced

        self.timestep += 1

        # Check for termination
        if np.all(np.abs(self.state['bike_states'] - self.balanced_bikes) <= 1) and self.state['bikes_on_truck'] <= 1:
            terminated = True
            reward += 500  # Large reward for achieving overall balance
        elif self.timestep >= self.max_steps_per_episode:
            truncated = True
            info['reason'] = 'max_steps_reached'

        return self.state, reward, terminated, truncated, info

    def render(self, mode='ansi'):
        if mode != 'ansi':
            raise NotImplementedError("Render mode not supported: {}".format(mode))

        output = ""
        truck_position = self.state['truck_position']
        bike_states = self.state['bike_states']
        bikes_on_truck = self.state['bikes_on_truck']

        nodes = ['A', 'B', 'C', 'D', 'E']
        for idx in range(len(nodes)):
            if idx == truck_position:
                output += f"Station {nodes[idx]}: 🚚, {bike_states[idx]} bikes\n"
            else:
                output += f"Station {nodes[idx]}: ✓, {bike_states[idx]} bikes\n"

        output += f"Bikes on Truck: {bikes_on_truck}\n"
        output += f"Current Timestep: {self.timestep}/{self.max_steps_per_episode}\n"

        print(output)

# # test
# # 创建环境实例
# env = RedistributionEnv()
# state = env.reset(66)
# print("Initial State:")
# env.render()  # 渲染初始状态
# print('.......................................')
# total_reward1 = 0
# action = env.actions['pickup']
# state, reward, terminated, truncated, info = env.step(action)
# total_reward1 += reward
# print(f'Action: {action}, Reward: {reward}, Terminated: {terminated}')
# env.render()
# action = env.actions['move_to_1']
# state, reward, terminated, truncated, info = env.step(action)
# total_reward1 += reward
# print(f'Action: {action}, Reward: {reward}, Terminated: {terminated}')
# env.render()
# action = env.actions['pickup']
# state, reward, terminated, truncated, info = env.step(action)
# total_reward1 += reward
# print(f'Action: {action}, Reward: {reward}, Terminated: {terminated}')
# env.render()
# action = env.actions['move_to_2']
# state, reward, terminated, truncated, info = env.step(action)
# total_reward1 += reward
# print(f'Action: {action}, Reward: {reward}, Terminated: {terminated}')
# env.render()
# action = env.actions['dropoff']
# state, reward, terminated, truncated, info = env.step(action)
# total_reward1 += reward
# print(f'Action: {action}, Reward: {reward}, Terminated: {terminated}')
# env.render()
# action = env.actions['move_to_4']
# state, reward, terminated, truncated, info = env.step(action)
# total_reward1 += reward
# print(f'Action: {action}, Reward: {reward}, Terminated: {terminated}')
# env.render()
# action = env.actions['pickup']
# state, reward, terminated, truncated, info = env.step(action)
# total_reward1 += reward
# print(f'Action: {action}, Reward: {reward}, Terminated: {terminated}')
# env.render()
# action = env.actions['move_to_3']
# state, reward, terminated, truncated, info = env.step(action)
# total_reward1 += reward
# print(f'Action: {action}, Reward: {reward}, Terminated: {terminated}')
# env.render()
# action = env.actions['dropoff']
# state, reward, terminated, truncated, info = env.step(action)
# total_reward1 += reward
# print(f'Action: {action}, Reward: {reward}, Terminated: {terminated}')
# env.render()
# print(f'total_reward1: {total_reward1}')
# print("....................................................")
# print("....................................................")
# # env = RedistributionEnv()
# state = env.reset(40)
# print("Initial State:")
# env.render()  # 渲染初始状态
# print('.......................................')
# total_reward2 = 0
# action = env.actions['pickup']
# state, reward, terminated, truncated, info = env.step(action)
# total_reward2 += reward
# print(f'Action: {action}, Reward: {reward}, Terminated: {terminated}')
# env.render()
# action = env.actions['move_to_4']
# state, reward, terminated, truncated, info = env.step(action)
# total_reward2 += reward
# print(f'Action: {action}, Reward: {reward}, Terminated: {terminated}')
# env.render()
# action = env.actions['dropoff']
# state, reward, terminated, truncated, info = env.step(action)
# total_reward2 += reward
# print(f'Action: {action}, Reward: {reward}, Terminated: {terminated}')
# env.render()
# action = env.actions['move_to_3']
# state, reward, terminated, truncated, info = env.step(action)
# total_reward2 += reward
# print(f'Action: {action}, Reward: {reward}, Terminated: {terminated}')
# env.render()
# action = env.actions['pickup']
# state, reward, terminated, truncated, info = env.step(action)
# total_reward2 += reward
# print(f'Action: {action}, Reward: {reward}, Terminated: {terminated}')
# env.render()
# action = env.actions['move_to_2']
# state, reward, terminated, truncated, info = env.step(action)
# total_reward2 += reward
# print(f'Action: {action}, Reward: {reward}, Terminated: {terminated}')
# env.render()
# action = env.actions['dropoff']
# state, reward, terminated, truncated, info = env.step(action)
# total_reward2 += reward
# print(f'Action: {action}, Reward: {reward}, Terminated: {terminated}')
# env.render()
# action = env.actions['move_to_0']
# state, reward, terminated, truncated, info = env.step(action)
# total_reward2 += reward
# print(f'Action: {action}, Reward: {reward}, Terminated: {terminated}')
# env.render()
# action = env.actions['dropoff']
# state, reward, terminated, truncated, info = env.step(action)
# total_reward2 += reward
# print(f'Action: {action}, Reward: {reward}, Terminated: {terminated}')
# env.render()
# action = env.actions['move_to_4']
# state, reward, terminated, truncated, info = env.step(action)
# total_reward2 += reward
# print(f'Action: {action}, Reward: {reward}, Terminated: {terminated}')
# env.render()
# action = env.actions['dropoff']
# state, reward, terminated, truncated, info = env.step(action)
# total_reward2 += reward
# print(f'Action: {action}, Reward: {reward}, Terminated: {terminated}')
# env.render()
# print(f'total_reward2: {total_reward2}')


# print("....................................................")
# print("....................................................")
# print("Attempting movement...")
#
# # 遍历所有车站的所有可能移动
# for from_station in range(env.num_docks):
#     for to_station in range(env.num_docks):
#         if from_station != to_station:
#             # 重置到起始站点
#             env.state['truck_position'] = from_station
#             env.render()
#             print(f"Move from Station {from_station} to Station {to_station}")
#             # 执行移动动作
#             action = env.actions[f"move_to_{to_station}"]
#             _, reward, terminated, _, _ = env.step(action)
#             print(f"Performed move from {from_station} to {to_station}")
#             print(f"Reward: {reward}, Done: {terminated}")
#             env.render()  # 渲染每个动作之后的状态
#             print('*********************************')

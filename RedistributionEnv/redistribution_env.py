import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from typing import Tuple


class RedistributionEnv(gym.Env):
    metadata = {'render.modes': ['ansi']}

    def __init__(self, num_docks=5, dock_max_bikes=20, max_steps_per_episode=100, max_bikes_on_truck=5):
        self.num_docks = num_docks
        self.dock_max_bikes = dock_max_bikes
        self.max_steps_per_episode = max_steps_per_episode
        self.max_bikes_on_truck = max_bikes_on_truck
        self.action_space = spaces.Discrete(num_docks * (num_docks - 1) + 2)
        self.actions = {f"move_{i}_to_{j}": i * (num_docks - 1) + j - (1 if j > i else 0) for i in range(num_docks) for
                        j in range(num_docks) if i != j}
        self.actions["pickup"] = num_docks * (num_docks - 1)
        self.actions["dropoff"] = num_docks * (num_docks - 1) + 1
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
        self.reward_range = (-10, 60)
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
        self.balanced_bikes = total_bikes // self.num_docks
        self.timestep = 0
        return self.state, {}

    def step(self, action) -> Tuple[dict, float, bool, bool, dict]:
        assert self.action_space.contains(action), "Invalid action"
        reward = -1  # 每个时间步的固定负奖励
        terminated = False
        truncated = False
        info = {}

        if action < self.num_docks * (self.num_docks - 1):  # Movement
            from_station = action // (self.num_docks - 1)
            offset = action % (self.num_docks - 1)
            to_station = offset + (1 if offset >= from_station else 0)

            travel_cost = self.distance_matrix[from_station][to_station]

            # 检查两个车站的车辆数量是否都在平均数量的误差范围内
            if abs(self.state['bike_states'][from_station] - self.balanced_bikes) <= 1 and abs(
                    self.state['bike_states'][to_station] - self.balanced_bikes) <= 1:
                reward -= 2 + travel_cost  # 无效移动时增加惩罚
            else:
                reward -= travel_cost  # 增加移动的惩罚

            self.state['truck_position'] = to_station

        elif action == self.actions["pickup"]:  # Pickup bikes
            dock_index = self.state['truck_position']
            bikes_in_dock = self.state['bike_states'][dock_index]
            if bikes_in_dock > self.balanced_bikes and self.state['bikes_on_truck'] < self.max_bikes_on_truck:
                bikes_out = min(bikes_in_dock - int(np.ceil(self.balanced_bikes)),
                                self.max_bikes_on_truck - self.state['bikes_on_truck'])
                self.state['bike_states'][dock_index] -= bikes_out
                self.state['bikes_on_truck'] += bikes_out
                reward += 3  # fixed reward
                reward += 5 - bikes_out  # Encourage efficient redistribution
            else:
                reward -= 5

        elif action == self.actions["dropoff"]:  # Dropoff bikes
            dock_index = self.state['truck_position']
            bikes_in_dock = self.state['bike_states'][dock_index]
            if bikes_in_dock < self.balanced_bikes and self.state['bikes_on_truck'] > 0:
                bikes_needed = int(np.floor(self.balanced_bikes)) - bikes_in_dock
                bikes_out = min(bikes_needed, self.state['bikes_on_truck'])
                self.state['bike_states'][dock_index] += bikes_out
                self.state['bikes_on_truck'] -= bikes_out
                reward += 3  # fixed reward
                reward += 5 - bikes_out  # Encourage efficient redistribution
            else:
                reward -= 5

        # 经过测试这一步对获得最终奖励很重要。
        # Balance reward for individual stations. 每一步中检查所有站点状态并给予接近平衡状态的小额奖励。
        for bikes in self.state['bike_states']:
            if abs(bikes - self.balanced_bikes) <= 1:
                reward += 2  # Small reward for being close to balanced

        self.timestep += 1

        # Check for termination
        if np.all(np.abs(self.state['bike_states'] - self.balanced_bikes) <= 1) and self.state['bikes_on_truck'] == 0:
            terminated = True
            reward += 80  # Large reward for achieving overall balance
        elif self.timestep >= self.max_steps_per_episode:
            truncated = True
            info['reason'] = 'max_steps_reached'

        return self.state, reward, terminated, truncated, info

    def render(self, mode='ansi'):
        if mode != 'ansi':
            raise NotImplementedError("Render mode not supported: {}".format(mode))

        output = "\n"
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
#
# # 重置环境，开始新的episode
# state = env.reset(66)
#
# print("Initial State:")
# env.render()  # 渲染初始状态
# print('.......................................')
#
# # 尝试执行 Pickup 动作
# print("Attempting pickup...")
# pickup_action = env.actions['pickup']
# state, reward, terminated, _, _ = env.step(pickup_action)  # 执行 pickup 动作
# print(f"Reward after pickup: {reward}, Done: {terminated}")
# env.render()  # 渲染动作后的状态
#
# # 执行 Movement 动作 从0到3
# print("Attempting move_0_to_3...")
# move_0_to_3 = env.actions['move_0_to_3']
# state, reward, terminated, _, _ = env.step(move_0_to_3)
# print(f"Reward after move_0_to_3: {reward}, Done: {terminated}")
# env.render()
#
# # 尝试执行 Dropoff 动作
# print("Attempting dropoff...")
# dropoff_action = env.actions['dropoff']
# state, reward, terminated, _, _ = env.step(dropoff_action)  # 执行 dropoff 动作
# print(f"Reward after dropoff: {reward}, Done: {terminated}")
# env.render()  # 渲染动作后的状态
#
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
#             action = env.actions[f"move_{from_station}_to_{to_station}"]
#             _, reward, terminated, _, _ = env.step(action)
#             print(f"Performed move from {from_station} to {to_station}")
#             print(f"Reward: {reward}, Done: {terminated}")
#             env.render()  # 渲染每个动作之后的状态
#             print('*********************************')

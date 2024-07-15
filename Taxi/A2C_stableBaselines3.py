import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_reward = 0

    def _on_step(self) -> bool:
        # 获取当前步的奖励
        self.episode_reward += self.locals['rewards']

        # 检查回合是否结束
        if self.locals['dones']:
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0

        return True

    def get_rewards(self):
        return np.array(self.episode_rewards)

# 创建Taxi-v3环境，并包裹在Monitor中
env = gym.make('Taxi-v3')
env = Monitor(env)

# 创建A2C模型，使用MlpPolicy策略
model = A2C('MlpPolicy', env, verbose=1)

# 记录训练过程中的奖励
reward_callback = RewardCallback()

# 训练模型
model.learn(total_timesteps=20000, callback=reward_callback)

# 获取奖励数据
episode_rewards = reward_callback.get_rewards()

# 直接绘制奖励数据
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('A2C Training Rewards')
plt.text(0, min(episode_rewards), f'Total training time: {model.num_timesteps / 1000:.2f}k timesteps', fontsize=12)
plt.savefig("a2c_taxi_training_rewards.png")
plt.show()

# 评估模型
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f'Mean reward: {mean_reward} +/- {std_reward}')

# 保存模型
model.save("a2c_taxi_v3")

# # 测试训练好的模型
# obs, info = env.reset()  # 注意: 这里需要解包元组
# for i in range(100):
#     action = model.predict(obs, deterministic=True)[0].item()  # 确保动作是整数
#     obs, reward, terminated, truncated, info = env.step(action)
#     env.render()
#     if terminated or truncated:
#         obs, info = env.reset()  # 再次解包元组

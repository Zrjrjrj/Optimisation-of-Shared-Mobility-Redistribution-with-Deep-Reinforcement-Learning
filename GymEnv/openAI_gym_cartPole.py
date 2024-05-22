import gym

# Create CartPole environment and specify render mode
env = gym.make("CartPole-v1", render_mode='human')

# Initialize the environment with a seed for reproducibility
for episode in range(5):
    observation = env.reset(seed = 66)
    for t in range(50):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info, truncated = env.step(action)
        if done or truncated:
            if truncated:
                print(f"Episode {episode + 1} finished due to truncation after {t + 1} timesteps")
            else:
                print(f"Episode {episode + 1} finished after {t + 1} timesteps")
            break

env.close()

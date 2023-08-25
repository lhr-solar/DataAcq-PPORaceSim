from fixed_gym import *
from stable_baselines3 import PPO

gym.register(
    id="SolarCar-v0",
    entry_point="solar_car_env:SolarCarEnv",
    max_episode_steps=300,
)

env = gym.make("SolarCar-v0", render_mode="human", time_step_duration=1)

model = PPO.load("ppo_car_racing", env=env)

obs, info = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

from fixed_gym import gym

from stable_baselines3 import PPO

gym.register(
    id="SolarCar-v0",
    entry_point="solar_car_2:SolarCar",
    max_episode_steps=300,
)

env = gym.make("SolarCar-v0", render_mode="human", timeStepDuration=1/60)

model = PPO.load("ppo_car_racing", env=env)

obs, info = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

from fixed_gym import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.env_checker import check_env

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

gym.register(
    id="SolarCar-v0",
    entry_point="solar_car_2:SolarCar",
    max_episode_steps=2500,
)

# check_env("SolarCar-v0")

env = make_vec_env("SolarCar-v0", n_envs=1,
                   env_kwargs=dict(render_mode="computer", timeStepDuration=5))

model = PPO.load("ppo_car_racing", env=env, verbose=1,
                 tensorboard_log='log', device="cuda")

model.learn(total_timesteps=100000*100, progress_bar=True)

model.save("ppo_car_racing")

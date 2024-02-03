import argparse

import numpy as np
from fixed_gym import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser(prog="DataAcq-PPORaceSim")
parser.add_argument("-r", "--human", action="store_true")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-n", "--new", action="store_true", default=False)
parser.add_argument("-c", "--continue_training", action="store_true", default=True)
parser.add_argument("-f", "--file", type=str, default="ppo_car_racing")
parser.add_argument("-e", "--episode_length", type=int, default=28800)
parser.add_argument("-d", "--device", type=str, default="cuda")
parser.add_argument("-p", "--play", action="store_true", default=False)

args = parser.parse_args()

if __name__ == "__main__":
  EPISODE_LENGTH = args.episode_length

  gym.register(
    id="SolarCar-v0",
    entry_point="solar_car_env:SolarCarEnv",
    max_episode_steps=EPISODE_LENGTH,
  )

  render_mode = "human" if args.human else "computer"
  env = make_vec_env("SolarCar-v0", n_envs=1,
                    env_kwargs=dict(render_mode=render_mode), seed=np.random.randint(0, 2 ** 31))

  device = args.device
  verbose = 1 if args.verbose else 0
  path = args.file
  model = None
  if args.play:
    model = PPO.load("ppo_car_racing", env=env)

    obs, info = env.reset()
    while True:
      action, _states = model.predict(obs)
      obs, reward, terminated, truncated, info = env.step(action)
      env.render()
  else:
    if args.continue_training and not args.new:
      print("Continuing training from", path)
      model = PPO.load(path, env=env, verbose=verbose,
                  tensorboard_log='log', device=device)
    else:
      print("Starting new training", path)
      model = PPO("MlpPolicy", env, verbose=verbose, tensorboard_log='log', device=device)

    model.learn(total_timesteps=EPISODE_LENGTH, progress_bar=True)

    model.save(path)




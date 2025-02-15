import argparse

import numpy as np
from fixed_gym import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from solar_car_v2.solar_car import SolarCar

import os

import cProfile

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(prog="DataAcq-PPORaceSim")
parser.add_argument("-r", "--human", action="store_true")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-n", "--new", action="store_true", default=False)
parser.add_argument("-c", "--continue_training", action="store_true", default=True)
parser.add_argument("-f", "--file", type=str, default="ppo_car_racing_2")
parser.add_argument("-e", "--episode_length", type=int, default=28800)
parser.add_argument("-d", "--device", type=str, default="cuda")
parser.add_argument("-p", "--play", action="store_true", default=False)
parser.add_argument("-env", "--env_count", type=int, default=1)
parser.add_argument("-b", "--batch_size", type=int, default=64)
parser.add_argument("-ec", "--episode_count", type=int, default=1)

args = parser.parse_args()
episode_length = args.episode_length

gym.register(
    id="SolarCar-v0",
    entry_point=SolarCar,
    max_episode_steps=episode_length,
)

if __name__ == "__main__":
    n_envs = args.env_count if not args.play else 1
    render_mode = "human" if args.human or args.play else "computer"

    env = make_vec_env(
        lambda render_mode: SolarCar(render_mode),
        n_envs=n_envs,
        env_kwargs=dict(render_mode=render_mode),
        seed=np.random.randint(0, 2**31),
        vec_env_cls=SubprocVecEnv,
    )

    device = args.device
    verbose = 1 if args.verbose else 0
    path = args.file
    episode_count = args.episode_count
    model = None

    n_steps = episode_count * episode_length

    if args.play:
        model = PPO.load("ppo_car_racing", env=env)

        obs, _ = env.reset()

        while True:
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated = env.step(action)
            env.render()
    else:
        if args.continue_training and not args.new:
            print("Continuing training from", path)
            model = PPO.load(
                path,
                env=env,
                verbose=verbose,
                tensorboard_log="log",
                device=device,
                n_steps=n_steps,
            )
        else:
            print("Starting new training", path)
            model = PPO(
                "MlpPolicy",
                env,
                verbose=verbose,
                tensorboard_log="log",
                device=device,
                n_steps=n_steps,
            )

        model.learn(
            total_timesteps=episode_count * episode_length * n_envs, progress_bar=True
        )

        model.save(path)

        env.close()

from fixed_gym import *
import numpy as np
from typing import Optional
import datetime
import matplotlib.pyplot as plt
from helper import plot_spline_3d

import pygame

from solar_car import Track
from solar_car import Car


class SolarCarEnv(gym.Env):
    metadata = {"render_modes": [
        "human", "computer"], "render_fps": 60, "time_step_duration": 1}

    def generate_track(self):
        # geo_json = {
        #     "type": "FeatureCollection",
        #     "features": [{
        #         "type": "Feature",
        #         "geometry": {
        #             "type": "Point",
        #             "coordinates": [random.randrange(0, self.window_size), random.randrange(0, self.window_size)]
        #         },
        #         "properties": {
        #             "elevation": random.uniform(0, 0.05)
        #         }
        #     } for _ in range(10)]
        # }
        #self.track = Track(geo_json=geo_json)
        self.track = Track(track_file= "elevation_data\elevation_data")

    def __init__(self, render_mode, time_step_duration=10):
        self.render_mode = render_mode
        self.time_step_duration = time_step_duration

        # shape is [distance along track, velocity, slope, battery %, solar radiation, future solar radiation]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,),  dtype=np.float32)

        # shape is [acceleration]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

        self.prev_reward = 0
        self.prev_distance = 0
        self.prev_soc = 0
        self.distance = 0
        self.window = None
        self.window_size = 512  # The size of the PyGame window
        self.clock = None
        self.size = 5

        self.start_time = datetime.datetime(2023, 1, 10, 9, 0, 0).timestamp()
        self.time = self.start_time

        self.generate_track()
        self.car = Car(self.track, self.time, self.time_step_duration)

        self._agent_location = 0
        self._target_location = self.track.track_length
        self.reward = 0

        self.velocity = 0
        self.acc = 0

        self.solar_radiation = 0
        self.soc = 0
        self.current = 0
        if render_mode == "human":
            self.fig = plt.figure(figsize=(10,6), layout = 'constrained')
            self.ax = self.fig.add_subplot(3, 3, (1, 6), projection='3d')
            coords = self.car.track.evaluate_cs(self.car.dist)
            self.currentPoint = self.ax.scatter(*coords, marker='*', color='red')
            self.time_array = [self.time]
            # Distance graph
            self.distAx = self.fig.add_subplot(3, 3, (7))
            self.dist_array = [self.distance]
            self.distAx.plot(self.time_array, self.dist_array)
            self.distAx.set_title('Distance')
            # Velocity Graph
            self.velAx = self.fig.add_subplot(3, 3, (8))
            self.vel_array = [self.velocity]
            self.velAx.plot(self.time_array, self.vel_array)
            self.velAx.set_title('Velocity')
            # Battery Graph
            self.batteryAx = self.fig.add_subplot(3, 3, (9))
            self.battery_array = [self.soc]
            self.batteryAx.plot(self.time_array, self.battery_array)
            self.batteryAx.set_title('Battery')

    
            plot_spline_3d(self.track.cmr, ax=self.ax)
            plt.ion()
            plt.show()

    def _get_obs(self):
        self.soc = self.car.battery.get_soc()
        self.solar_radiation = self.car.weather.get_intensity(self.car.time)
        self.velocity = self.car.motors.current_speed
        self.current = self.car.motors.current_current
        self.distance = self.car.dist
        self.current = self.car.battery.current_draw

        obs = np.array([
            self.distance,
            self.velocity,
            self.track.elevation_slope(
                self.track.distance_to_t(self.distance))/180 * np.pi,
            self.soc,
            self.solar_radiation
        ])

        return obs

    def _get_info(self):
        return {
            "distance": self.distance,
        }

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict] = None,):
        super().reset(seed=seed)

        self._agent_location = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        self.reward = 0
        self.prev_reward = 0
        self.prev_distance = 0
        self.prev_soc = 0
        self.distance = 0
        self.velocity = 0
        self.acc = 0
        self.time = self.start_time
        self.current = 0
        self.generate_track()
        #if 3d restart the plot
        if self.render_mode == "human":
            #Update graph data
            self.battery_array = [self.soc]
            self.time_array = [self.time]
            self.dist_array = [self.distance]
            self.vel_array = [self.velocity]

        self.car.reset(self.track)

        return observation, info

    def step(self, action):
        gas = action[0]
        gas = np.clip(gas, -1, 1)
        self.acc = gas

        step_reward = 0
        terminated = False
        truncated = False

        if action is not None:
            ok, dist = self.car.step(gas)

            if not ok:
                terminated = True
                self.reward -= 10000

            # this makes moving at about 3km/s outpace (????????? - Sohan)
            # self.reward += dist
            # step_reward = self.reward - self.prev_reward
            # self.prev_reward = self.reward
            self.reward = np.abs((self.distance - self.prev_distance)/(self.soc - self.prev_soc))
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            self.prev_distance = self.distance
            self.prev_soc = self.soc


        self.time += self.time_step_duration
        observation = self._get_obs()
        info = self._get_info()

        self.render()

        return observation, step_reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            return self._render_frame(self)

    def _render_frame(self):
        coords = self.car.track.evaluate_cs(self.car.dist)
        self.currentPoint.remove()
        self.currentPoint = self.ax.scatter(*coords, marker='*', color='red')
        self.time_array.append(self.time)
        self.dist_array.append(self.distance)
        self.vel_array.append(self.velocity)
        self.battery_array.append(self.soc)
        self.distAx.cla()
        self.velAx.cla()
        self.batteryAx.cla()
        self.distAx.plot(self.time_array, self.dist_array)
        self.velAx.plot(self.time_array, self.vel_array)
        self.batteryAx.plot(self.time_array, self.battery_array)

        self.ax.legend()
        plt.pause(0.01)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

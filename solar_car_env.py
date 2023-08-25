from fixed_gym import *
import numpy as np
from typing import Optional
import datetime
import random

import pygame

from solar_car import Track
from solar_car import Car


class SolarCarEnv(gym.Env):
    metadata = {"render_modes": [
        "human", "computer"], "render_fps": 60, "time_step_duration": 1/60}

    def generate_track(self):
        geo_json = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [random.randrange(0, self.window_size), random.randrange(0, self.window_size)]
                },
                "properties": {
                    "elevation": random.randrange(0, 15)
                }
            } for _ in range(10)]
        }

        self.track = Track(geo_json=geo_json)

    def __init__(self, render_mode="human", time_step_duration=10):
        self.render_mode = render_mode
        self.time_step_duration = time_step_duration

        # shape is [distance along track, velocity, slope, battery %, solar radiation, future solar radiation]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,),  dtype=np.float32)

        # shape is [acceleration]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

        self.prev_reward = 0
        self.prev_distance = 0
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
            self._render_frame()

        self.reward = 0
        self.prev_reward = 0
        self.prev_distance = 0
        self.distance = 0
        self.velocity = 0
        self.acc = 0
        self.time = self.start_time
        self.current = 0
        self.generate_track()

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

            # this makes moving at about 3km/s outpace
            self.reward += dist
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward

        self.time += self.time_step_duration
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, step_reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        details_surface_size = 200
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size + details_surface_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        padding = 10
        map_surface_size = self.window_size - padding * 2
        map_canvas = pygame.Surface(
            (map_surface_size, map_surface_size))
        map_canvas.fill((255, 255, 255))
        inner_padding = 25

        width = self.track.bounding_box[1][0] - self.track.bounding_box[0][0]
        height = self.track.bounding_box[1][1] - self.track.bounding_box[0][1]
        map_size = np.array([width, height]) * \
            map_surface_size + inner_padding * 2
        map = pygame.Surface(map_size)
        map.fill((255, 255, 255))

        # Draw the track
        for i in np.linspace(0, self.track.t_len, 100, endpoint=False):
            p = self.track.evaluate_cs(i)[:2]
            pygame.draw.circle(
                map,
                (100, 100, 100),
                p * map_surface_size + inner_padding,
                2,
            )

        # First we draw the waypoints
        for waypoint in self.track.points:
            pygame.draw.circle(
                map,
                (0, 0, 255),
                waypoint[:2] * map_surface_size + inner_padding,
                3,
            )

        # Now we draw the agent
        pygame.draw.circle(
            map,
            (255, 0, 0),
            (self.track.evaluate_cs(self.distance)[
             :2] * map_surface_size + inner_padding),
            4,
        )

        map_canvas.blit(map, map_surface_size / 2 - map_size / 2)
        canvas.blit(map_canvas, (padding, padding))

        details_surface = pygame.Surface(
            (self.window_size - padding * 2, details_surface_size))
        details_surface.fill((0, 0, 0))

        # Text for reward
        font = pygame.font.SysFont("Arial", 20)
        text = font.render('Reward: {:.2f}'.format(self.reward),
                           True, (255, 255, 255))
        details_surface.blit(text, (0, 0))

        text = font.render('Distance: {:.2f}'.format(self.distance),
                           True, (255, 255, 255))
        details_surface.blit(text, (0, 20))

        text = font.render('Velocity: {:.2f}'.format(self.velocity),
                           True, (255, 255, 255))
        details_surface.blit(text, (0, 40))

        text = font.render('Battery: {:.2f}'.format(self.soc),
                           True, (255, 255, 255))
        details_surface.blit(text, (0, 60))

        text = font.render("Solar Radiation: {:.1f}".format(self.solar_radiation),
                           True, (255, 255, 255))

        details_surface.blit(text, (0, 80))

        text = font.render('Time: {:.2f}'.format(self.time - self.start_time),
                           True, (255, 255, 255))
        details_surface.blit(text, (0, 100))

        text = font.render('Pedal: {:.2f}'.format(self.acc),
                           True, (255, 255, 255))
        details_surface.blit(text, (0, 120))

        text = font.render('Track Length: {:.2f}'.format(self.track.track_length),
                           True, (255, 255, 255))
        details_surface.blit(text, (0, 140))

        text = font.render('Laps Completed: {:.2f}'.format(self.car.dist / self.track.track_length),
                           True, (255, 255, 255))
        details_surface.blit(text, (0, 160))

        text = font.render('Current Draw: {:.2f}'.format(self.current),
                           True, (255, 255, 255))
        details_surface.blit(text, (225, 0))

        self.window.blit(details_surface, (padding, self.window_size))

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

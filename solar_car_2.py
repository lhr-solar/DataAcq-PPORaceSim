from fixed_gym import *
import numpy as np
from typing import Optional, Union
import datetime
import random

import pygame

from ka_chow.track import Track
from ka_chow.car import Car


class SolarCar(gym.Env):
    metadata = {"render_modes": [
        "human", "computer"], "render_fps": 60, "timeStepDuration": 1/60}

    def generateTrack(self):
        geojson = {
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

        self.track = Track(geojson=geojson)

    def __init__(self, render_mode="human", timeStepDuration=5):
        self.render_mode = render_mode
        self.timeStepDuration = timeStepDuration

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

        self.startTime = datetime.datetime(2023, 1, 10, 9, 0, 0).timestamp()
        self.time = self.startTime

        self.generateTrack()
        self.car = Car(self.track, self.time)

        self._agent_location = 0
        self._target_location = self.track.trackLength
        self.reward = 0

        self.velocity = 0
        self.acc = 0
        self.maxAcc = 50
        self.mass = 1000

        self.solarRadiation = 0
        self.battery = 0

    def _get_obs(self):
        obs = np.array([
            self.distance,
            self.velocity,
            self.track.elevationSlope(
                self.track.distanceToT(np.clip(self.distance, 0, self.track.trackLength)))/180 * np.pi,
            self.car.battery.get_capacity() / self.car.battery.max_capacity,
            self.car.array.calculate_irradiance()
        ])

        self.battery = self.car.battery.get_capacity() / self.car.battery.max_capacity
        self.solarRadiation = self.car.array.calculate_irradiance()

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
        self.time = self.startTime
        self.generateTrack()

        self.car.battery.current_capacity = self.car.battery.max_capacity

        return observation, info

    def step(self, action):
        acc = action[0]
        acc = np.clip(acc, -1, 1)
        self.acc = acc

        step_reward = 0
        terminated = False
        truncated = False

        # mass of car
        m = self.mass
        maxSpeed = self.track.maxSpeed(self.distance,  m)

        # acceleration from friction
        if self.velocity != 0:
            accF = -1/50 * self.velocity
        else:
            accF = 0

        if self.battery <= .05:
            self.acc = 0

        self.velocity = np.clip(
            (self.velocity + (self.acc * self.maxAcc + accF) * (self.timeStepDuration)), -maxSpeed, maxSpeed)

        self.time += self.timeStepDuration  # FPS
        slope = self.track.elevationSlope(
            self.track.distanceToT(np.clip(self.distance, 0, self.track.trackLength)))/180 * np.pi

        dist = self.car.drive(
            self.velocity, slope, self.time) / 1000
        self.distance += dist
        self.distance = np.clip(self.distance, 0, self.track.trackLength)

        if action is not None:
            # self.reward -= 1
            # this makes moving at about 3km/s outpace
            self.reward += (dist * 150)
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward

            if self.distance >= self.track.trackLength:
                # truncated = True
                self.distance = 0

            # if self.distance == 0:
            #     step_reward = -10

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, step_reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        detailsSurfaceSize = 200
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size + detailsSurfaceSize)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        padding = 10
        mapSurfaceSize = self.window_size - padding * 2
        mapCanvas = pygame.Surface(
            (mapSurfaceSize, mapSurfaceSize))
        mapCanvas.fill((255, 255, 255))
        innerPadding = 25

        width = self.track.boundingBox[1][0] - self.track.boundingBox[0][0]
        height = self.track.boundingBox[1][1] - self.track.boundingBox[0][1]
        mapSize = np.array([width, height]) * mapSurfaceSize + innerPadding * 2
        map = pygame.Surface(mapSize)
        map.fill((255, 255, 255))

        # Draw the track
        for i in np.linspace(0, self.track.tLen, 100, endpoint=False):
            p = self.track.evaluateCS(i)[:2]
            pygame.draw.circle(
                map,
                (100, 100, 100),
                p * mapSurfaceSize + innerPadding,
                2,
            )

        # First we draw the waypoints
        for waypoint in self.track.points:
            pygame.draw.circle(
                map,
                (0, 0, 255),
                waypoint[:2] * mapSurfaceSize + innerPadding,
                3,
            )

        # Now we draw the agent
        pygame.draw.circle(
            map,
            (255, 0, 0),
            (self.track.cmr.evaluate(
                self.track.distanceToT(self.distance))[:2] * mapSurfaceSize + innerPadding),
            4,
        )

        mapCanvas.blit(map, mapSurfaceSize / 2 - mapSize / 2)
        canvas.blit(mapCanvas, (padding, padding))

        detailsSurface = pygame.Surface(
            (detailsSurfaceSize, detailsSurfaceSize))
        detailsSurface.fill((0, 0, 0))

        # Text for reward
        font = pygame.font.SysFont("Arial", 20)
        text = font.render('Reward: {:.2f}'.format(self.reward),
                           True, (255, 255, 255))
        detailsSurface.blit(text, (0, 0))

        text = font.render('Distance: {:.2f}'.format(self.distance),
                           True, (255, 255, 255))
        detailsSurface.blit(text, (0, 20))

        text = font.render('Velocity: {:.2f}'.format(self.velocity),
                           True, (255, 255, 255))
        detailsSurface.blit(text, (0, 40))

        text = font.render('Battery: {:.2f}'.format(self.battery),
                           True, (255, 255, 255))
        detailsSurface.blit(text, (0, 60))

        text = font.render("Solar Radiation: {:.1f}".format(self.solarRadiation),
                           True, (255, 255, 255))

        detailsSurface.blit(text, (0, 80))

        text = font.render('Time: {:.2f}'.format(self.time - self.startTime),
                           True, (255, 255, 255))
        detailsSurface.blit(text, (0, 100))

        text = font.render('Pedal: {:.2f}'.format(self.acc),
                           True, (255, 255, 255))
        detailsSurface.blit(text, (0, 120))

        self.window.blit(detailsSurface, (padding, self.window_size))

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

from typing import Tuple
import numpy as np
from solar_car.battery import Battery
from solar_car.motors import Motors
from solar_car.solar_array import SolarArray
from solar_car.track import Track
from solar_car.weather import Weather
import time
import logging


class Car:
    """
    The car class is the main class of the solar car. It contains the battery, dynamics, and solar array.

    Parameters
    ----------
    time_step : float, optional
        The time step length of the car in seconds. The default is 1.0.
    """

    def __init__(self, track: Track, start_time: int, time_step: float = 1.0, coords: tuple = (40.7128, -74.0060)):
        self.coords = coords
        self.start_time = start_time
        self.time = self.start_time
        self.time_step = time_step

        self.weather = Weather(*coords, start_time)
        self.solar_array = SolarArray()
        self.battery = Battery(time_step)
        self.motors = Motors(time_step)
        self.track = track
        self.dist = 0
        self.t = 0

        self.okay = True

    def step(self, gas: float) -> Tuple[bool, float]:
        """
        Steps the car forward by one time step.

        Parameters
        ----------
        gas : float
            The amount of gas to apply to the car. This is a value between -1 and 1.

        Returns
        -------
        bool
            Whether or not the car is okay. The car will not step if it is not okay.
        float
            The distance traveled in this time step. Only valid if okay.

        """
        if not self.okay:
            logging.fatal(
                'Car cannot step because it has suffered a critical failure')
            return False, 0

        start = time.time()

        physical_max_speed = self.track.max_speed(self.t)

        voltage = self.battery.get_voltage()
        motors_current_draw, distance_traveled = self.motors.step(
            gas, voltage, physical_max_speed)

        self.dist += distance_traveled / 1000  # convert to km
        self.t = self.track.distance_to_t(self.dist)  # convert to km

        irradiance = self.weather.get_intensity(self.time)
        # solar_power = self.solar_array.step(irradiance)
        solar_power = self.solar_array.step() # time step is 1 second, so energy = power in this case
        solar_current_draw = solar_power / voltage

        total_draw = motors_current_draw - solar_current_draw
        self.battery.set_draw(total_draw)
        batter_okay = self.battery.step()

        if not batter_okay:
            logging.fatal('Battery is dead!')
            self.okay = False
            return self.okay, 0

        self.time += self.time_step
        end = time.time()
        logging.debug(f'Car step complete in {end - start} seconds')
        logging.debug(f"Time: {self.time}")
        logging.debug(
            f'Battery Voltage: {voltage}V')
        logging.debug(
            f'Battery Cell Voltage: {self.battery.get_cell_voltage()}V')
        logging.debug(
            f'Battery SoC: {self.battery.get_soc() * 100}%')
        logging.debug(f'Motors Current Draw: {motors_current_draw}A')
        logging.debug(f'Solar Irradiance: {irradiance}W/m^2')
        logging.debug(f'Solar Array Power: {solar_power}W')
        logging.debug(f'Solar Array Current: {-solar_current_draw}A')
        logging.debug(f'Total Current Draw: {total_draw}A')
        logging.debug(f'Gas: {gas}')
        logging.debug(f'Speed: {self.motors.current_speed} m/s')
        logging.debug(f'Total Distance Traveled: {self.dist}m')
        logging.debug(f'Distance Traveled This Step: {distance_traveled}m')
        logging.debug(f'Track length: {self.track.track_length}km')
        logging.debug(f'Physical Speed Limit: {physical_max_speed}m/s')
        logging.debug(f'Ground Level: {self.track.ground_level}m')
        logging.debug(
            f'Laps Completed: {int(self.dist / self.track.track_length)}')
        logging.debug(
            f'Track Completion: {np.fmod(self.dist / 1000 / self.track.track_length, 1) * 100}%')

        return self.okay, distance_traveled

    def is_okay(self) -> bool:
        return self.okay

    def reset(self, track: Track = None):
        self.__init__(track if track else self.track,
                      self.start_time, self.time_step, self.coords)


if __name__ == "__main__":
    import datetime

    logging.basicConfig(level=logging.DEBUG)

    start_time = int(datetime.datetime(2023, 1, 10, 9, 0, 0).timestamp())

    track = Track(track_file='track2.json')
    c = Car(track, start_time, 10)
    for i in range(1000):
        c.step(.2)
        time.sleep(1)

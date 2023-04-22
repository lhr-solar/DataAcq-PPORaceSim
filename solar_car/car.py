from battery import Battery
from motors import Motors
from solar_array import SolarArray
from weather import Weather
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

    def __init__(self, start_time: int, time_step: float = 1.0, coords: tuple = (40.7128, -74.0060)):
        self.time = start_time
        self.time_step = time_step

        self.weather = Weather(*coords, start_time)
        self.solar_array = SolarArray()
        self.battery = Battery(time_step)
        self.motors = Motors(time_step)

        self.okay = True

    def step(self, gas: float) -> bool:
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

        """
        if not self.okay:
            logging.fatal('Car is dead!')
            return False

        start = time.time()
        voltage = self.battery.get_voltage()
        motors_current_draw = self.motors.step(gas, voltage)

        irradiance = self.weather.get_intensity(self.time)
        solar_power = self.solar_array.step(irradiance)
        solar_current_draw = solar_power / voltage

        total_draw = motors_current_draw - solar_current_draw
        self.battery.set_draw(total_draw)
        batter_okay = self.battery.step()

        if not batter_okay:
            logging.fatal('Battery is dead!')
            self.okay = False
            return False

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

        return self.okay

    def is_okay(self) -> bool:
        return self.okay


if __name__ == "__main__":
    import datetime

    logging.basicConfig(level=logging.DEBUG)

    start_time = int(datetime.datetime(2023, 1, 10, 9, 0, 0).timestamp())

    c = Car(start_time, 10)
    for i in range(1000):
        c.step(.2)

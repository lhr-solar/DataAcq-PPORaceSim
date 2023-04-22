import ka_chow.battery as battery
import ka_chow.solar_array as solar_array
import ka_chow.motor as motor

import time
import datetime

import numpy as np

from ka_chow.track import Track


class Car:

    ELECTRONICS_POWER = 77  # W
    VOLTAGE_BATTERY = 96  # V
    MAX_BATTERY_CAPACITY_WH = 5400

    def __init__(self, track: Track, time: int) -> None:

        self.lon, self.lat = track.coords
        self.temperature = 25  # update
        self.time = time

        self.battery = battery.Battery(self.MAX_BATTERY_CAPACITY_WH)
        self.array = solar_array.Array(self, self.temperature)
        self.motor = motor.Motor()

    def drive(self, speed, slope, time):
        # print("Car is driving")
        deltaTime = time - self.time

        arrayPower = self.array.get_power()

        self.motor.update_parameters(speed, slope, time)
        motorPower = max(0, self.motor.get_current() * self.VOLTAGE_BATTERY)

        used_power = (arrayPower - self.ELECTRONICS_POWER -
                      motorPower)*deltaTime/3600  # Wh

        self.battery.update(used_power)
        # print("Motor Power: " + str(motorPower) + " W")
        # print("Battery: " + str(self.battery.get_capacity()) + " Wh")
        self.time = time

        return speed * deltaTime * np.cos(slope)


if __name__ == "__main__":
    t = Track("track2.json")
    car = Car(t, datetime.datetime(2023, 1, 10, 9, 0, 0).timestamp())
    i = 0
    while True:
        car.drive(10, 0, i)
        i += 1
        time.sleep(1)

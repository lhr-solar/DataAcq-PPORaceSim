import numpy as np

# Mostly inspired from https://electronics.stackexchange.com/questions/270971/what-determines-how-much-power-the-motor-draws
# These are all generalizations of the concepts explained in this post and are not meant to be super accurate.


class Motors:
    max_voltage = 300  # Volts
    max_current = 580  # Amps

    current_voltage = 0  # Volts
    current_current = 0  # Amps

    braking_resistance = 0.1  # Ohms
    friction_resistance = 0.1  # Ohms

    motor_acceleration = 12  # m/s^2
    current_speed = 0  # m/s
    max_speed = 20  # m/s

    def __init__(self, time_step: float):
        self.time_step = time_step

    def step(self, gas: float, voltage: float) -> float:
        """
        Steps the simulation forward by one time step.

        Parameters
        ----------
        gas : float
            The amount of gas being applied to the car in range of [-1, 1].

        voltage : float
            The voltage of the battery.

        Returns
        -------
        float
            The current being drawn from the motors.

        """

        self.current_voltage = np.clip(voltage, 0, self.max_voltage)
        max_speed = (self.current_voltage / self.max_voltage) * self.max_speed

        if self.current_speed == 0:
            acc = self.motor_acceleration * gas
        else:
            acc = np.clip(10 / self.current_speed * gas, -
                          self.motor_acceleration, self.motor_acceleration)
        self.current_current = np.abs(
            acc) / self.motor_acceleration * self.max_current
        self.current_speed = np.clip(
            self.current_speed + acc * self.time_step, 0, max_speed)

        if gas == 0:
            self.current_speed = np.clip(
                self.current_speed - (self.friction_resistance * self.time_step * np.sign(self.current_speed)), 0, max_speed)

        return self.current_current

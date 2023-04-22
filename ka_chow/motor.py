import ka_chow.dynamics as dynamics


class Motor:
    def __init__(self, speed=0, slope=0):  # pedal Incline in percent
        #self.pedalIncline = pedalIncline
        #self.pedalIncline = self.current / 50
        self.slope = slope
        self.speed = speed
        self.torque = 0
        self.current = 0

        wheel_radius = 0.285496  # m
        mass = 317.515  # kg
        rolling_friction = .004
        static_friction = .7
        drag_coefficient = .2
        frontal_area = 1  # m^2
        self.dynamics = dynamics.Dynamics(
            mass, rolling_friction, static_friction, drag_coefficient, self.speed, frontal_area, wheel_radius)

    def update_parameters(self, speed, slope, time):
        self.slope = slope
        self.speed = speed
        self.dynamics.update_params(time, speed)

    # current provided by motor calculated by force from Dynamics
    def get_current(self):
        # parameters need to be verified
        self.torque = self.dynamics.total_torque(self.slope)
        # equation from torque-current curve on motor data sheet
        self.current = (29/34)*self.torque + 1
        return self.current

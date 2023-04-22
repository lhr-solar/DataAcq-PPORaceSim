# model of the dynamics system based on the solar vehicle simulation paper
# TODO: implement the other equations from the paper

"""
This class is meant to model the dynamics systems and energy losses
The system will take in paramaters such as the car's mass, the velocity it's going at,
& the friction coefficient of the car as preliminary constants. 
It will then calculate the required vehicle propelling force.
"""

import math

class Dynamics:

    GRAVITY = 9.81; #constant for gravity

    AIR_MASS_DENSITY = 1.225; #kg/m^3

    # parameters for the constructor: mass, vehicle tire friction coefficient, aerodynamic drag coefficient
    # current velocity, and frontal area
    def __init__(self, mass, tire_frictk, tire_fricts, drag_coeff, velocity, frontal_area, tire_radius):
        self.mass = mass
        self.tire_frictk = tire_frictk
        self.tire_fricts = tire_fricts
        self.drag_coeff = drag_coeff
        self.velocity = velocity
        self.old_velocity = velocity
        self.frontal_area = frontal_area
        self.tire_radius = tire_radius
        self.time = 0

    # first variable of the equation: the rolling resistance force
    def rolling_resistance(self):
        return self.mass * self.GRAVITY * self.tire_frictk

    # second variable of the equation: the aerodynamic drag force
    def aerodynamic_drag(self):
        return 0.5 * self.drag_coeff * self.AIR_MASS_DENSITY * self.velocity * self.velocity * self.frontal_area

    # third variable of the equation: the climbing resistance force
    # theta will come as an angle from track data
    def climbing_resistance(self, theta):
        return self.mass * self.GRAVITY * math.sin(theta)

    # fourth variable of the equation: the linear acceleration force
    # acceleration is only taken into account if the vehicle's velocity has changed
    def linear_acceleration(self):
        # if old velocity is equal to new velocity, then return 0
        if self.old_velocity == self.velocity:
            return 0
        # if old velocity is not equal to new velocity, then calculate the linear acceleration force, avoid division by 0
        else:
            if self.time == 0:
                return 0
            else:
                return self.mass * (self.velocity - self.old_velocity) / self.time

    #this method calculates the max velocity that we can go at on a turn
    def max_velocity(self, radius):
        if radius == 0:
            return 10
        return math.sqrt(self.GRAVITY * self.mass * self.tire_fricts * radius)

    # this method calculates the total required vehicle propelling force
    def total_propelling_force(self, slope):
        print("Rolling Resistance: {0:.2f} N".format(self.rolling_resistance()))
        print("Aerodynamic Drag: {0:.2f} N".format(self.aerodynamic_drag()))
        print("Climbing Resistance: {0:.2f} N".format(self.climbing_resistance(slope)))
        print("Linear Acceleration: {0:.2f} N".format(self.linear_acceleration()))

        return self.rolling_resistance() + self.aerodynamic_drag() + self.climbing_resistance(slope) + self.linear_acceleration()

    def updateParams(self, time, velocity):
        self.time = time
        self.old_velocity = self.velocity
        self.velocity = velocity

    def get_velocity(self): 
        return self.velocity
    
    def total_torque(self, slope):
        return self.total_propelling_force(slope) * self.tire_radius

# test code
if __name__ == "__main__":
    # create a new instance of the newDynamics class
    newDynamics = newDynamics(100, 0.01, 0.5, 10, 1, 0.285496) # mass = 100 kg, tire friction coefficient = 0.01, drag coefficient = 0.5, velocity = 10 m/s, frontal area = 1 m^2, radius = 0.285496
    assert newDynamics.rolling_resistance() == 9.81 # rolling resistance force = 9.81 N
    assert newDynamics.aerodynamic_drag() == 0.5 * 0.5 * 1.225 * 10 * 10 * 1
    assert newDynamics.climbing_resistance(0) == 0 # climbing resistance force = 0 N
    assert newDynamics.linear_acceleration() == 0 # linear acceleration force = 0 N
    assert newDynamics.total_propelling_force(0) == 9.81 + 0.5 * 0.5 * 1.225 * 10 * 10 * 1 # total propelling force = 9.81 N + 0.5 * 0.5 * 1.225 * 10 * 10 * 1 N = 9.81 N + 0.5 * 0.5 * 1.225 * 100 N = 9.81 N + 0.5 * 0.5 * 122.5 N = 9.81 N + 61.25 N = 71.06 N

    newDynamics.update_time(1)
    assert newDynamics.linear_acceleration() == 0 # linear acceleration force = 0 N

    newDynamics.update_velocity(20)
    assert newDynamics.linear_acceleration() == 1000 # linear acceleration force = 1000 N
    assert newDynamics.total_propelling_force(0) == 9.81 + 0.5 * 0.5 * 1.225 * 20 * 20 * 1 + 1000 # total propelling force = 9.81 N + 0.5 * 0.5 * 1.225 * 20 * 20 * 1 N + 1000 N = 9.81 N + 0.5 * 0.5 * 1.225 * 400 N + 1000 N = 9.81 N + 200 N + 1000 N = 1211.81 N

    newDynamics.update_velocity(30)
    assert newDynamics.linear_acceleration() == 100 * (30 - 20) / 1 # linear acceleration force = 100 kg * (30 m/s - 20 m/s) / 1 s = 1000 N
    assert newDynamics.total_propelling_force(0) == 9.81 + 0.5 * 0.5 * 1.225 * 30 * 30 * 1 + 1000 # total propelling force = 9.81 N + 0.5 * 0.5 * 1.225 * 30 * 30 * 1 N + 1000 N = 9.81 N + 0.5 * 0.5 * 1.225 * 900 N + 1000 N = 9.81 N + 450 N + 1000 N = 1460.81 N
import old.dynamics as dynamics

class Motor:
    def __init__(self, speed = 0, slope = 0):#pedal Incline in percent
        #self.pedalIncline = pedalIncline
        #self.pedalIncline = self.current / 50
        self.slope = slope
        self.speed = speed
        self.torque = 0
        self.current = 0
        
        wheel_radius = 0.285496 #m 
        mass = 317.515 #kg
        rolling_friction = .004
        static_friction = .7
        drag_coefficient = .2
        frontal_area = 1 #m^2
        self.dynamics = dynamics.Dynamics(mass, rolling_friction, static_friction, drag_coefficient, self.speed, frontal_area,wheel_radius)
    
        
    def updateParameters(self, speed, slope, time):
        self.slope = slope
        self.speed = speed
        self.dynamics.updateParams(speed, time)
        
    #current provided by motor calculated by force from Dynamics
    def currentMotor(self):
        #parameters need to be verified

        print(f"slope: {self.slope}")
        print("Propelling force: {0:.2f} N".format(self.dynamics.total_propelling_force(self.slope)))
        self.torque = self.dynamics.total_torque(self.slope)
        print("Motor Torque: {0:.2f} N-m".format(self.torque))
        self.current = (29/34)*self.torque + 1 #equation from torque-current curve on motor data sheet
        return self.current

#the current the motor would draw = percentage of the pedal incline * max current allowed (50 Amps)
#pedal incline can be output to driver if needed
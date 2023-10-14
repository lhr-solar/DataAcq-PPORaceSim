import math

class Dynamics:
    
    GRAVITY = 9.81
    AIR_MASS_DENSITY = 1.225 # kg/m^3
    
    def __init__(self, mass, tire_frict, drag_coeff, velocity, frontal_area, tire_radius, time):
        self.mass = mass
        self.tire_frict = tire_frict
        self.drag_coeff = drag_coeff
        self.velocity = velocity
        self.prev_velocity = velocity
        self.frontal_area = frontal_area
        self.tire_radius = tire_radius
        self.time = time
        self.prev_time = 0
        
    def update_params(self, time, velocity):
        self.prev_velocity = self.velocity
        self.velocity = velocity
        
        self.prev_time = self.time
        self.time = time
        
    def inertial_force(self):
        return self.mass * (self.velocity - self.prev_velocity) / (self.time - self.prev_time)
    
    def road_slope_force(self, slope): #
        return self.mass * self.GRAVITY * math.sin(slope)
    
    def road_load_force(self, slope):
        return self.mass * self.GRAVITY * self.tire_frict * math.cos(slope)
    
    def aerodynamic_drag_force(self):
        return 0.5 * self.AIR_MASS_DENSITY * self.drag_coeff * self.frontal_area * self.velocity**2
    
    def total_force(self, slope):
        return self.inertial_force() + self.road_slope_force(slope) + self.road_load_force(slope) + self.aerodynamic_drag_force()
    
    def total_torque(self, slope):
        return self.total_force(slope) * self.tire_radius
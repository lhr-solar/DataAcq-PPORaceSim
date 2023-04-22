import math
import datetime
from old.weather import Weather


#model for a solar cell array
class Array:
    def __init__(self, weather, temperature=25):
        self.num_c60 = 158 #temporary number of cells
        self.num_e60 = 88
        self.solarcell_c60 = SolarCell(153.328, 0.225, temperature, 0.00342) #unsure of the temperature_coefficients
        self.solarcell_e60 = SolarCell(153.328, 0.237, temperature, 0.00363)
        self.weather = weather
        self.temperature = temperature

    def calculateIrradiance(self):
        #angle_to_sun = Weather.get_angle_to_sun(self.weather)
        intensity_from_sun = self.weather.get_intensity()
        return intensity_from_sun
    
    def update_temperature(self, temperature):
        self.temperature = temperature
        

    def get_power(self):
        self.solarcell_c60.update_temperature(self.temperature)
        self.solarcell_e60.update_temperature(self.temperature)
        irradiance = self.calculateIrradiance()
        return self.solarcell_c60.get_power_gen(irradiance) * self.num_c60 + self.solarcell_e60.get_power_gen(irradiance) * self.num_e60
    
    
    
#model for a solar cell in a solar panel
class SolarCell:
    #@param area is in cm^2, efficiency is a percentage, temperature is in Celsius
    #@param temperature coefficient is in %/C, negative value for how much it decreases
    #self.area is in m^2
    def __init__(self, area, efficiency, temperature, temperature_coefficient):
        self.area = area / 10000
        self.efficiency = efficiency
        self.temperature = temperature
        self.temperature_coefficient = temperature_coefficient
        
        self.standard_temp = 25 #standard test condition temperature
        self.standard_intensity = 1000 #standard test condition intensity

    def get_area(self):
        return self.area

    def get_efficiency(self):
        return self.efficiency

    def update_temperature(self, temperature):
        self.temperature = temperature
       
    #light_intensity is in W/m^2
    def get_power_gen(self, light_intensity):
        temperature_loss = (self.temperature - self.standard_temp) * self.temperature_coefficient
        return self.area * self.efficiency * light_intensity * (1 - temperature_loss)
    
'''
if __name__ == "__main__":
    weather = Weather(30,-97,datetime.datetime.now(datetime.timezone.utc))
    array = Array(weather, 35)
    print(array.get_power())

'''
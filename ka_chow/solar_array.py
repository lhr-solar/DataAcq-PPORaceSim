import ka_chow.weather as weather


class Array:
    def __init__(self, car, temperature):
        self.weather = weather.Weather(car.lat, car.lon, car.time)
        # TODO: self.weather.get_temperature() #add method to weather
        self.temperature = temperature

        self.num_c60 = 158  # temporary number of cells
        self.num_e60 = 88
        AREA_C60 = 153.328  # cm^2
        AREA_E60 = 153.328  # cm^2
        EFFICIENCY_C60 = 0.225  # percentage
        EFFICIENCY_E60 = 0.237  # percentage
        TEMP_COEFF_C60 = 0.00342  # percentage #VERIFY
        TEMP_COEFF_E60 = 0.00363  # percentage
        self.solarcell_c60 = SolarCell(
            AREA_C60, EFFICIENCY_C60, self.temperature, TEMP_COEFF_C60)
        self.solarcell_e60 = SolarCell(
            AREA_E60, EFFICIENCY_E60, self.temperature, TEMP_COEFF_E60)

        self.car = car

    def calculate_irradiance(self):
        #angle_to_sun = Weather.get_angle_to_sun(self.weather)
        intensity_from_sun = self.weather.get_intensity()
        return intensity_from_sun

    def get_power(self):
        self.weather.pull_weather_data(self.car.time)
        self.solarcell_c60.update_temperature(self.temperature)
        self.solarcell_e60.update_temperature(self.temperature)
        irradiance = self.calculate_irradiance()
        return self.solarcell_c60.get_power_gen(irradiance) * self.num_c60 + self.solarcell_e60.get_power_gen(irradiance) * self.num_e60


# model for a solar cell in a solar panel
class SolarCell:
    # @param area is in cm^2, efficiency is a percentage, temperature is in Celsius
    # @param temperature coefficient is in %/C, negative value for how much it decreases
    # self.area is in m^2
    def __init__(self, area, efficiency, temperature, temperature_coefficient):
        self.area = area / 10000
        self.efficiency = efficiency
        self.temperature = temperature
        self.temperature_coefficient = temperature_coefficient

        self.standard_temp = 25  # standard test condition temperature
        self.standard_intensity = 1000  # standard test condition intensity

    def get_area(self):
        return self.area

    def get_efficiency(self):
        return self.efficiency

    def update_temperature(self, temperature):
        self.temperature = temperature

    # light_intensity is in W/m^2
    def get_power_gen(self, light_intensity):
        temperature_loss = (self.temperature -
                            self.standard_temp) * self.temperature_coefficient
        return self.area * self.efficiency * light_intensity * (1 - temperature_loss)

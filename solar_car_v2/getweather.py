import requests
import pandas as pd
import pvlib
import datetime
import numpy as np


class Weather:
    def __init__(self, step_size: float):
        self.time = 0
        self.step_size = step_size 
        self.df = pd.read_csv('solar_car_v2/weather5min.csv')
        # self.df.rename(columns={"GHI":"Global Horizontal Irradiance", "DHI": "Diffuse horizontal irradiance", "DNI":"Direct Normal Irradiance"})
        
        
    def update(self, heading):
        self.time += self.step_size
        self.heading = heading

    def get_attribute(self, attribute):
        attr = self.df[attribute]
        index = int(self.time / 60 / 5) 
        return attr[index]

    def get_irradiance(self):
        location = pvlib.location.Location(latitude=30.26, longitude=-97.75) #Location for Austin right now
        surface_df = pvlib.tracking.calc_surface_orientation(self.heading, 0, 90)
        surface_tilt, surface_azimuth = surface_df["surface_tilt"], surface_df["surface_azimuth"]

        times = pd.to_datetime(self.df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
        solar_position = location.get_solarposition(times)
        df_poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt,
            surface_azimuth,
            dni=self.get_attribute('DNI'),
            ghi=self.get_attribute('GHI'),
            dhi=self.get_attribute('DHI'),
            dni_extra=None, 
            airmass=None,
            albedo=0.25, 
            surface_type=None,
            solar_zenith = solar_position['apparent_zenith'],
            solar_azimuth = solar_position['azimuth'],
            model = 'isotropic'
        )
        index = int(self.time / 60 / 5) 
        return df_poa.iloc[index, :]
    
    def cell_temp(self):
        params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
        cell_temp = pvlib.temperature.sapm_cell(
            self.get_irradiance()['poa_global'],
            temp_air = self.get_attribute('Temperature'),
            wind_speed = self.get_attribute('Wind Speed'),
            a = params['a'],
            b = params['b'],
            deltaT = params['deltaT'],
            irrad_ref = 1000
        )
        return cell_temp
    
    def dc_power(self):
        """
        power output of an array in direct current
        """
        gamma_pdc = -0.004  # divide by 100 to go from %/°C to 1/°C
        nameplate = 1e3
        power = pvlib.pvsystem.pvwatts_dc(
            self.get_irradiance()['poa_direct'], 
            self.cell_temp(),
            nameplate, 
            gamma_pdc
            )
        return power

    # def IV_curve(self):
    #     matrix = np.meshgrid(self.cell_temp(), self.get_irradiance['poa_direct'])
    #     #creates a combination of temperature and irradiance conditions
    #     #np.meshgrid() makes a grid of all combinations
    #     CECmoduleinfo = pvlib.pvsystem.retrieve_sam(name = "CECMOD")
    #     #cali energy commision to get a bunch of info at STC
    #     monocrys = CECmoduleinfo['Canadian_Solar_Inc__CS5P_220M']
    #     CECparams = pvlib.pvsystem.calcparams_cec(
    #         effective_irradiance = self.get_irradiance['poa_direct'],
    #         temp_cell = self.cell_temp(),
    #         alpha_sc = monocrys.alpha_sc,
    #         a_ref = monocrys.a_ref,
    #         I_L_ref = monocrys.I_L_ref,
    #         I_o_ref = monocrys.I_o_ref,
    #         R_sh_ref = monocrys.R_sh_ref,
    #         R_s = monocrys.R_s,
    #         Adjust = monocrys.Adjust,
    #         EgRef=1.121, 
    #         dEgdT=-0.0002677,
    #         irrad_ref=1000, 
    #         temp_ref=25
    #     )


        
w = Weather(0.01)
w.update(90)
# print(w.get_irradiance())
print(w.cell_temp())
print(w.dc_power())

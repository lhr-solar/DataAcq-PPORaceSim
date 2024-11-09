from ..PVSource.PVCell.PVCellNonideal import PVCellNonideal


class SolarCell:
    """
    @param area is in cm^2
    @param efficiency is a percentage
    @param annualRadiation is in kWh/m2/year
    @param performanceRatio is a coefficient
    """
    pv_model = PVCellNonideal()
    num_cells = 10  # Number of cells in the array
    voltage = 18  # Operating voltage (V)
    irradiance = 800  # Irradiance in W/m²
    temperature = 25  # Temperature in Celsius
    time_period = 3600  # Time period in seconds (e.g., 1 hour)

    def __init__(self, area, efficiency, annualRadiation, performanceRatio):
        self.area = area / 10000
        self.efficiency = efficiency
        self.annualRadiation = annualRadiation
        self.performanceRatio = performanceRatio
        
    def get_energy_gen(self): #How much energy we get from 1 array
        energygenerated = self.pv_model.getEnergygen(numCells = 1, voltage = 0, irradiance = 0.001, temperature = 0, time_period = 1)
        return energygenerated
        #return self.area * self.efficiency * self.annualRadiation * self.performanceRatio


class SolarArray:
    def __init__(self):
        self.annualRadiation = 1500 # temporary
        self.performanceRatio = 0.5 # temporary
        self.num_c60 = 158  # temporary number of cells
        self.num_e60 = 88
        AREA_C60 = 153.328  # cm^2
        AREA_E60 = 153.328  # cm^2
        EFFICIENCY_C60 = 0.225  # percentage
        EFFICIENCY_E60 = 0.237  # percentage

        self.solarcell_c60 = SolarCell(AREA_C60, EFFICIENCY_C60, self.annualRadiation, self.performanceRatio)
        self.solarcell_e60 = SolarCell(AREA_E60, EFFICIENCY_E60, self.annualRadiation, self.performanceRatio)
        
    def step(self): # total energy we get from entire array
        return self.solarcell_c60.get_energy_gen() * self.num_c60 + self.solarcell_e60.get_energy_gen() * self.num_e60
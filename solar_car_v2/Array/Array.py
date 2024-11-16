import math
import numpy as np

from scipy import constants

from cell import Cell
from utils import normalize


class ThreeParamCell(Cell):
    def __init__(self, params: dict, data_fp=None) -> None:
        super().__init__(params=params, data_fp=data_fp)

    def get_voltage(
        self, current: float, irrad: float, temp: float
    ) -> float:
        if irrad == 0.0:
            raise Exception("Incident irradiance is too low!")
        if temp == 0.0:
            raise Exception("Cell temperature is too low!")

        # Reference parameters
        ref_g = self._params["ref_irrad"]
        ref_v_oc = self._params["ref_voc"]
        ref_i_sc = self._params["ref_isc"]

        # Curve Fitting parameters
        fit_n1 = self._params["fit_fwd_ideality_factor"]
        fit_n2 = self._params["fit_rev_ideality_factor"]
        fit_i_d = self._params["fit_rev_sat_curr"]

        if fit_n1 == 0.0 or fit_n2 == 0.0:
            raise Exception("Cell ideality factor is too low!")

        i_l = current
        g = irrad
        t_c = temp

        k_b = constants.k
        q = constants.e

        v_t = k_b * t_c / q
        i_sc = ref_i_sc * g / ref_g

        # Add 0.00001 for satisfying the domain condition when g/ref_g = 0.
        v_oc = ref_v_oc + fit_n1 * v_t * math.log((g / ref_g) + 0.00001)

        if i_l <= i_sc - 1 * 10**-10:
            v_l = (fit_n1 * v_t) * math.log(
                (1 - i_l / i_sc) * (math.exp(v_oc / (fit_n1 * v_t)) - 1)
            )
        else:
            v_l = -math.log((i_l - i_sc) / fit_i_d + 1) * fit_n2 * v_t

        return v_l

    def get_current(
        self, voltage: float, irrad: float, temp: float
    ) -> float:

        if irrad == 0.0:
            raise Exception("Incident irradiance is too low!")
        if temp == 0.0:
            raise Exception("Cell temperature is too low!")

        # Reference parameters
        ref_g = self._params["ref_irrad"]
        ref_v_oc = self._params["ref_voc"]
        ref_i_sc = self._params["ref_isc"]

        # Curve Fitting parameters
        fit_n1 = self._params["fit_fwd_ideality_factor"]
        fit_n2 = self._params["fit_rev_ideality_factor"]
        fit_i_d = self._params["fit_rev_sat_curr"]

        if fit_n1 == 0.0 or fit_n2 == 0.0:
            raise Exception("Cell ideality factor is too low!")

        v_l = voltage
        g = irrad
        t_c = temp

        k_b = constants.k
        q = constants.e

        v_t = k_b * t_c / q
        i_sc = ref_i_sc * g / ref_g

        # Add 0.00001 for satisfying the domain condition when g/ref_g = 0.
        v_oc = ref_v_oc + v_t * math.log((g / ref_g) + 0.00001)

        if v_l > 0.0:
            if v_l / v_t > 100:
                # Domain assumption that our load voltage cannot be well past open
                # circuit voltage: the ratio of load voltage versus thermal voltage
                # can overfill the exponential term.
                return 0.0

            i_l = i_sc * (
                1
                - (math.exp(v_l / (fit_n1 * v_t)) - 1) / (math.exp(v_oc / (fit_n1 * v_t)) - 1)
            )
        else:
            i_l = fit_i_d * (math.exp(-v_l / (fit_n2 * v_t)) - 1) + i_sc

        return i_l

class SolarCell:
    """
    @param area is in cm^2
    @param efficiency is a percentage
    @param annualRadiation is in kWh/m2/year
    @param performanceRatio is a coefficient
    """
    # pv_model = ThreeParamCell()

    # Variables used: current, irrad, temp, voltage, power = current*voltage
    num_cells = 10  # Number of cells in the array
    time_period = 3600  # Time period in seconds (e.g., 1 hour)
    voltage =  0 # Operating voltage (V)


    def __init__(self, area, efficiency, annualRadiation, performanceRatio):
        self.area = area / 10000
        self.efficiency = efficiency
        self.annualRadiation = annualRadiation
        self.performanceRatio = performanceRatio


class SolarArray(SolarCell):
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




a = ThreeParamCell(params = {
        "ref_irrad": 1000.0,  # W/m^2
        "ref_temp": 298.15,  # Kelvin
        "ref_voc": 0.721,  # Volts
        "ref_isc": 6.15,  # Amps
        "fit_fwd_ideality_factor": 2,
        "fit_rev_ideality_factor": 1,
        "fit_rev_sat_curr": 1 * 10**-5,
    })


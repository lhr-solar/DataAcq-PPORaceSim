import time
import logging
import os
from solar_car.battery.solar_car_battery_manager import SolarCarBatteryManager

import liionpack as lp
import pybamm
import numpy as np

parameter_values = pybamm.ParameterValues("Chen2020")


class Battery:
    """
    The battery class is the battery of the solar car. It contains the battery model and the battery manager. It modifies PyBaMM and liionpack to work on a step by step basis.

    Parameters
    ----------
    time_step : float, optional
        The time step length of the simulation in seconds. The default is 1.0.
    """

    current_draw = 0
    _step = 1

    def __init__(self, time_step: float):
        self.time_step = time_step

        # Need accurate numbers for this. Let's just assume Telsa for now?
        self.np = 14  # number of parallel cells
        self.ns = 6  # number of series cells
        self.netlist = lp.setup_circuit(
            self.np, self.ns, V=25, I=300)

        logging.info("Initializing battery simulation")
        start = time.time()

        output_variables = [
            "X-averaged negative particle surface concentration [mol.m-3]",
        ]
        lp.logger.disabled = True
        self.sim = SolarCarBatteryManager()
        self.sim.solve(
            netlist=self.netlist,
            sim_func=lp.basic_simulation,
            parameter_values=parameter_values,
            output_variables=output_variables,
            inputs=None,
            initial_soc=1,
            nproc=0,
            dt=self.time_step,
            setup_only=True,
        )
        # the first two steps are to prevent division by zero errors caused by the first two steps being 0
        logging.info("Battery setup complete")
        self.sim._step(0, None)
        self.sim.step = 0
        self.full_charge = np.average(self._output(
        )["X-averaged negative particle surface concentration [mol.m-3]"][-1])
        end = time.time()
        logging.info(f'Battery initialized in {end - start} seconds')

    def set_draw(self, current: float):
        self.current_draw = current

    def step(self):
        try:
            start = time.time()
            self.sim.protocol[self._step] = self.current_draw
            ok = self.sim._step(self._step, None)
            self.sim.step = self._step
            self._step += 1
            end = time.time()
            logging.debug(f'Battery step complete in {end - start} seconds')
            return ok
        except Exception as e:
            logging.fatal(f'Battery step failed: {e}')
            return False

    def _output(self):
        return self.sim.step_output()

    def get_soc(self) -> float:
        """
        Returns the state of charge of the battery.

        Returns
        -------
        float
            The state of charge of the battery between 0 and 1.
        """
        output = self._output()
        average = np.average(
            output["X-averaged negative particle surface concentration [mol.m-3]"][-1])
        return average / self.full_charge

    def get_voltage(self) -> float:
        return self._output()["Pack terminal voltage [V]"][-1] * self.np

    def get_cell_voltage(self) -> float:
        return self._output()["Pack terminal voltage [V]"][-1]


if __name__ == "__main__":
    battery = Battery(1)
    battery.step()
    print(battery.get_soc())

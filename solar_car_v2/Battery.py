import time
import logging
import os

import liionpack as lp
import pybamm
import numpy as np
from liionpack import CasadiManager
from Array import ThreeParamCell
from getweather import get_irradiance


parameter_values = pybamm.ParameterValues("Chen2020")

        
class Battery:
    """
    The battery class is the battery of the solar car. It contains the battery model and the battery manager. 
    It modifies PyBaMM and liionpack to work on a step by step basis.

    Parameters
    ----------
    time_step : float, optional
        The time step length of the simulation in seconds. The default is 1.0.
    """
    pv_model = ThreeParamCell(params = {
        "ref_irrad": 1000.0,  # W/m^2
        "ref_temp": 298.15,  # Kelvin
        "ref_voc": 0.721,  # Volts
        "ref_isc": 6.15,  # Amps
        "fit_fwd_ideality_factor": 2,
        "fit_rev_ideality_factor": 1,
        "fit_rev_sat_curr": 1 * 10**-5,
    })


    
    num_cells = 10  # Number of cells in the array
    voltage = 0.647  # mpp(V) from powergen data should be updating based on time though
    irradiance = get_irradiance()  # from get wehather
    temperature = 25  # Temperature in Celsius
    time_period = 3600  # Time period in seconds (e.g., 1 hour)
    current_draw = 1.2 #mpp(I) from powergen data
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

    def set_draw(self):
        self.current_draw = self.pv_model.getCurrent(voltage=self.voltage, irradiance=self.irradiance, temperature=self.temperature)

    def update(self, voltage, irradiance, temperature):
        self.voltage = voltage
        self.irradiance = irradiance
        self.temperature = temperature

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

class SolarCarBatteryManager(CasadiManager):
    """
    This is a custom battery manager for the solar car. It is a subclass of the CasadiManager class from liionpack. It is used to simulate the battery in a step by step manner.
    This manager was created so that the battery could be simulated in a step by step manner, rather PyBaMM's experiment model.

    The primary change is overwriting the solve method. 
    It has the additional parameters, dt, and minSteps and removed experiment. 
    The method is close to the original but experiment is no longer used rather for the variable protocol, an array, to handle current demand step by step. 
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def solve(
        self,
        netlist,
        sim_func,
        parameter_values,
        inputs,
        output_variables,
        initial_soc,
        nproc,
        dt=1,
        minSteps = 100000,
        setup_only=False,
    ):
        self.netlist = netlist
        self.sim_func = sim_func

        self.parameter_values = parameter_values
        self.check_current_function()
        # Get netlist indices for resistors, voltage sources, current sources
        self.Ri_map = netlist["desc"].str.find("Ri") > -1
        self.V_map = netlist["desc"].str.find("V") > -1
        self.I_map = netlist["desc"].str.find("I") > -1
        self.Terminal_Node = np.array(netlist[self.I_map].node1)
        self.Nspm = np.sum(self.V_map)

        self.split_models(self.Nspm, nproc)

        # Generate the protocol from the supplied experiment
        # self.protocol = lp.generate_protocol_from_experiment(
        #     experiment, flatten=True)
        # self.dt = experiment.period
        # self.Nsteps = len(self.protocol)
        self.dt = dt
        self.Nsteps = minSteps
        self.protocol = np.array([1] + [0] * (self.Nsteps - 1))
        netlist.loc[self.I_map, ("value")] = self.protocol[0]
        # Solve the circuit to initialise the electrochemical models
        V_node, I_batt = lp.solve_circuit_vectorized(netlist)

        # The simulation output variables calculated at each step for each battery
        # Must be a 0D variable i.e. battery wide volume average - or X-averaged for
        # 1D model
        self.variable_names = [
            "Terminal voltage [V]",
            "Surface open-circuit voltage [V]",
        ]
        if output_variables is not None:
            for out in output_variables:
                if out not in self.variable_names:
                    self.variable_names.append(out)
            # variable_names = variable_names + output_variables
        self.Nvar = len(self.variable_names)

        # Storage variables for simulation data
        self.shm_i_app = np.zeros([self.Nsteps, self.Nspm], dtype=np.float32)
        self.shm_Ri = np.zeros([self.Nsteps, self.Nspm], dtype=np.float32)
        self.output = np.zeros(
            [self.Nvar, self.Nsteps, self.Nspm], dtype=np.float32)

        # Initialize currents in battery models
        self.shm_i_app[0, :] = I_batt * -1

        # Step forward in time
        self.V_terminal = np.zeros(self.Nsteps, dtype=np.float32)
        self.record_times = np.zeros(self.Nsteps, dtype=np.float32)

        self.v_cut_lower = parameter_values["Lower voltage cut-off [V]"]
        self.v_cut_higher = parameter_values["Upper voltage cut-off [V]"]

        # Handle the inputs
        self.inputs = inputs

        self.inputs_dict = lp.build_inputs_dict(
            self.shm_i_app[0, :], self.inputs, None)
        # Solver specific setup
        self.setup_actors(nproc, self.inputs_dict, initial_soc)
        # Get the initial state of the system
        self.evaluate_actors()
        if not setup_only:
            self._step_solve_step(None)
            return self.step_output()
import time
import logging

import liionpack as lp
import pybamm
import numpy as np
from liionpack import CasadiManager

chen_values = pybamm.ParameterValues("Chen2020")


class Battery:
    """
    The battery class is the battery of the solar car. It contains the battery model and the battery manager.
    It modifies PyBaMM and liionpack to work on a step by step basis.

    Parameters
    ----------
    time_step : float, optional
        The time step length of the simulation in seconds. The default is 1.0.
    """

    current_draw = 1.2  # mpp(I) from powergen data
    _step = 1

    parameter_values = pybamm.ParameterValues(
        {
            **chen_values,
            # cell
            "Nominal cell capacity [A.h]": 45,  # 5 Ah per cell, 9 per module
            "Contact resistance [Ohm]": 0,
            "Cell cooling surface area [m2]": 0.104,
            "Cell volume [m3]": 0.00208,
            "Cell thermal expansion coefficient [m.K-1]": 1.1e-06,
            # separator
            "Separator porosity": 0.47,
            "Separator Bruggeman coefficient (electrolyte)": 1.5,
            "Separator density [kg.m-3]": 397.0,
            "Separator specific heat capacity [J.kg-1.K-1]": 700.0,
            "Separator thermal conductivity [W.m-1.K-1]": 0.16,
            # sim
            "Reference temperature [K]": 298.15,
            "Ambient temperature [K]": 298.15,
            "Initial temperature [K]": 298.15,
            # random guess from Chat
            "Total heat transfer coefficient [W.m-2.K-1]": 5.53,
        }
    )

    def __init__(self, time_step: float):
        self.time_step = time_step

        self.np = 1  # number of parallel cells
        self.ns = 32  # number of series cells
        self.netlist = lp.setup_circuit(
            self.np,
            self.ns,
            V=120,
            I=40,
            Ri=13.5e-3,
        )

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
            parameter_values=self.parameter_values,
            output_variables=output_variables,
            inputs=None,
            initial_soc=1,
            nproc=1,
            dt=self.time_step,
            setup_only=True,
        )
        # the first two steps are to prevent division by zero errors caused by the first two steps being 0
        logging.info("Battery setup complete")
        self.sim._step(0, None)
        self.sim.step = 0
        self.full_charge = np.average(
            self._output()[
                "X-averaged negative particle surface concentration [mol.m-3]"
            ][-1]
        )
        end = time.time()
        logging.info(f"Battery initialized in {end - start} seconds")

    def update(self, current: float):
        self.current_draw = current

    def step(self):
        start = time.time()
        self.sim.protocol[self._step] = self.current_draw
        ok = self.sim._step(self._step, None)
        self.sim.step = self._step
        self._step += 1
        end = time.time()
        logging.debug(f"Battery step complete in {end - start} seconds")
        return ok

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
            output["X-averaged negative particle surface concentration [mol.m-3]"][-1]
        )
        return average / self.full_charge

    def get_voltage(self) -> float:
        return self._output()["Pack terminal voltage [V]"][-1] * self.np

    def get_cell_voltage(self) -> float:
        return self._output()["Pack terminal voltage [V]"][-1]


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
        minSteps=10000,
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
        self.split_models(
            self.Nspm, nproc
        )  # splits simulation to allow parallel computation

        # Generate the protocol from the supplied experiment
        # self.protocol = lp.generate_protocol_from_experiment(
        #     experiment, flatten=True)
        # self.dt = experiment.period
        # self.Nsteps = len(self.protocol)
        self.dt = dt
        self.Nsteps = minSteps
        self.protocol = np.array([1, -1] + [0] * (self.Nsteps - 2))
        netlist.loc[self.I_map, ("value")] = self.protocol[0]
        # Solve the circuit to initialise the electrochemical models
        V_node, I_batt = lp.solve_circuit_vectorized(
            netlist
        )  # from liionpack is for calculating power loss through heating of resistors in a cicruit

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

        # Storage variables for simulation data - allocate memory
        self.shm_i_app = np.zeros([self.Nsteps, self.Nspm], dtype=np.float32)
        self.shm_Ri = np.zeros([self.Nsteps, self.Nspm], dtype=np.float32)
        self.output = np.zeros([self.Nvar, self.Nsteps, self.Nspm], dtype=np.float32)

        # Initialize currents in battery models
        self.shm_i_app[0, :] = I_batt * -1

        # Step forward in time
        self.V_terminal = np.zeros(self.Nsteps, dtype=np.float32)
        # self.V_terminal = V_node[self.Terminal_Node]
        self.record_times = np.zeros(self.Nsteps, dtype=np.float32)

        # voltage cutoff values
        self.v_cut_lower = parameter_values["Lower voltage cut-off [V]"]
        self.v_cut_higher = parameter_values["Upper voltage cut-off [V]"]

        # Handle the inputs
        self.inputs = inputs

        self.inputs_dict = lp.build_inputs_dict(self.shm_i_app[0, :], self.inputs, None)
        # Solver specific setup
        self.setup_actors(nproc, self.inputs_dict, initial_soc)
        # Get the initial state of the system
        self.evaluate_actors()
        if not setup_only:
            self._step_solve_step(None)
            return self.step_output()

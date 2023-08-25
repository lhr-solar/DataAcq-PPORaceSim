import liionpack as lp
import numpy as np
from liionpack import CasadiManager

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
        minSteps = 50000,
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
        
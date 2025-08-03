# =======================================================================================
# PROJECT CHRONO - http://projectchrono.org
#
# Copyright (c) 2021 projectchrono.org
# All right reserved.
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE file at the top level of the distribution and at
# http://projectchrono.org/license-chrono.txt.
#
# =======================================================================================
# Authors: Huzaifa Unjhawala, Json Zhou
# =======================================================================================
#
# This file contains a gym environment for the cobra rover in a terrain of 20 x 20. The
# environment is used to train the rover to reach a goal point in the terrain. The goal
# point is randomly generated in the terrain. The rover is initialized at the center of
# the terrain. Obstacles can be optionally set (default is 0).
#
# =======================================================================================
#
# Action Space: The action space is normalized throttle and steering between -1 and 1.
# multiply against the max wheel angular velocity and wheel steer angle to provide the
# wheel angular velocity and wheel steer angle for all 4 wheels of the cobra rover model.
# Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)
#
# =======================================================================================
#
# Observation Space: The observation space is a 1D array consisting of the following:
# 1. Delta x of the goal in local frame of the vehicle
# 2. Delta y of the goal in local frame of the vehicle
# 3. Vehicle heading
# 4. Heading needed to reach the goal
# 5. Velocity of vehicle
# =======================================================================================


# Chrono imports
import pychrono as chrono
from pychrono import vehicle as veh
from pychrono import irrlicht

# Standard Python imports
import numpy as np

# Gym chrono imports
# Custom imports
from .chrono_base import ChronoBaseEnv

# Gymnasium imports
import gymnasium as gym
from .track import generate_path, generate_terrain

from .getweather import Weather

# Array and battery
from .Battery import Battery
from .Array.Array import ThreeParamCell


class SolarCar(ChronoBaseEnv):
    max_speed = 97.0
    """
    Max speed of the solar car in km/h. \n
    97 km/h ~= 60 mph \n 
    """

    action_space = gym.spaces.Box(low=(0), high=max_speed, shape=(1,), dtype=np.float64)
    """
    Action space for the solar car environment. \n
    shape: [desired_speed] \n
    """

    observation_space = gym.spaces.Box(low=-20, high=20, shape=(5,), dtype=np.float64)
    """
    Observation space for the solar car environment. \n
    shape: [distance along track, velocity, slope, battery %, solar radiation, future solar radiation] \n
    """

    init_pos = chrono.ChVector3d(0, 0, 0)
    """
    Initial position of the solar car. \n
    """

    road_length = 1.8288
    """
    Half the length of the road in meters. \n
    6 feet in the US.
    """

    max_time = 28800
    """
    Max time of the simulation in seconds. \n
    8 hours in seconds. Typical raceday.
    """

    step_size = 1e-3
    """
    Step size of the simulation in seconds. \n
    """

    steps_per_action = round(1 / (step_size * 10))
    """
    Number of steps per action. \n
    """

    def __init__(self, render_mode="human"):
        chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(1.0)
        chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.1)

        ChronoBaseEnv.__init__(self, render_mode)

        self.render_mode = render_mode

        veh.SetDataPath(chrono.GetChronoDataPath())

        # Terain JSON specification file
        self.rigidterrain_file = veh.GetDataFile("terrain/RigidPlane.json")

        # HMMWV specification files (vehicle, powertrain, and tire models)
        self.vehicle_file = veh.GetDataFile("gator/json/Gator_Vehicle.json")
        self.engine_file = veh.GetDataFile("gator/json/Gator_EngineSimple.json")
        # self.engine_file = veh.GetDataFile("gator/Gator_EngineSimple.json")
        self.transmission_file = veh.GetDataFile(
            "gator/json/Gator_AutomaticTransmissionSimpleMap.json"
        )
        self.tire_file = veh.GetDataFile("gator/json/Gator_Wheel.json")
        self.rigidtire_file = veh.GetDataFile("hmmwv/tire/HMMWV_Pac02Tire.json")
        self.bodyfile = chrono.GetChronoDataFile("solid_json.json")

        self.engine = None
        self.transmission = None
        self.powertrain = None
        self.driver = None
        self.vehicle_pos = self.init_pos
        self.terrain = None
        self.speed_controller = None
        self.vis = None
        self.array = None
        self.battery = None
        self.prev_SOC = 0
        self.voltage = 0.5  # UPDATE VALUE??

        # ---------------------------------
        # Gym Environment variables
        # ---------------------------------
        self.steps = 0
        # Maximum simulation time (seconds)
        # Holds reward of the episode
        self.reward = 0
        # Position of goal as numpy array
        self.goal = None
        # Distance to goal at previos time step -> To gauge "progress"
        self._vector_to_goal = None
        self._old_distance = None
        # Observation of the environment
        self.observation = None
        # Flag to determine if the environment has terminated -> In the event of timeOut or reach goal
        self._terminated = False
        # Flag to determine if the environment has truncated -> In the event of a crash
        self._truncated = False
        # Flag to check if the render setup has been done -> Some problem if rendering is setup in reset
        self._render_setup = False
        # Flag to count success while testing
        self._success = False

        # other external systems
        self.path, self.points, self.distances = generate_path()
        self.weather = Weather(self.step_size)

        self.waypoint_rewards = [
            [chrono.ChVector3d(x, y, 0), False] for (x, y, z) in self.points
        ]

        self.array = ThreeParamCell(
            params={
                "ref_irrad": 1000.0,  # W/m^2
                "ref_temp": 298.15,  # Kelvin
                "ref_voc": 0.721,  # Volts
                "ref_isc": 6.15,  # Amps
                "fit_fwd_ideality_factor": 2,
                "fit_rev_ideality_factor": 1,
                "fit_rev_sat_curr": 1 * 10**-5,
            }
        )

        self.soc = 1
        self.vehicle = None

    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state -> Set up for standard gym API

        Args:
            seed: Seed for the random number generator
            options: Options for the simulation (dictionary)
        """

        self.m_system = chrono.ChSystemNSC()
        self.m_system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
        self.m_system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        self.path, self.points, self.distances = generate_path()

        self.vehicle = veh.WheeledVehicle(self.m_system, self.vehicle_file)
        starting_point = self.path.Eval(0, 0)
        heading = self.path.EvalDer(0, 0)
        starting_point.z = 0.25

        ang = np.arctan2(heading.y, heading.x)

        self.vehicle.Initialize(
            chrono.ChCoordsysd(starting_point, ang, chrono.ChVector3d(0, 0, 1))
        )

        self.vehicle.GetChassis().SetFixed(True)
        self.vehicle.SetChassisVisualizationType(veh.VisualizationType_MESH)
        self.vehicle.SetChassisRearVisualizationType(veh.VisualizationType_PRIMITIVES)
        self.vehicle.SetSuspensionVisualizationType(veh.VisualizationType_MESH)
        self.vehicle.SetSteeringVisualizationType(veh.VisualizationType_MESH)
        self.vehicle.SetWheelVisualizationType(veh.VisualizationType_MESH)
        self.vehicle.SetTireVisualizationType(veh.VisualizationType_MESH)

        # Create and initialize the powertrain system
        self.engine = veh.ReadEngineJSON(self.engine_file)
        self.transmission = veh.ReadTransmissionJSON(self.transmission_file)
        self.powertrain = veh.ChPowertrainAssembly(self.engine, self.transmission)
        self.vehicle.InitializePowertrain(self.powertrain)

        # self.vehicle.
        # self.vehicle.SetTireType(veh.TireModelType_TMEASY)
        # self.vehicle.SetTireStepSize(self.step_size)

        for wheel in self.vehicle.GetAxles()[0].GetWheels():
            tire = veh.ReadTireJSON(
                veh.GetDataFile("gator/json/Gator_TMeasyTireFront.json")
            )
            self.vehicle.InitializeTire(
                tire, wheel, veh.VisualizationType_MESH, veh.TireModelType_RIGID
            )

        for wheel in self.vehicle.GetAxles()[1].GetWheels():
            tire = veh.ReadTireJSON(
                veh.GetDataFile("gator/json/Gator_TMeasyTireRear.json")
            )
            self.vehicle.InitializeTire(
                tire, wheel, veh.VisualizationType_MESH, veh.TireModelType_RIGID
            )

        # for axle in self.vehicle.GetAxles():
        #     tireL = veh.RigidTire(self.rigidtire_file)
        #     self.vehicle.InitializeTire(
        #         tireL, axle.m_wheels[0], veh.VisualizationType_MESH
        #     )
        #     tireR = veh.RigidTire(self.rigidtire_file)
        #     self.vehicle.InitializeTire(
        #         tireR, axle.m_wheels[1], veh.VisualizationType_MESH
        #     )

        # Initialize the vehicle position -> get gator_theta to set the goal position
        # self.vehicle.Initialize(chrono.ChCoordsysD(initloc, initrot))

        self.vehicle.GetSystem().SetCollisionSystemType(
            chrono.ChCollisionSystem.Type_BULLET
        )
        self.vehicle.GetChassisBody().EnableCollision(True)
        self.vehicle.GetChassisBody().SetFixed(False)
        self.vehicle.GetSystem().GetSolver().AsIterative().SetMaxIterations(480)
        a = chrono.ImportSolidWorksSystem(self.bodyfile)

        self.steps = 0

        # arbitrary step size for battery for now
        self.battery = Battery(self.step_size * self.steps_per_action * 10)

        # Create the terrain, we probably want the terrain to match the path
        self.terrain = generate_terrain(self.vehicle.GetSystem(), self.path)
        self.terrain.Initialize()

        # We should make the path more complex here, but this is fine for now
        self.driver = veh.ChPathFollowerDriver(
            self.vehicle,
            self.path,
            "my_path",
            0.0,
        )
        self.driver.GetSpeedController().SetGains(0.5, 0.1, 0)
        self.driver.GetSteeringController().SetGains(0.1, 0.3, 0)
        self.driver.GetSteeringController().SetLookAheadDistance(5)
        self.driver.Initialize()

        # -----------------------------
        # Get the intial observation
        # -----------------------------
        self.observation = self.get_observation()
        self.reward = 0

        self._terminated = False
        self._truncated = False
        self._success = False

        self.waypoint_rewards = [
            [chrono.ChVector3d(x, y, 0), False] for (x, y, z) in self.points
        ]
        self.waypoint_rewards[0][1] = True

        self.render()

        return self.observation, {}

    def step(self, action):
        """Take a step in the environment - Frequency by default is 10 Hz.

        Steps the simulation environment using the given action. The action is applied for a single step.

        Args:
            action (2 x 1 np.array): Action to be applied to the environment, consisting of throttle and steering.
        """
        try:
            time = self.vehicle.GetSystem().GetChTime()

            print(action[0])
            desired_speed = action[0] / 3.6  # Convert to m/s

            self.driver.SetDesiredSpeed(desired_speed)

            if self.vehicle.GetSystem().GetNumShafts() > 0:
                shafts = self.vehicle.GetSystem().GetShafts()
                for shaft in shafts:
                    shaft.SetAppliedTorque(0)

            for _ in range(self.steps_per_action):
                driver_inputs = self.driver.GetInputs()

                self.driver.Synchronize(time)
                self.vehicle.Synchronize(time, driver_inputs, self.terrain)
                self.terrain.Synchronize(time)

                if self._render_setup:
                    self.vis.Synchronize(time, driver_inputs)
                    self.vis.Advance(self.step_size)

                self.driver.Advance(self.step_size)

                # most processing time
                self.vehicle.Advance(self.step_size)
                self.terrain.Advance(self.step_size)

                self.vehicle.GetSystem().DoStepDynamics(self.step_size)

            # Get the observation
            self.observation = self.get_observation()
            # Get reward
            self.reward += self.get_reward(desired_speed)

            self.weather.update(self.vehicle.GetChassisBody().GetRotAngle())

            # Update Array
            # power = self.weather.dc_power()
            self.array.update(
                self.voltage,
                self.weather.get_irradiance(),
                self.weather.get_attribute("Temperature"),
            )
            current = self.array.get_current()
            self.array.step()

            # Update Battery
            if self.steps % 10 == 0:
                self.battery.update(-current)
                self.battery.step()
                self.soc = self.battery.get_soc()
        except Exception as e:
            print("Error in step: ", e)
            self._truncated = True

        # Check if we are done
        self.is_terminated()
        self.is_truncated()

        self.steps += 1

        self.render()

        return self.observation, self.reward, self._terminated, self._truncated, {}

    def render(self):
        """Render the environment"""

        # ------------------------------------------------------
        # Add visualization - only if we want to see "human" POV
        # ------------------------------------------------------
        if self.render_mode == "human":
            if self._render_setup == False:
                # self.vis = veh.ChWheeledVehicleVisualSystemIrrlicht()
                self.vis = veh.ChVehicleVisualSystemIrrlicht()

                self.vis.SetWindowTitle("HMMWV JSON specification")
                self.vis.SetWindowSize(1280, 1024)

                trackPoint = chrono.ChVector3d(0.0, 0.0, 3)
                self.vis.SetChaseCamera(trackPoint, 6.0, 0.5)

                self.vis.Initialize()
                self.vis.AddLightDirectional()
                self.vis.AddSkyBox()
                # self.vis.SetChaseCameraPosition(chrono.ChVector3d(0, 0, 5))

                self.vis.AttachVehicle(self.vehicle)

                self._render_setup = True
            self.vis.BeginScene()
            self.vis.Render()
            self.vis.EndScene()

    def get_reward(self, speed):
        """Get the reward for the current step

        Get the reward for the current step based on the distance to the goal, and the distance the robot has traveled.

        Returns:
            float: Reward for the current step
        """

        points_for_moving_forward = ((speed) / self.max_speed) * 10

        if speed == 0:
            points_for_moving_forward = -1

        waypoint_reward = 0

        for waypoint in self.waypoint_rewards:
            waypoint_pos = waypoint[0]
            car_pos = self.vehicle.GetChassis().GetPos()
            car_pos_np = np.array([car_pos.x, car_pos.y])
            waypoint_pos_np = np.array([waypoint_pos.x, waypoint_pos.y])

            if not waypoint[1]:
                if np.linalg.norm(car_pos_np - waypoint_pos_np) < 1:
                    waypoint[1] = True
                    waypoint_reward = 100
                    print("Reached waypoint")
                    print("Vehicle Position: ", self.vehicle.GetChassis().GetPos())
                    print(
                        "--------------------------------------------------------------"
                    )
                    break

        return points_for_moving_forward + waypoint_reward

    def is_terminated(self):
        """Check if the environment is terminated"""

        # If we have exceeded the max time -> Terminate
        if self.vehicle.GetSystem().GetChTime() > self.max_time:
            print("--------------------------------------------------------------")
            print("Time out")
            print("Final position of car: ", self.vehicle.GetChassis().GetPos())
            print("Reward: ", self.reward)
            print("--------------------------------------------------------------")
            self._terminated = True
        return self._terminated

    def is_truncated(self):
        """Check if the environment is truncated

        Check if the rover has fallen off the terrain, and if so truncate and give a large penalty.
        """
        if self.vehicle.GetChassis().GetPos().z < 0:
            self._truncated = True

        return self._truncated

    def get_observation(self):
        """Get the observation from the environment

        Get the observation of the environment, consisting of the distances to the goal and heading and velocity of the vehicle.

        Returns:
            observation (5 x 1 np.array): Observation of the environment consisting of:
                1. Delta x of the goal in local frame of the vehicle
                2. Delta y of the goal in local frame of the vehicle
                3. Vehicle heading
                4. Heading needed to reach the goal
                5. Velocity of vehicle
        """
        observation = np.zeros(5)

        pos = self.vehicle.GetChassis().GetPos()

        goal_pos: chrono.ChVector3d = [p for p in self.waypoint_rewards if not p[1]][0][
            0
        ]
        self.vehicle_pos = pos
        delta_x = goal_pos.x - pos.x
        delta_y = goal_pos.y - pos.y
        observation[0] = delta_x
        observation[1] = delta_y

        heading_quartenion = self.vehicle.GetChassis().GetRot()
        heading_vector = chrono.RotVecFromQuat(heading_quartenion)
        observation[2] = np.arctan2(heading_vector.y, heading_vector.x)

        goal_pos.GetNormalized()
        heading_needed = np.arctan2(delta_y, delta_x)
        observation[3] = heading_needed

        observation[4] = self.vehicle.GetSpeed()

        return observation


if __name__ == "__main__":
    env = SolarCar(render_mode="human")
    env.reset()
    while True:
        env.step([10])
        if env._terminated:
            break

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

# Gym chrono imports
# Custom imports
from .chrono_base import ChronoBaseEnv

# Standard Python imports
import numpy as np

# Gymnasium imports
import gymnasium as gym
from .track import generate_path, generate_terrain

#Array and battery
from .Battery import Battery
from ..PVSource.PVCell.PVCell import PVCellNonideal

class SolarCar(ChronoBaseEnv):
    max_speed = 97.0
    """
    Max speed of the solar car in km/h. \n
    97 km/h ~= 60 mph \n 
    """

    action_space = gym.spaces.Box(
        low=-max_speed, high=max_speed, shape=(1,), dtype=np.float64
    )
    """
    Action space for the solar car environment. \n
    shape: [desired_speed] \n
    """

    observation_space = gym.spaces.Box(low=-20, high=20, shape=(5,), dtype=np.float64)
    """
    Observation space for the solar car environment. \n
    shape: [distance along track, velocity, slope, battery %, solar radiation, future solar radiation] \n
    """

    init_pos = chrono.ChVector3d(0, 0, 0.5)
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

    step_size = 2e-3
    """
    Step size of the simulation in seconds. \n
    """

    steps_per_action = round(1 / (step_size * 10))
    """
    Number of steps per action. \n
    """

    def __init__(self, render_mode="human"):
        ChronoBaseEnv.__init__(self, render_mode)

        self.render_mode = render_mode

        veh.SetDataPath(chrono.GetChronoDataPath() + "/")

        # Terain JSON specification file
        self.rigidterrain_file = veh.GetDataFile("terrain/RigidPlane.json")

        # HMMWV specification files (vehicle, powertrain, and tire models)
        
        self.vehicle_file = veh.GetDataFile("hmmwv/vehicle/HMMWV_Vehicle.json")
        self.engine_file = veh.GetDataFile("gator/json/GATOR_EngineSimple.json") #we'll figure it out
        self.transmission_file = veh.GetDataFile(
            "hmmwv/powertrain/HMMWV_AutomaticTransmissionShafts.json"
        )
        self.tire_file = veh.GetDataFile("hmmwv/tire/HMMWV_Pac02Tire.json")

        self.vehicle = None
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
        
        # Array Variables???
        # Irradiance, Temperature, Wind, Voltage, Current, Power
        # Each panel: reference irradiance, open-circuit voltage, short-circuit current, and ideality factors
        # Efficiency of the PV module, 

        
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

    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state -> Set up for standard gym API

        Args:
            seed: Seed for the random number generator
            options: Options for the simulation (dictionary)
        """

        self.path, self.points, self.distances = generate_path()

        self.vehicle = veh.WheeledVehicle(self.vehicle_file, chrono.ChContactMethod_NSC)
        starting_point = self.path.Eval(0, 0)
        starting_point.z = 0.5
        self.vehicle.Initialize(chrono.ChCoordsysd(starting_point))
        # vehicle.GetChassis().SetFixed(True)
        self.vehicle.SetChassisVisualizationType(veh.VisualizationType_PRIMITIVES)
        self.vehicle.SetChassisRearVisualizationType(veh.VisualizationType_PRIMITIVES)
        self.vehicle.SetSuspensionVisualizationType(veh.VisualizationType_PRIMITIVES)
        self.vehicle.SetSteeringVisualizationType(veh.VisualizationType_PRIMITIVES)
        self.vehicle.SetWheelVisualizationType(veh.VisualizationType_NONE)

        # Create and initialize the powertrain system
        self.engine = veh.ReadEngineJSON(self.engine_file)
        self.transmission = veh.ReadTransmissionJSON(self.transmission_file)
        self.powertrain = veh.ChPowertrainAssembly(self.engine, self.transmission)
        self.vehicle.InitializePowertrain(self.powertrain)

        # Create and initialize the tires
        for axle in self.vehicle.GetAxles():
            for wheel in axle.GetWheels():
                tire = veh.ReadTireJSON(self.tire_file)
                self.vehicle.InitializeTire(tire, wheel, veh.VisualizationType_MESH)

        self.vehicle.GetSystem().SetCollisionSystemType(
            chrono.ChCollisionSystem.Type_BULLET
        )

        self.steps = 0

        self.array = PVCellNonideal()
        self.battery = Battery(self.step_size)


        # Create the terrain, we probably want the terrain to match the path
        # self.terrain = veh.RigidTerrain(
        #     self.vehicle.GetSystem(), self.rigidterrain_file
        # )
        # self.terrain.Initialize()

        self.terrain = generate_terrain(self.vehicle.GetSystem(), self.path)
        self.terrain.Initialize()

        # We should make the path more complex here, but this is fine for now
        self.driver = veh.ChPathFollowerDriver(
            self.vehicle,
            self.path,
            "my_path",
            0,
        )
        self.driver.GetSpeedController().SetGains(0.4, 0, 0)
        self.driver.GetSteeringController().SetGains(0.4, 0, 0)
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

        return self.observation, {}

    def step(self, action):
        """Take a step in the environment - Frequency by default is 10 Hz.

        Steps the simulation environment using the given action. The action is applied for a single step.

        Args:
            action (2 x 1 np.array): Action to be applied to the environment, consisting of throttle and steering.
        """

        time = self.vehicle.GetSystem().GetChTime()

        desired_speed = action[0] / 3.6  # Convert to m/s

        self.driver.SetDesiredSpeed(desired_speed)

        for _ in range(self.steps_per_action):
            driver_inputs = self.driver.GetInputs()
            self.driver.Synchronize(time)
            self.vehicle.Synchronize(time, driver_inputs, self.terrain)
            self.terrain.Synchronize(time)
            if self._render_setup:
                pass
                # self.vis.Synchronize(time, driver_inputs)

            self.driver.Advance(self.step_size)
            self.vehicle.Advance(self.step_size)
            self.terrain.Advance(self.step_size)
            if self._render_setup:
                pass
                # self.vis.Advance(self.step_size)

            self.vehicle.GetSystem().DoStepDynamics(self.step_size)

        self.render()

        # Get the observation
        self.observation = self.get_observation()
        # Get reward
        self.reward = self.get_reward()

        #energy array gets
        self.energy = self.array.step()
        voltage = 0
        current = self.array.getCurrent() * voltage * self.step_size
        self.battery.set_draw(current)
        self.battery.step()
        self.soc = self.battery.get_soc()

        # Check if we are done
        self.is_terminated()
        self.is_truncated()

        self.steps += 1

        return self.observation, self.reward, self._terminated, self._truncated, {}

    def render(self):
        """Render the environment"""

        # ------------------------------------------------------
        # Add visualization - only if we want to see "human" POV
        # ------------------------------------------------------
        if self.render_mode == "human":
            if self._render_setup == False:
                self.vis = veh.ChWheeledVehicleVisualSystemIrrlicht()
                self.vis = irrlicht.ChVisualSystemIrrlicht(
                    self.vehicle.GetSystem(),
                    chrono.ChVector3d(0, 0, 150),
                    chrono.ChVector3d(0, 0, 0),
                )
                self.vis.SetWindowTitle("HMMWV JSON specification")
                self.vis.SetWindowSize(1280, 1024)

                self.vis.Initialize()
                # self.vis.AddLogo(chrono.GetChronoDataFile("logo_pychrono_alpha.png"))
                self.vis.AddLightDirectional()
                # self.vis.AddSkyBox()
                # self.vis.AttachVehicle(self.vehicle)

                self._render_setup = True
            self.vis.BeginScene()
            self.vis.Render()
            self.vis.EndScene()

    def get_reward(self):
        """Get the reward for the current step

        Get the reward for the current step based on the distance to the goal, and the distance the robot has traveled.

        Returns:
            float: Reward for the current step
        """
        #scale = 200
        #Get current position
        pos = self.vehicle.GetChassis().GetPos()
        
        #Use waypoints??
        points = np.array(points)
        tracker = chrono.ChBezierCurveTracker()

        # # (closest_point - pos).Length() - euclidean length in context of ChVector class
        # distance = pos.DistanceTo(closest_point)
        # # distance = (closest_point - points[-1]).Length()
        closest_point = chrono.ChVector()
        tracker.CalcClosestPoint(pos, closest_point)

        closest_point = lambda ch_vector: np.array([closest_point.x, closest_point.y, closest_point.z])
        index = np.where(np.all(points == closest_point, axis=1))[0][0]
        distance = self.distances[index-1]

        #check if closest point is behind us
        tangent = self.path.Eval(index, 0)
        direction = (closest_point - pos).Normalized()
        dot = direction.Dot(tangent)
        if dot < 0:
            #Behind us
            distance += (closest_point - points[index]).Length()
        else:
            distance -= (closest_point - points[index]).Length()
        

        reward = distance

        # turn closest_point into a np vector to find index
        # closest_point = lambda ch_vector: np.array([closest_point.x, closest_point.y, closest_point.z])

        #index = np.where(np.all(points == closest_point, axis=1))

        # for i in range(index, len(points)-1):
        #     distance

        return reward

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

    def is_truncated(self):
        """Check if the environment is truncated

        Check if the rover has fallen off the terrain, and if so truncate and give a large penalty.
        """

        return False

        # Vehicle should not leave the lane
        # if (
        #     (self.vehicle_pos.y > self.road_length)
        #     or (self.vehicle_pos.y < -self.road_length)
        #     or (self.vehicle_pos.z < 0)
        # ):
        #     print("--------------------------------------------------------------")
        #     print("Outside of lane")
        #     print("Vehicle Position: ", self.vehicle_pos)
        #     print("--------------------------------------------------------------")
        #     self.reward -= 400
        #     self._truncated = True
        # elif self.vehicle.GetChassis().GetPos().x > 50:
        #     print("--------------------------------------------------------------")
        #     print("Reached end of lane")
        #     print("Vehicle Position: ", self.vehicle.GetChassis().GetPos())
        #     print("--------------------------------------------------------------")
        #     self.reward += 400
        #     self._truncated = True

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
        self.vehicle_pos = pos
        observation[0] = pos.x

        # For not just the priveledged of the rover
        return observation








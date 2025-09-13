---

## Appendix: Quick Bootcamp for New Members

### Python Refresher

- **Python** is a high-level, interpreted language. If you know the basics (variables, loops, functions, classes), you’re ready for this project.
- **Modules & Packages:** Code is organized into files (modules) and folders (packages). You import code using `import module` or `from package import thing`.
- **Virtual Environments:** These help you manage dependencies for each project. (See `venv` or `conda`.)

---

### Numpy: The Backbone of Scientific Python

**What is Numpy?**
- Numpy is a library for fast math and data manipulation in Python. It is used for everything from simple math to machine learning and simulation.

**Key Concept: The Array**
- A Numpy array is like a supercharged Python list, but much faster and with more features.

**Examples:**
```python
import numpy as np

# Create a 1D array
a = np.array([1, 2, 3])
print(a)  # [1 2 3]

# Create a 2D array (matrix)
b = np.array([[1, 2], [3, 4]])
print(b)  # [[1 2]
		  #  [3 4]]

# Array math
c = a * 2           # [2 4 6]
d = a + np.array([4, 5, 6])  # [5 7 9]

# Useful functions
print(np.mean(a))   # 2.0
print(np.zeros((2,3)))  # 2x3 array of zeros
print(np.random.rand(3)) # 3 random numbers between 0 and 1

# Indexing and slicing
print(a[0])  # 1
print(b[:,1]) # Second column: [2 4]
```

**Why use Numpy?**
- It’s much faster than Python lists for math.
- It’s the standard for scientific computing in Python.
- Used everywhere in ML, RL, and simulation code.

---

### Gymnasium: Standard for RL Environments

**What is Gymnasium?**
- Gymnasium (formerly OpenAI Gym) is a toolkit for building and running reinforcement learning (RL) environments.

**Key Concepts:**
- An *environment* is a world where an agent acts (like a game or simulation).
- An *agent* is a program that learns to act in the environment.

**The Environment API:**
Every Gymnasium environment has these methods:
```python
obs = env.reset()  # Start a new episode, get the first observation
obs, reward, terminated, truncated, info = env.step(action)  # Take an action
env.render()  # (Optional) Show the environment visually
```

**Spaces:**
- `Box`: Continuous values (e.g., actions between -1 and 1) 
- `Discrete`: Integer values (e.g., 0, 1, 2)

**Example:**
```python
import gymnasium as gym
env = gym.make('CartPole-v1')
obs = env.reset()
done = False
while not done:
	action = env.action_space.sample()  # Pick a random action
	obs, reward, terminated, truncated, info = env.step(action)
	done = terminated or truncated
	env.render()
env.close()
```

**What does this mean?**
- The agent interacts with the environment by taking actions.
- The environment returns observations (what the agent sees), rewards (how well it did), and info.

**Why use Gymnasium?**
- It standardizes how environments work, so you can use any RL algorithm with any environment.
- It’s used by almost all RL research and projects.

---

### Reinforcement Learning (RL) and PPO: The Basics

**What is RL?**
- RL is a type of machine learning where an agent learns to make decisions by trial and error, getting rewards or penalties.

**Key Terms:**
- *Agent*: The learner/decision maker (e.g., the solar car)
- *Environment*: The world the agent interacts with (e.g., the race track)
- *Action*: What the agent can do (e.g., accelerate, turn)
- *Observation*: What the agent sees (e.g., speed, position)
- *Reward*: A number that tells the agent how well it’s doing

**The RL Loop:**
1. The agent observes the environment.
2. It picks an action.
3. The environment updates and gives a new observation and reward.
4. Repeat!

**What is PPO?**
- PPO (Proximal Policy Optimization) is a popular RL algorithm.
- It’s used because it’s stable, works well for continuous actions, and is easy to use.

**How PPO Works (Conceptually):**
- The agent tries different actions and learns which ones lead to higher rewards.
- It updates its strategy (policy) gradually, so it doesn’t forget what it’s learned.

**Why PPO for this project?**
- Our solar car needs to make smooth, continuous decisions (like a real driver).
- PPO is good for this kind of problem.

---

### How These Fit Together

- **Numpy** is used everywhere for math, data, and simulation.
- **Gymnasium** provides the environment interface for RL agents.
- **Python** ties it all together, letting you build, extend, and run the simulation and training code.
- **RL and PPO** are the learning algorithms that make the agent smarter over time.

---
# DataAcq-PPORaceSim Onboarding Guide

> **Welcome!** This guide will help you get started with the DataAcq-PPORaceSim project, focusing on the `solar_car_v2` module and the main training script `main_2.py`. This guide assumes you know Python, but not much else. If you get stuck, ask for help!

---

## Project Overview

This repository simulates and trains reinforcement learning (RL) agents to control a solar car in a racing environment. The main components are:

- **`solar_car_v2/`**: The main simulation and modeling code for the solar car and its environment.
- **`main_2.py`**: The entry point for training and running RL agents using the PPO algorithm.

> **Ignore:** The `solar_car/` folder is deprecated and not used in current development.

---

## 1. Key Technologies Used

### 1.1. [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
An open-source library for reinforcement learning (RL) in Python. It provides implementations of popular RL algorithms, including PPO (Proximal Policy Optimization).

**Why?**
- Well-maintained, easy to use, and integrates with custom environments.

### 1.2. [OpenAI Gym](https://www.gymlibrary.dev/)
Standard interface for RL environments. Our custom environment (`SolarCar`) is registered as a Gym environment so it can be used with RL libraries.

**Why?**
- Makes it easy to swap environments and algorithms.

### 1.3. Multiprocessing (SubprocVecEnv)
Allows running multiple environment instances in parallel for faster training.

**Why?**
- Greatly speeds up RL training by collecting more experience per unit time.

### 1.4. PyTorch
The deep learning library used under the hood by Stable Baselines3.

---


## 2. Code Structure: File-by-File

### 2.1. `solar_car_v2/` (Main Simulation Package)

#### Top-Level Files

- **`__init__.py`**: Marks this folder as a Python package. (Empty, but required.)

- **`solar_car.py`**: Implements the main Gym environment for the solar car. Defines the action and observation spaces, reward logic, and integrates all subsystems (battery, weather, track, etc.). Uses [Project Chrono](https://projectchrono.org/) for physics simulation. **Start here to understand how everything connects.**

- **`Battery.py`**: Models the car's battery using [PyBaMM](https://www.pybamm.org/) and [liionpack](https://github.com/pybamm-team/liionpack) for realistic battery simulation. Handles step-by-step updates, voltage, current, and temperature tracking.

- **`track.py`**: Generates and manages the race track/path for the car. Includes logic for random path generation and bounding box calculations.

- **`getweather.py`**: Loads and processes weather data (from `weather5min.csv`) using [pvlib](https://pvlib-python.readthedocs.io/). Computes solar position and irradiance for the simulation.

- **`chrono_base.py`**: Provides a base class for environments using Project Chrono. Handles rendering, threading, and basic simulation setup. All Chrono-based environments inherit from this.

- **`scm_parameters.py`**: Contains a class for setting soil/terrain parameters (Bekker, Mohr, Janosi models) for off-road simulation. Used to configure the ground/terrain physics.

- **`weather5min.csv`**: Weather data file (CSV) used by `getweather.py`.

#### Subfolders

- **`Array/`**: Models the solar array (solar panels) on the car.
	- `Array.py`: Main logic for the solar array, including a `ThreeParamCell` class for modeling cell output.
	- `cell.py`: Interface for a single PV cell.
	- `pv.py`: Abstract base class for photovoltaic (PV) devices, with data fitting and IV curve logic.
	- `utils.py`: Utility functions for data normalization and dictionary updates.
	- `__init__.py`: Marks the folder as a package.

- **`data/`**: Contains data files and 3D models for simulation.
	- `solid_json.json`: Example JSON data for solid objects.
	- `fonts/`: Bitmap and XML font files for rendering.
	- `gator/`, `hmmwv/`: 3D models, textures, and JSON configs for different vehicle types (not always used, but useful for extending the sim).
	- `terrain/`: JSON and mesh files for different terrain types (height maps, obstacles, etc.).

- **`motor/`**: Contains a simple `Motor` class with efficiency, speed, and torque models (currently basic, but can be extended for more realism).

#### How These Pieces Fit Together

1. **`solar_car.py`** is the main environment. It creates and manages the car, battery, solar array, weather, and track.
2. **`Battery.py`** and **`Array/`** provide realistic models for energy storage and generation.
3. **`getweather.py`** supplies real-world weather/irradiance data to the array and battery.
4. **`track.py`** generates the path the car must follow.
5. **`chrono_base.py`** and **`scm_parameters.py`** handle the physics and terrain.
6. **`data/`** and **`motor/`** provide supporting files and models.

---

### 2.2. `main_2.py` (Training & Running RL Agents)

This script is the main entry point for training or running the RL agent. It:

- Uses `argparse` to parse command-line arguments for configuration (see below for options).
- Registers the custom Gym environment (`SolarCar-v0`) using the `SolarCar` class from `solar_car_v2/solar_car.py`.
- Sets up parallel environments using `SubprocVecEnv` for faster training.
- Loads an existing PPO model or creates a new one using Stable Baselines3.
- Trains the agent or runs it in play mode, saving results and models to disk.

**Key Technologies Introduced:**
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/): RL library for PPO and other algorithms.
- [OpenAI Gym](https://www.gymlibrary.dev/): Standard RL environment interface.
- [PyTorch](https://pytorch.org/): Deep learning backend for Stable Baselines3.
- Multiprocessing: For parallel environment simulation.

**If you want to change how the agent is trained or evaluated, edit this file.**

---

---

## 3. Getting Started

### 3.1. Setting Up Your Environment

We strongly recommend dev containers for consistency. If you prefer to set up manually, follow the step-by-step in .devcontainer/Dockerfile

### 3.2. Running the Simulation

To **train** a new agent:

```bash
python main_2.py -n -ec 10
```

To **continue training** an existing agent:

```bash
python main_2.py -ec 10
```

To **watch the agent play**:

```bash
python main_2.py -p -r
```

#### Common Arguments

- `-n`: Start new training (ignore previous models)
- `-ec`: Number of episodes to train
- `-p`: Play mode (run a trained agent)
- `-r`: Render the environment visually
- `-f`: Model file name (default: `ppo_car_racing_2`)
- `-env`: Number of parallel environments

See all options with:

```bash
python main_2.py --help
```

---

## 4. Technical Decisions Explained

### 4.1. Why Stable Baselines3 and PPO?
- PPO is robust and works well for continuous control problems like driving.
- Stable Baselines3 is actively maintained and easy to extend.

### 4.2. Why Use Gym Environments?
- Standardizes the interface for RL agents and environments.
- Makes it easy to use third-party RL libraries.

### 4.3. Why Multiprocessing?
- RL training is slow if you only use one environment. Running many in parallel speeds things up.

### 4.4. Why Modularize the Car/Environment?
- Each part of the car (battery, motor, etc.) is a separate module. This makes the code easier to maintain and extend.

---

## 5. Learning Resources

- [Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [OpenAI Gym Docs](https://gymnasium.farama.org)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

## 6. Contributing

1. **Branch from `main` or `test`**
2. Make your changes
3. Test your code
4. Open a Pull Request (PR)

---

## 7. FAQ

**Q: I only know Python. How do I learn RL?**
A: Start with the [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/) guide.

**Q: Where do I add new features?**
A: Most new features go in `solar_car_v2/`. For new training logic, edit `main_2.py`.

**Q: Who do I ask for help?**
A: Ask your mentor or open an issue in the repo.

---

**Welcome aboard!**

# Getting Started

## Prereqs

- [Anaconda](https://www.anaconda.com/)
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)
- [Pygame](https://github.com/pygame/pygame)

Follow the guide for your OS to install Anaconda. Then, install Gymnasium using Anaconda while in the environment you want to use Gymnasium in. Preferably a new one for this project.

```bash
# Create new env
conda create -n driver python=3.9.16
conda activate driver

conda install numpy=1.21.6
pip install stable-baselines3==2.0.0a2
pip install gymnasium==0.27.1
pip install pygame
```

You are now ready to run code

Note: We would prefer to use conda to install these dependencies, but pip seems to work better.

## Starting a new model

```python
python new_train.py
```

## Training an existing model

```python
python continue_train.py
```

## Testing a model

```python
python play_latest.py
```

# Contributing

###

The model is currently located in solar_car_2.py. The model for the car is in solar_car.

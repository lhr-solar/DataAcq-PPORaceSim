# Getting Started

## Prereqs

- [Anaconda](https://www.anaconda.com/)

Follow the guide for your OS to install Anaconda. Then, install the following prereqs using Anaconda, preferably in a new environment.

```bash
# Create new env
conda create -n driver python=3.9.16
conda activate driver

conda install numpy=1.21.5  -y
conda install tensorboard -y
conda install nomkl -y
pip install stable-baselines3[extra]==2.1.0
pip install gymnasium==0.29.1
pip install pygame==2.5.1
pip install scipy==1.10.1
pip install liionpack
pip install opencv-python==4.7.0.72
pip install splines
```

You are now ready to run code

## Start training a new model

```python
python main -n
```

## Train an existing model

```python
python main
```

## Testing a model

```python
python main -p
```

# Contributing

###

The env is currently located in solar_car_env.py. The car model is defined in solar_car.

# Appendix

## Dependencies:

- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium): Library for creating reinforced learning environments
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3): Reinforced learning algorithms
- [Pygame](https://github.com/pygame/pygame): Game library used to visualize model
- [liionpack](https://github.com/pybamm-team/liionpack): Battery model

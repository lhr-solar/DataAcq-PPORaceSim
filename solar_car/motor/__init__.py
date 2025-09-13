class Motor:
    effiency: lambda x: (-1 / (0.01 * (x)) + 100)
    speed: lambda x: -4 * x + 904
    torque: lambda x: (34 / 29 * x) - 1

    def __init__(self):
        pass

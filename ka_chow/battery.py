import numpy as np


class Battery:

    def __init__(self, max_capacity, current_capacity=None):
        if(current_capacity == None):
            self.current_capacity = max_capacity
        else:
            self.current_capacity = current_capacity

        self.max_capacity = max_capacity

    def get_capacity(self):
        return self.current_capacity

    def update(self, change):
        self.current_capacity = np.clip(
            self.current_capacity + change, 0, self.max_capacity)

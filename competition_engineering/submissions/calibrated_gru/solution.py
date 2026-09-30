"""Frozen tune-selected output calibration; GRU inputs/state unchanged."""
import numpy as np
from gru import PredictionModel as GRU

class PredictionModel:
    def __init__(self):
        self.gru = GRU()
        self.scale = np.array([0.75, 0.75], dtype=np.float32)
        self.bias = np.array([-0.1, -0.1], dtype=np.float32)

    def predict(self, data_point):
        prediction = self.gru.predict(data_point)
        if prediction is None:
            return None
        return prediction * self.scale + self.bias

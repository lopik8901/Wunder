"""Train-only ridge correction of frozen calibrated official GRU."""
from pathlib import Path
import numpy as np
from gru import PredictionModel as GRU

class PredictionModel:
    def __init__(self):
        self.gru = GRU()
        with np.load(Path(__file__).with_name("ridge.npz")) as z:
            self.mean = z["mean"]
            self.scale = z["scale"]
            self.coef = z["coef"]
            self.strengths = z["strengths"]
            self.base_scale = z["base_scale"].astype(np.float32)
            self.base_bias = z["base_bias"].astype(np.float32)

    def predict(self, data_point):
        prediction = self.gru.predict(data_point)
        if prediction is None:
            return None
        prediction = prediction * self.base_scale + self.base_bias
        normalized = np.clip((data_point.state-self.mean)/self.scale, -8, 8)
        features = np.concatenate(([1.], normalized, prediction)).astype(np.float32)
        return (prediction + self.strengths * (features @ self.coef)).astype(np.float32)

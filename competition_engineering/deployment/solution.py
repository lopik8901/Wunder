"""CPU deployment of the frozen promoted residual model."""
from pathlib import Path
import numpy as np
from gru import PredictionModel as GRU


class PredictionModel:
    def __init__(self):
        self.gru = GRU()
        with np.load(Path(__file__).with_name("model.npz")) as z:
            for name in z.files:
                setattr(self, name, z[name])
        self.weights = [getattr(self, f"weights_{k}") for k in range(3)]
        self.biases = [getattr(self, f"biases_{k}") for k in range(3)]
        self.histories = [np.zeros((3, 114), np.float32),
                          np.zeros((5, 32), np.float32),
                          np.zeros((9, 32), np.float32)]
        self.layers = [np.empty(114, np.float32), np.empty(32, np.float32),
                       np.empty(32, np.float32), np.empty(32, np.float32)]
        self.packed = [np.empty(342, np.float32), np.empty(96, np.float32),
                       np.empty(96, np.float32)]
        self.features = np.empty(115, np.float32)
        self.last_seq = None
        self.step = 0

    def predict(self, data_point):
        seq = int(data_point.seq_ix)
        if seq != self.last_seq:
            self.last_seq = seq
            self.step = 0
            for h in self.histories:
                h.fill(0)
        raw = self.gru.predict(data_point)
        self.last_raw = raw
        row = data_point.state
        if raw is None:
            base = None
            self.layers[0][112:] = 0
        else:
            base = (raw * self.base_scale + self.base_bias).astype(np.float32)
            np.subtract(row, self.ridge_mean, out=self.features[1:113])
            np.divide(self.features[1:113], self.ridge_scale, out=self.features[1:113])
            self.features[1:113].clip(-8, 8, out=self.features[1:113])
            self.features[0] = 1
            self.features[113:] = base
            base = (base + self.ridge_strengths * (self.features @ self.ridge_coef)).astype(np.float32)
            self.layers[0][112:] = base
        np.subtract(row, self.mean, out=self.layers[0][:112])
        np.divide(self.layers[0][:112], self.scale, out=self.layers[0][:112])
        self.layers[0][:112].clip(-8, 8, out=self.layers[0][:112])
        for k, (h, buf) in enumerate(zip(self.histories, self.packed)):
            dilation = 1 << k
            pos = self.step % len(h)
            h[pos] = self.layers[k]
            width = h.shape[1]
            buf[:width] = h[(pos - 2*dilation) % len(h)]
            buf[width:2*width] = h[(pos - dilation) % len(h)]
            buf[2*width:] = h[pos]
            np.dot(self.weights[k], buf, out=self.layers[k+1])
            np.add(self.layers[k+1], self.biases[k], out=self.layers[k+1])
            np.maximum(self.layers[k+1], 0, out=self.layers[k+1])
        self.step += 1
        if base is None:
            return None
        correction = self.head @ self.layers[3] + self.head_bias
        return (base + self.strengths * correction).astype(np.float32)

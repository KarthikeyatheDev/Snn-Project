from norse.torch.module.lif import LIFRecurrentCell
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class SNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        features_dim=64,
        num_layers=2,
        neurons_per_layer=64,
        lif_threshold=1.0,
        time_window=10
    ):
        super().__init__(observation_space, features_dim)
        self.time_window = time_window

        layers = []
        input_size = observation_space.shape[0]

        for i in range(num_layers):
            # Only input_size and hidden_size
            layers.append(
                LIFRecurrentCell(
                    input_size=input_size,
                    hidden_size=neurons_per_layer
                )
            )
            input_size = neurons_per_layer

        self.snn_layers = nn.ModuleList(layers)
        self.out_layer = nn.Linear(input_size, features_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Initialize states for each layer
        states = [None] * len(self.snn_layers)
        spike_rate = 0.0

        # Time window loop
        for t in range(self.time_window):
            input_t = x
            for i, layer in enumerate(self.snn_layers):
                # Call layer and let it manage its own state
                output, states[i] = layer(input_t, states[i])
                input_t = output
                spike_rate += output.mean()

        spike_rate = spike_rate / (self.time_window * len(self.snn_layers))
        self.latest_spike_rate = spike_rate.item()

        out = self.out_layer(input_t)
        return out
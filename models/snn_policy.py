import torch
import torch.nn as nn
import norse.torch as norse


class SNNPolicy(nn.Module):
    """
    Shallow Spiking Neural Network with LIF neurons
    """
    def __init__(self, input_dim, hidden_dim=64, time_window=10):
        super().__init__()
        self.time_window = time_window

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = norse.LIFRecurrentCell(hidden_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.lif2 = norse.LIFRecurrentCell(hidden_dim, hidden_dim)

    def forward(self, x):
        s1, s2 = None, None
        spike_rate = 0.0

        for _ in range(self.time_window):
            z = self.fc1(x)
            z, s1 = self.lif1(z, s1)

            z = self.fc2(z)
            z, s2 = self.lif2(z, s2)

            spike_rate += (z > 0).float().mean()

        spike_rate = spike_rate / self.time_window
        return z, spike_rate

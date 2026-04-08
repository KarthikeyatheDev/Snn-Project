# models/snn_policy.py
import torch
import torch.nn as nn


# Dummy neuron classes (you already have your own LIF/IF etc.)
class LIFNeuron(nn.Module):
    def __init__(self, size, v_threshold, tau):
        super().__init__()
        self.v_threshold = v_threshold
        self.tau = tau
        self.mem = None

    def forward(self, x):
        # dummy; replace with your actual SNN neuron
        if self.mem is None:
            self.mem = torch.zeros_like(x)
        self.mem = self.mem + x
        spike = (self.mem >= self.v_threshold).float()
        self.mem = self.mem * (1 - spike)
        return spike


class IFNeuron(nn.Module):
    def __init__(self, size, v_threshold):
        super().__init__()
        self.v_threshold = v_threshold
        self.mem = None

    def forward(self, x):
        # dummy; replace with your SNN
        if self.mem is None:
            self.mem = torch.zeros_like(x)
        self.mem = self.mem + x
        spike = (self.mem >= self.v_threshold).float()
        self.mem = self.mem * (1 - spike)
        return spike


class SNNPolicy(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,  # e.g., [64]
        output_size: int,
        n_layers: int = 1,
        neuron_type: str = "LIF",  # "LIF" or "IF"
        timesteps: int = 10,
        v_threshold: float = 1.0,
        tau: float = 10.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.n_layers = n_layers
        self.neuron_type = neuron_type
        self.timesteps = timesteps
        self.v_threshold = v_threshold
        self.tau = tau

        # Build layers (example: 1 hidden SNN layer)
        assert n_layers == 1, "shallow SNN only (1 layer)"

        # Linear + SNN neuron
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], output_size)

        if neuron_type == "LIF":
            self.neuron = LIFNeuron(hidden_sizes[0], v_threshold, tau)
        elif neuron_type == "IF":
            self.neuron = IFNeuron(hidden_sizes[0], v_threshold)
        else:
            raise ValueError(f"Unknown neuron_type: {neuron_type}")

    def forward(self, x):
        # Shape: (batch, input_size)
        batch_size = x.size(0)

        # Initialize membrane for each timestep
        if self.neuron.mem is None:
            self.neuron.mem = torch.zeros(
                batch_size, self.hidden_sizes[0], device=x.device
            )

        spike = None
        for t in range(self.timesteps):
            h = self.fc1(x)
            spike = self.neuron(h)
            if t == self.timesteps - 1:
                break

        # Final readout
        out = self.fc2(spike)
        return out

    def reset_state(self):
        # Important for SNNs: reset membrane between episodes
        if hasattr(self.neuron, "mem"):
            self.neuron.mem = None

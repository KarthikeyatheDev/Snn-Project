from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from models.snn_policy import SNNPolicy


class SNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Injects SNN into Stable-Baselines3 PPO
    """
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)

        self.snn = SNNPolicy(
            input_dim=observation_space.shape[0],
            hidden_dim=features_dim,
            time_window=10
        )

        self.latest_spike_rate = 0.0

    def forward(self, observations):
        features, spike_rate = self.snn(observations)
        self.latest_spike_rate = spike_rate
        return features

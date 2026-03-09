# train/train_pendulum.py

import gymnasium as gym
from stable_baselines3 import PPO
from train.snn_extractor import SNNFeatureExtractor


def train_snn_pendulum():

    env = gym.make("Pendulum-v1")

    policy_kwargs = dict(
        features_extractor_class=SNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128)
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        verbose=0,
        batch_size=64,
        clip_range=0.2,
        seed=42
    )

    model.learn(total_timesteps=200_000)
    return model
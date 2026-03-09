# train/train_ann.py

import gymnasium as gym
from stable_baselines3 import PPO
from models.ann_policy import ANNFeatureExtractor


def train_ann(env_name,time_steps):

    env = gym.make(env_name)

    policy_kwargs = dict(
        features_extractor_class=ANNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128)
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        clip_range=0.2,
        seed=42
    )

    model.learn(total_timesteps=time_steps)
    return model
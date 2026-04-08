# train/train_pendulum.py - NAS for Pendulum-v1 ONLY
import torch
import torch.nn.functional as F
import gymnasium as gym
import numpy as np

from models.snn_policy import SNNPolicy


def make_env():
    env = gym.make("Pendulum-v1")
    return env


def evaluate_policy(policy, env, max_eval_eps, device):
    policy.eval()
    returns = []
    for _ in range(max_eval_eps):
        obs, info = env.reset()
        total_reward = 0.0
        terminated, truncated = False, False

        while not (terminated or truncated):
            with torch.no_grad():
                obs_tensor = torch.tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                logits = policy(obs_tensor)
                action = torch.tanh(logits).cpu().numpy()[0]  # Continuous action

            obs, rew, terminated, truncated, info = env.step(action)
            total_reward += rew

        returns.append(total_reward)
    return float(np.mean(returns))


def train_snn_policy(
    config: dict,
    num_train_steps: int = 100_000,  # Pendulum needs more steps
    max_eval_eps: int = 100,
    device: str = "cpu",
):
    env = make_env()

    # Pendulum: 3-dim obs, 1-dim continuous action (tanh output)
    input_size = 3
    output_size = 1

    hidden_sizes = (
        [config["hidden_sizes"]]
        if isinstance(config["hidden_sizes"], int)
        else config["hidden_sizes"]
    )

    policy = SNNPolicy(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        n_layers=config["n_layers"],
        neuron_type=config["neuron_type"],
        timesteps=config["timesteps"],
        v_threshold=config["v_threshold"],
        tau=config["tau"],
    ).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    # Training loop (PPO-style for continuous actions)
    batch_obs, batch_acts, batch_rews = [], [], []
    obs, info = env.reset()

    total_steps = 0
    episode_return = 0.0

    while total_steps < num_train_steps:
        policy.eval()
        with torch.no_grad():
            obs_tensor = torch.tensor(
                obs, dtype=torch.float32, device=device
            ).unsqueeze(0)
            logits = policy(obs_tensor)
            action = torch.tanh(logits).cpu().numpy()[0]  # [-1,1] action

        next_obs, rew, terminated, truncated, info = env.step(action)

        batch_obs.append(obs)
        batch_acts.append(action)
        batch_rews.append(rew)

        obs = next_obs
        episode_return += rew
        total_steps += 1

        if terminated or truncated or total_steps % 2000 == 0:
            # Simple policy gradient update
            policy.train()
            optimizer.zero_grad()

            obs_batch = torch.tensor(
                np.array(batch_obs), dtype=torch.float32, device=device
            )
            logits_batch = policy(obs_batch)
            actions_batch = torch.tensor(
                np.array(batch_acts), dtype=torch.float32, device=device
            )

            # Gaussian policy loss (simple version)
            mean = torch.tanh(logits_batch)
            log_prob = -0.5 * ((actions_batch - mean) ** 2 + np.log(2 * np.pi))
            loss = -(log_prob * episode_return).mean()

            loss.backward()
            optimizer.step()

            # Reset
            batch_obs, batch_acts, batch_rews = [], [], []
            obs, info = env.reset()
            episode_return = 0.0
            policy.reset_state()

    avg_return = evaluate_policy(policy, env, max_eval_eps, device)
    env.close()
    return avg_return

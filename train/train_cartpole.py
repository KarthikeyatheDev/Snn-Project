# train/train_cartpole.py - FULLY GYMNASIUM COMPATIBLE
import torch
import torch.nn.functional as F
import gymnasium as gym  # ← FIXED: gymnasium instead of gym
import numpy as np

from models.snn_policy import SNNPolicy


def make_env(env_name):
    env = gym.make(env_name)
    return env


def evaluate_policy(policy, env, max_eval_eps, device):
    policy.eval()
    returns = []

    for _ in range(max_eval_eps):
        obs, info = env.reset()  # ← FIXED: returns (obs, info)
        if isinstance(obs, tuple):
            obs = obs[0]
        total_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            with torch.no_grad():
                obs_tensor = torch.tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                logits = policy(obs_tensor)
                action = logits.argmax(dim=1).item()

            obs, rew, terminated, truncated, info = env.step(action)  # ← FIXED: 5-tuple
            total_reward += rew

            if isinstance(obs, tuple):
                obs = obs[0]

        returns.append(total_reward)

    return float(np.mean(returns))


def train_snn_policy(
    env_name: str,
    config: dict,
    num_train_steps: int,
    max_eval_eps: int = 100,
    device: str = "cpu",
):
    """
    Train an SNN policy on a Gymnasium env (CartPole‑only).
    Returns: average episode return over max_eval_eps.
    """
    env = make_env(env_name)

    # Build SNN policy from config
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
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

    # Use a simple REINFORCE‑style optimizer (you can plug in your own)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # Simple RL loop (CartPole PPO/REINFORCE style)
    batch_obs = []
    batch_acts = []
    batch_rtgs = []
    obs, info = env.reset()  # ← FIXED: returns (obs, info)
    if isinstance(obs, tuple):
        obs = obs[0]
    total_steps = 0
    episode_return = 0.0
    episode_steps = 0

    while total_steps < num_train_steps:
        policy.eval()
        with torch.no_grad():
            obs_tensor = torch.tensor(
                obs, dtype=torch.float32, device=device
            ).unsqueeze(0)
            logits = policy(obs_tensor)
            action = logits.argmax(dim=1).item()

        next_obs, rew, terminated, truncated, info = env.step(
            action
        )  # ← FIXED: 5-tuple
        if isinstance(next_obs, tuple):
            next_obs = next_obs[0]

        batch_obs.append(obs)
        batch_acts.append(action)
        batch_rtgs.append(rew)

        obs = next_obs
        episode_return += rew
        episode_steps += 1
        total_steps += 1

        if terminated or truncated or episode_steps >= 200:
            # Simple REINFORCE: reward‑to‑go is just the episode return
            ret = episode_return
            policy.train()
            optimizer.zero_grad()

            # Convert to tensors
            obs_batch = torch.tensor(batch_obs, dtype=torch.float32, device=device)
            act_batch = torch.tensor(batch_acts, dtype=torch.long, device=device)

            # Forward
            logits = policy(obs_batch)
            log_probs = F.log_softmax(logits, dim=1)
            log_pi_a = log_probs[torch.arange(len(act_batch)), act_batch]

            # Loss = -∇ log π * return
            loss = -(log_pi_a * ret).mean()
            loss.backward()
            optimizer.step()

            # Reset
            batch_obs = []
            batch_acts = []
            batch_rtgs = []
            obs, info = env.reset()  # ← FIXED: returns (obs, info)
            if isinstance(obs, tuple):
                obs = obs[0]
            episode_return = 0.0
            episode_steps = 0

            # Reset membrane for SNN
            policy.reset_state()

    # Final evaluation
    avg_return = evaluate_policy(policy, env, max_eval_eps, device)

    env.close()
    return avg_return

# nas/objective_acrobot.py
# Fix: Ensure torch is imported for SNN NAS objective
import torch

import optuna
import torch
from train.train_acrobot import train_snn_policy
from models.snn_policy import SNNPolicy


def acrobot_nas_objective(trial: optuna.Trial) -> float:
    """
    Optuna objective for shallow SNN on Acrobot-v1.
    Returns: average episode return over evaluation.
    """
    # Sample config from NAS space
    from nas.search_space import build_snn_config

    config = build_snn_config(trial)

    # Train on Acrobot-v1 (3x more steps - harder than CartPole)
    avg_return = train_snn_policy(
        config=config,
        num_train_steps=150_000,  # 3x CartPole (Acrobot needs more training)
        max_eval_eps=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Optional hardware‑aware penalty (e.g., number of params)
    dummy_net = SNNPolicy(
        input_size=6,  # Acrobot: 6-dim observation
        hidden_sizes=(
            config["hidden_sizes"]
            if isinstance(config["hidden_sizes"], list)
            else [config["hidden_sizes"]]
        ),
        output_size=3,  # Acrobot: 3 discrete actions
        n_layers=config["n_layers"],
        neuron_type=config["neuron_type"],
        timesteps=config["timesteps"],
        v_threshold=config["v_threshold"],
        tau=config["tau"],
    )
    n_params = sum(p.numel() for p in dummy_net.parameters())
    penalty = 1e-6 * n_params

    # Optuna maximizes reward; smaller networks better under hardware constraints
    return avg_return - penalty

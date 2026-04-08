# Fix: Ensure torch is imported for SNN NAS objective
import torch

# nas/objective.py
import optuna
import torch
from train.train_cartpole import train_snn_policy
from models.snn_policy import SNNPolicy


def snn_nas_objective(trial: optuna.Trial) -> float:
    """
    Optuna objective for shallow SNN on CartPole-v1.
    Returns: average episode return over evaluation.
    """
    # Sample config from NAS space
    from nas.search_space import build_snn_config

    config = build_snn_config(trial)

    # Train on CartPole-v1 with same budget as Phase‑2
    avg_return = train_snn_policy(
        env_name="CartPole-v1",
        config=config,
        num_train_steps=10_000,  # adjust to your Phase‑2 training steps
        max_eval_eps=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Optional hardware‑aware penalty (e.g., number of params)
    dummy_net = SNNPolicy(
        input_size=4,
        hidden_sizes=(
            config["hidden_sizes"]
            if isinstance(config["hidden_sizes"], list)
            else [config["hidden_sizes"]]
        ),
        output_size=2,
        n_layers=config["n_layers"],
        neuron_type=config["neuron_type"],
        timesteps=config["timesteps"],
        v_threshold=config["v_threshold"],
        tau=config["tau"],
    )
    n_params = sum(p.numel() for p in dummy_net.parameters())
    penalty = 1e-6 * n_params

    # Optuna maximizes reward; smaller networks are better under hardware constraints
    return avg_return - penalty

# nas/search_space.py
import optuna


def build_snn_config(trial: optuna.Trial):
    """Map a trial to a config dict for shallow SNN on CartPole."""
    return {
        "n_layers": trial.suggest_categorical("n_layers", [1]),  # force shallow
        "hidden_sizes": trial.suggest_categorical("hidden_sizes", [32, 64, 128]),
        "neuron_type": trial.suggest_categorical(
            "neuron_type", ["LIF", "IF"]
        ),  # your SNN types
        "timesteps": trial.suggest_categorical("timesteps", [5, 10, 20]),
        "v_threshold": trial.suggest_float("v_threshold", 0.5, 1.5),
        "tau": trial.suggest_float("tau", 5.0, 20.0),
    }

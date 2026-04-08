# nas/objective_acrobot.py - FIXED with more training
from train.train_acrobot import train_snn_policy
from nas.search_space import build_snn_config

def acrobot_nas_objective(trial):
    config = build_snn_config(trial)
    # Acrobot needs 3x more training than CartPole (harder task)
    return train_snn_policy(
        config=config,
        num_train_steps=150_000,  # 3x more than default 50k
        max_eval_eps=100,
    )
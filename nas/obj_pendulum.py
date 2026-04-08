# nas/objective_pendulum.py  
from train.train_pendulum import train_snn_policy
from nas.search_space import build_snn_config

def pendulum_nas_objective(trial):
    config = build_snn_config(trial)
    return train_snn_policy(config)
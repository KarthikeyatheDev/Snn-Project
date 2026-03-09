# Spiking Neural Networks for Reinforcement Learning

This project explores the use of **Spiking Neural Networks (SNNs)** in reinforcement learning and compares their performance with traditional **Artificial Neural Networks (ANNs)** using **Proximal Policy Optimization (PPO)**.

The experiments are implemented using **Stable-Baselines3** and **Gymnasium** environments.

## Objective

The goal of this project is to evaluate whether SNN-based policies can perform competitively with standard ANN policies in reinforcement learning tasks.

We compare:

- **SNN-based policy networks**
- **ANN-based policy networks**

across multiple control environments.

## Environments

The following Gymnasium environments are used:

- CartPole-v1
- Acrobot-v1
- Pendulum-v1

These environments provide a mix of:

- simple control tasks
- harder sparse-reward tasks
- continuous control tasks

## Algorithms

The reinforcement learning algorithm used is:

- **PPO (Proximal Policy Optimization)** from Stable-Baselines3

Both SNN and ANN models are trained using the same PPO framework to ensure fair comparison.

## Project Structure

```text
project/
│
├── models/
│   ├── snn_policy.py        # Spiking neural network policy
│   └── ann_policy.py        # Artificial neural network policy
│
├── train/
│   ├── snn_extractor.py     # Feature extractor for SNN policy
│   ├── train_cartpole.py    # Train SNN on CartPole
│   ├── train_acrobot.py     # Train SNN on Acrobot
│   ├── train_pendulum.py    # Train SNN on Pendulum
│   └── train_ann.py         # Train ANN baseline
│
├── results/
│   ├── results.json         # Experiment metrics
│   └── *.png                # Generated plots
│
└── run.ipynb # Main experiment notebook
```

## Experiment Workflow

The experiments are run from the **Jupyter Notebook**:

```text
run.ipynb
```

Steps:

1. Train SNN model for each environment
2. Train ANN baseline model
3. Evaluate trained models
4. Compute performance metrics
5. Save results to JSON
6. Generate comparison plots

## Metrics

The following metrics are recorded:

- Mean reward
- Reward standard deviation
- Spike rate (SNN only)
- Parameter count

## Generated Outputs

All outputs are saved in the `results/` folder:

- `results.json` — experiment metrics
- `reward_comparison.png` — SNN vs ANN rewards
- `spike_rates.png` — spike activity of SNN models
- `parameter_counts.png` — model size comparison

## Hyperparameters

Common PPO parameters used in experiments:

```text
learning_rate = 1e-4 or 3e-4
n_steps = 2048
batch_size = 64
clip_range = 0.2
seed = 42
```

Training steps:

| Environment | Timesteps |
|-------------|-----------|
| CartPole    | 100,000   |
| Acrobot     | 200,000   |
| Pendulum    | 200,000   |

## Installation

Install dependencies:

```bash
pip install gymnasium
pip install stable-baselines3
pip install torch
pip install matplotlib
pip install numpy
```

## Running the Experiments

Run the Jupyter notebook:

``` text
jupyter notebook run.ipynb
```

Execute cells sequentially to train models, evaluate performance, and generate plots.

## Results

The results compare:

- SNN vs ANN performance
- spike activity in SNN models
- model complexity

All metrics and plots are saved automatically in the `results/` folder.

## License

This project is for research and educational purposes.

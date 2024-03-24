# DeepRL

## DeepRL projects for ACS MPhil 2024

### Part 1: Implement DQN with equinox + BO for hyperparams

1) set up conda environment + log into Weights & Biases if want to track experiments

```
conda env create -f DQN_environment.yaml
conda activate DQN_equinox
wandb login
```

2) Run experiments in DQN_equinox directory. You can choose whether to trigger the BO or not by adding --enable_bo, and whether to track the experiments on Weights and Biases by --track.

```
python dqn_equinox_BO_rc.py --enable_bo --track
```

### Part 2: Implement MORel and evaluate dynamics model uncertainties

1) Set up conda environment + log into Weights & Biases if want to track experiments

```
conda env create -f morel.yaml
conda activate Morel
wandb login
```

2) Run main training loop, include --wandb to log on W&B.

* Running the training will save a series of pickle files in the *save_files* folder.

```
python train_rc.py --wandb
```

3) To view the dynamics uncertainties on the trained dynamics models and policy:

* Go to ```Uncertainty_checks.ipynb```, activate the Morel environment and run all from start.

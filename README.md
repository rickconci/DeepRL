# DeepRL

## DeepRL projects for ACS MPhil 2024

### Part 1: Implement DQN with equinox + BO for hyperparams

1) set up conda environment

```
conda env create -f DQN_environment.yml
conda activate DQN_equinox
```

2) Log into Weights & Biases if want to track experiments

```
wandb login
```

3) Move to the DQN_equinox directory. You can choose whether to trigger the BO or not by adding --enable_bo, and whether to track the experiments on Weights and Biases by --track.

```
python dqn_equinox_BO_rc.py --enable_bo --track
```

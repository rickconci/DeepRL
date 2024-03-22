

import argparse
from dataclasses import dataclass
import os

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "DeepRL_DQN_Lunar_BO"
    """the wandb's project name"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    enable_bo: bool = False
    ''' whether to enable BO hyperparam optimisation '''

    
    # Algorithm specific arguments
    env_id: str = "LunarLander-v2"
    """the id of the environment"""
    total_timesteps: int = 1500000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 100000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1 #0.005 #1
    """the target network update rate"""
    target_network_frequency: int = 1000 #500
    """the timesteps it takes to update the target network"""
    batch_size: int = 64 #128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Run experiment with custom parameters.")
    parser.add_argument("--exp_name", type=str, default=Args.exp_name, help="The name of this experiment")
    parser.add_argument("--seed", type=int, default=Args.seed, help="Seed of the experiment")
    parser.add_argument("--track", action='store_true', help="If toggled, track experiment with Weights and Biases")
    parser.add_argument("--wandb_project_name", type=str, default=Args.wandb_project_name, help="The Weights and Biases project name")
    parser.add_argument("--capture_video", action='store_true', help="Whether to capture videos of the agent performances")
    parser.add_argument("--save_model", action='store_true', help="Whether to save the model into the `runs/{run_name}` folder")
    parser.add_argument("--enable_bo", action="store_true", help="Enable Bayesian Optimization")

    parser.add_argument("--env_id", type=str, default=Args.env_id, help="The ID of the environment")
    parser.add_argument("--total_timesteps", type=int, default=Args.total_timesteps, help="Total timesteps of the experiments")
    parser.add_argument("--learning_rate", type=float, default=Args.learning_rate, help="The learning rate of the optimizer")
    parser.add_argument("--num_envs", type=int, default=Args.num_envs, help="The number of parallel game environments")
    parser.add_argument("--buffer_size", type=int, default=Args.buffer_size, help="The replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=Args.gamma, help="The discount factor gamma")
    parser.add_argument("--tau", type=float, default=Args.tau, help="The target network update rate")
    parser.add_argument("--target_network_frequency", type=int, default=Args.target_network_frequency, help="The timesteps it takes to update the target network")
    parser.add_argument("--batch_size", type=int, default=Args.batch_size, help="The batch size of sample from the reply memory")
    parser.add_argument("--start_e", type=float, default=Args.start_e, help="The starting epsilon for exploration")
    parser.add_argument("--end_e", type=float, default=Args.end_e, help="The ending epsilon for exploration")
    parser.add_argument("--exploration_fraction", type=float, default=Args.exploration_fraction, help="The fraction of `total_timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning_starts", type=int, default=Args.learning_starts, help="Timestep to start learning")
    parser.add_argument("--train_frequency", type=int, default=Args.train_frequency, help="The frequency of training")

    # Parse arguments
    args = parser.parse_args()
    
    return Args(**vars(args))

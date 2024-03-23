# morel imports
from numpy.lib.npyio import save
from morel.models.Dynamics_rc import DynamicsEnsemble
from morel.models.Policy_rc import PPO2
from morel.fake_env_rc import FakeEnv

import numpy as np
from tqdm import tqdm
import os
import random 
import wandb 
import pickle 

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

from .utils_rc import SEED
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


class Morel():
    def __init__(self, obs_dim, action_dim, args):
        self.args = args
        self.dynamics = DynamicsEnsemble(obs_dim + action_dim, obs_dim + 1, threshold=1.0, use_wandb=self.args.wandb)
        self.policy = PPO2(obs_dim, action_dim, use_wandb=self.args.wandb)
        self.normalise = args.normalise_data


    def train(self, dataloader, dynamics_data ):

        self.dynamics_data = dynamics_data
        #self.dynamics_data.change_video_path('Dynamics_model')

        if self.args.wandb:
            wandb.init(project=self.args.project_name, name="Dynamics Training", config=self.args)

        print("---------------- Beginning Dynamics Training ----------------")
        self.dynamics.train(dataloader, epochs=10)

        print("---------------- Ending Dynamics Training ----------------")
        if self.args.wandb:
            wandb.finish()

        #self.dynamics_data.change_video_path('Policy_training')
        env = FakeEnv(self.dynamics,
                            self.dynamics_data.observation_mean,
                            self.dynamics_data.observation_std,
                            self.dynamics_data.action_mean,
                            self.dynamics_data.action_std,
                            self.dynamics_data.delta_mean,
                            self.dynamics_data.delta_std,
                            self.dynamics_data.reward_mean,
                            self.dynamics_data.reward_std,
                            self.dynamics_data.initial_obs_mean,
                            self.dynamics_data.initial_obs_std,
                            self.dynamics_data.source_observation,
                            uncertain_penalty=-50.0,
                            normalise_data=self.normalise)

        if self.args.wandb:
            wandb.init(project=self.args.project_name, name="Policy Training", config=self.args)

        print("---------------- Beginning Policy Training ----------------")
        self.policy.train(env)
        print("---------------- Ending Policy Training ----------------")
        if self.args.wandb:
            wandb.finish()

        print("---------------- Successfully Completed Training ----------------")

    def eval(self, env): 
        if self.args.wandb:
            wandb.init(project=self.args.project_name, name="Policy Evaluation", config=self.args, monitor_gym=True)

        print("---------------- Beginning Policy Evaluation ----------------")
        total_rewards = []
        eval_actions_list = []
        eval_obs_list = []
        eval_reward_list = []

        for i in tqdm(range(50)):
            _, eval_obs,_, _, eval_actions, _, _, info, _, eval_reward = self.policy.generate_experience(env, 1024, 0.95, 0.99, eval_render = True)
            total_rewards.extend(info["episode_rewards"])
            eval_obs_list.append(eval_obs)
            eval_actions_list.append(eval_actions)
            eval_reward_list.append(eval_reward)
            
            if self.args.wandb:
                wandb.log({'Metrics/eval_episode_reward': sum(info["episode_rewards"]) / len(info["episode_rewards"])})

        with open('eval_obs.pkl', 'wb') as file:
            print("Saving eval_obs!!")
            pickle.dump(eval_obs_list, file)
            
        with open('eval_actions.pkl', 'wb') as file:
            print("Saving eval_actions!!")
            pickle.dump(eval_actions_list, file)

        with open('eval_reward_locations.pkl', 'wb') as file:
                print("Saving eval_reward_locations!!")
                pickle.dump(eval_reward_list, file)

        print("Final evaluation reward: {}".format(sum(total_rewards)/len(total_rewards)))

        print("---------------- Ending Policy Evaluation ----------------")
        if self.args.wandb:
            wandb.finish()

    def save(self, save_dir):
        if(not os.path.isdir(save_dir)):
            os.mkdir(save_dir)

        self.policy.save(save_dir)
        self.dynamics.save(save_dir)

    def load(self, load_dir):
        self.policy.load(load_dir)
        self.dynamics.load(load_dir)




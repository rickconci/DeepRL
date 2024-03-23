import argparse
import json
import subprocess
import numpy as np
from tqdm import tqdm
import os
import glob
import tarfile
import wandb  # Import wandb

# dataset imports
import gymnasium
from gymnasium.wrappers import RecordVideo

import pandas as pd
import random 

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

from morel.utils_rc import SEED, Mazes, find_zero_positions

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


from morel.morel_rc import Morel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#PointMaze_UMazeDense-v3
#PointMaze_Large-v3
#PointMaze_UMaze-v3



class Offline_data_creator(Dataset):
    def __init__(self, env_name='PointMaze_UMaze-v3', n_pos_reward_traj=4, max_episode_steps = 1000,  normalise_data = True, video_path = '', capture_video= False, seed = None):
        self.device = device  
        self.seed = seed
        self.n_pos_reward_traj = n_pos_reward_traj
        self.capture_video = capture_video
        self.video_path = video_path
        self.env_name = env_name
        self.max_episode_steps = max_episode_steps
        self.normalise_data = normalise_data
        

        render_mode = "rgb_array" if capture_video else "human"
        self.base_env = gymnasium.make(env_name, 
                                       render_mode=render_mode, 
                                       max_episode_steps=max_episode_steps,
                                       continuing_task = False,
                                       reset_target = False)
        
        
        self.env = self.wrap_env(video_path)
    
        if self.seed is not None:
            self.env.action_space.seed(self.seed)

        
        zero_positions = find_zero_positions(Mazes[env_name])
        self.fixed_goal_cell = np.array(random.choice(zero_positions))
        self.remaining_positions = np.array([pos for pos in zero_positions if not np.array_equal(pos, self.fixed_goal_cell)])
    
    def wrap_env(self, video_path):
        if self.capture_video:
            return RecordVideo(self.base_env, video_folder=f"videos/{video_path}")
        else:
            return self.base_env
        
    def create_dataset(self):
        
        states, actions, next_states, terminated_flags, rewards = [], [], [], [], []
        pos_rewards_count = 0

        while pos_rewards_count < self.n_pos_reward_traj :

            options = {
                'goal_cell': self.fixed_goal_cell,
                'reset_cell': random.choice(self.remaining_positions)
            }
            print('goal_cell', self.fixed_goal_cell)
            print('reset cell', options['reset_cell'])

            state = self.env.reset(options=options)[0]['observation']

            print('state', state)

            for _ in range(self.max_episode_steps):
                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_state = next_state['observation']

                if self.capture_video == False:
                    self.env.render()
            
                states.append(state)
                actions.append(action)
                next_states.append(next_state)
                terminated_flags.append(terminated)
                rewards.append(reward)
            
                state = next_state

                if reward ==1 :
                    print('reward!!')
                    pos_rewards_count += reward
                    break
                if terminated or truncated:
                    break  
            
        print(len(states))
        self.states = np.array(states)
        self.actions = np.array(actions)
        self.next_states = np.array(next_states)
        self.rewards = np.array(rewards)
        self.terminated_flags = np.array(terminated_flags)

        self.df = self.clean_data()
        self.calculate_statistics(self.df)
    
    

    def clean_data(self):
        # Initialize lists to hold the unpacked data
    
        # Create a DataFrame
        data = {
            'state_x_coord': self.states[:, 0],
            'state_y_coord': self.states[:, 1],
            'state_x_vel': self.states[:, 2],
            'state_y_vel': self.states[:, 3],
            'action_x': self.actions[:, 0],  
            'action_y': self.actions[:, 1],  
            'next_state_x_coord': self.next_states[:, 0],
            'next_state_y_coord': self.next_states[:, 1],
            'next_state_x_vel': self.next_states[:, 2],
            'next_state_y_vel': self.next_states[:, 3],
            'reward': self.rewards,
            'terminated_flags': self.terminated_flags
        }

        df = pd.DataFrame(data)

        df['delta_x_coord'] = df['next_state_x_coord'] - df['state_x_coord']
        df['delta_y_coord'] = df['next_state_y_coord'] - df['state_y_coord']
        df['delta_x_vel'] = df['next_state_x_vel'] - df['state_x_vel']
        df['delta_y_vel'] = df['next_state_y_vel'] - df['state_y_vel']

        df.to_pickle('Dataset_df.pkl')

        return df

    
    def calculate_statistics(self, df):
        
        # Normalize data
        observations =  df[['state_x_coord', 'state_y_coord', 'state_x_vel','state_y_vel'] ]
        self.observation_mean =observations.mean()
        self.observation_std = observations.std()
        
        actions = df[['action_x', 'action_y']]
        self.action_mean = actions.mean()
        self.action_std =  actions.std()
        
        deltas =  df[['delta_x_coord', 'delta_y_coord', 'delta_x_vel','delta_y_vel'] ]
        self.delta_mean = deltas.mean()
        self.delta_std =  deltas.std()
        
        self.reward_mean = df['reward'].mean()
        self.reward_std = df['reward'].std()

        if self.normalise_data:
            self.source_action = ((actions - self.action_mean)/ self.action_std).values
            self.source_observation = ((observations - self.observation_mean)/self.observation_std).values
            self.target_delta = ((deltas - self.delta_mean)/self.delta_std).values
            self.target_reward = ((df['reward'] - self.reward_mean)/self.reward_std).values
        else:
            self.source_action = actions.values
            self.source_observation = observations.values
            self.target_delta = deltas.values
            self.target_reward = df['reward'].values


        # Initial observations
        self.done_indices = df['terminated_flags'].values
        self.initial_indices = np.roll(self.done_indices, 1)
        self.initial_indices[0] = True
    
        # Calculate distribution parameters for initial states
        self.initial_obs = self.source_observation[self.initial_indices]
        self.initial_obs_mean = self.initial_obs.mean(axis = 0)
        self.initial_obs_std = self.initial_obs.std(axis = 0)

        # Remove transitions from terminal to initial states
        self.source_action = np.delete(self.source_action, self.done_indices, axis = 0)
        self.source_observation = np.delete(self.source_observation, self.done_indices, axis = 0)
        self.target_delta = np.delete(self.target_delta, self.done_indices, axis = 0)
        self.target_reward = np.delete(self.target_reward, self.done_indices, axis = 0)


    def change_video_path(self, video_path):
        self.video_path = video_path
        self.env = self.wrap_env(video_path)


    def __len__(self):
            # Directly use the DataFrame's length
            return len(self.source_observation)

    def __getitem__(self, idx):
        feed = torch.FloatTensor(np.concatenate([self.source_observation[idx], self.source_action[idx]])).to(self.device)
        target = torch.FloatTensor(np.concatenate([self.target_delta[idx], self.target_reward[idx:idx+1]])).to(self.device)

        return feed, target
    
    def get_all_items(self):
        all_feeds = torch.FloatTensor(np.concatenate([self.source_observation, self.source_action], axis=1)).to(self.device)
        rewards_reshaped = self.target_reward.reshape(-1, 1)
        all_targets = torch.FloatTensor(np.concatenate([self.target_delta, rewards_reshaped], axis=1)).to(self.device)
        
        return all_feeds, all_targets


#def upload_assets(comet_experiment, log_dir):
    # This function might need to be adapted for wandb if you want to upload assets
    # For wandb, you can use wandb.save() to save files

def main(args):
    if(not args.no_log):
        # Create necessary directories
        if(not os.path.isdir(args.log_dir)):
            os.mkdir(args.log_dir)

        # Create log_dir for run
        run_log_dir = os.path.join(args.log_dir,args.exp_name)
        if(os.path.isdir(run_log_dir)):
            cur_count = len(glob.glob(run_log_dir + "_*"))
            run_log_dir = run_log_dir + "_" + str(cur_count)
        os.mkdir(run_log_dir)


    # Instantiate dataset   
    dynamics_data = Offline_data_creator(env_name= args.env_name, 
                                          n_pos_reward_traj=args.n_pos_reward_traj, 
                                          max_episode_steps = args.max_episode_steps, 
                                          normalise_data = args.normalise_data,
                                          video_path='initial_random', 
                                          capture_video=False, 
                                          seed=SEED)
    dynamics_data.create_dataset()


    dataloader = DataLoader(dynamics_data, batch_size=128, shuffle = True)

    num_observations = dynamics_data.env.observation_space['observation'].shape[0]
    num_actions = dynamics_data.env.action_space.shape[0]
    agent = Morel(num_observations, num_actions, args) 

    agent.train(dataloader, dynamics_data)

    if(not args.no_log):
        agent.save(os.path.join(run_log_dir, "models"))
    
    dynamics_data.change_video_path('Policy_eval')
    agent.eval(dynamics_data.env)

    dynamics_data.env.close()

def eval(args):
    
    dynamics_data = Offline_data_creator(env_name= args.env_name, 
                                         n_pos_reward_traj=args.n_pos_reward_traj, 
                                          max_episode_steps = args.max_episode_steps, 
                                          normalise_data = args.normalise_data,
                                          video_path='initial_random', 
                                          capture_video=False, 
                                          seed=SEED)
    dynamics_data.change_video_path('Policy_eval')

    num_observations = dynamics_data.env.observation_space['observation'].shape[0]
    num_actions = dynamics_data.env.action_space.shape[0]
    agent = Morel(num_observations, num_actions, args)
    agent.load(args.load_dir ) 

    agent.eval(dynamics_data.env)
    dynamics_data.env.close()

def create_initial_data(args):

    dynamics_data = Offline_data_creator(env_name= args.env_name, 
                                         n_pos_reward_traj=args.n_pos_reward_traj, 
                                         max_episode_steps = args.max_episode_steps, 
                                         normalise_data = args.normalise_data,
                                         video_path='initial_random', 
                                         capture_video=False, 
                                         seed=SEED)
    dynamics_data.create_dataset()
    dynamics_data.env.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training')

    parser.add_argument('--env_name', type=str, default = 'PointMaze_UMaze-v3')
    parser.add_argument('--n_pos_reward_traj', type=int, default = 10)
    parser.add_argument('--max_episode_steps', type=int, default = 5000)
    parser.add_argument('--normalise_data', type=bool, default=False)

    parser.add_argument('--log_dir', type=str, default='./results')
    parser.add_argument('--wandb', action='store_true')  
    parser.add_argument('--project_name', type=str, default='Morel_tests')
    parser.add_argument('--exp_name', type=str, default='exp_test')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--load_dir', type=str , default = '/Users/riccardoconci/Local_documents/ACS submissions/CausalRL/DeepRL_miniproject/MOReL/results/exp_test_89/models')
    

    args = parser.parse_args()
    #main(args)
    eval(args)
    #create_initial_data(args)

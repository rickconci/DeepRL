import numpy as np
import random

# torch imports
import torch
import os 

from .utils_rc import SEED
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FakeEnv:
    def __init__(self, dynamics_model,
                        obs_mean,
                        obs_std,
                        action_mean,
                        action_std,
                        delta_mean,
                        delta_std,
                        reward_mean,
                        reward_std,
                        initial_obs_mean,
                        initial_obs_std,
                        start_states,
                        timeout_steps = 300,
                        uncertain_penalty = -100, 
                        normalise_data = True):
        self.dynamics_model = dynamics_model
        self.normalise_data = normalise_data

        self.uncertain_penalty = uncertain_penalty
        self.start_states = start_states
        #print('start_states', start_states )

        self.input_dim = self.dynamics_model.input_dim
        self.output_dim = self.dynamics_model.output_dim

        self.device = device
        #print('obs_mean', type(obs_mean))
        # Save data transform parameters
        self.obs_mean = torch.tensor(obs_mean.to_numpy()).float().to(self.device)
        self.obs_std = torch.tensor(obs_std.to_numpy()).float().to(self.device)
        self.action_mean = torch.tensor(action_mean.to_numpy()).float().to(self.device)
        self.action_std = torch.tensor(action_std.to_numpy()).float().to(self.device)
        self.delta_mean = torch.tensor(delta_mean.to_numpy()).float().to(self.device)
        self.delta_std = torch.tensor(delta_std.to_numpy()).float().to(self.device)
        self.reward_mean = torch.Tensor([reward_mean]).float().to(self.device)
        self.reward_std = torch.Tensor([reward_std]).float().to(self.device)

        self.initial_obs_mean = torch.Tensor(initial_obs_mean).float().to(self.device)
        self.initial_obs_std = torch.Tensor(initial_obs_std).float().to(self.device)

        self.timeout_steps = timeout_steps

        self.state = None

    def reset(self):
        idx = np.random.choice(self.start_states.shape[0])
        next_obs = torch.tensor(self.start_states[idx]).float().to(self.device)

        if self.normalise_data:
            self.state = (next_obs - self.obs_mean)/self.obs_std
        else:
            self.state = next_obs
        # print("reset!")
        # self.state = torch.normal(self.obs_mean, self.obs_std)
        # next_obs = self.obs_mean + self.obs_std*self.state
        self.steps_elapsed = 0
        #print('env_reset: output', next_obs)
        return next_obs

    def step(self, action_unnormalized, obs = None):
        if self.normalise_data:
            action = (action_unnormalized - self.action_mean)/self.action_std

        if obs is not None:
            if self.normalise_data:
                self.state = (torch.tensor(obs).float().to(self.device) - self.obs_mean)/self.obs_std
            else:
                self.state = torch.tensor(obs).float().to(self.device)
            # self.state = torch.unsqueeze(self.state,0)
            print(self.state.shape)
            print(action.shape)
        
        #print('state',self.state.shape )
        #print('action_unnormalized', action_unnormalized.shape)

        if self.normalise_data:
            input = torch.cat([self.state, action],0)
        else:
            input = torch.cat([self.state, action_unnormalized],0)

        #print('predict_input', input.shape)
        predictions = self.dynamics_model.predict(input)
        #print('predictions', predictions.shape)

        deltas = predictions[:,0:self.output_dim-1]

        rewards = predictions[:,-1]

        # Calculate next state
        if self.normalise_data:
            deltas_unnormalized = self.delta_std*torch.mean(deltas,0) + self.delta_mean
            state_unnormalized = self.obs_std*self.state + self.obs_mean
            next_obs = deltas_unnormalized + state_unnormalized
            self.state = (next_obs - self.obs_mean)/self.obs_std
        else:
            #print('deltas mean', torch.mean(deltas,0).shape)
            #print('deltas', deltas.shape)
            #print('state', self.state.shape)
            next_obs = torch.mean(deltas,0) + self.state
            self.state = next_obs


        uncertain = self.dynamics_model.usad(predictions.cpu().numpy())
        #print('uncertainty', uncertain)

        if self.normalise_data:
            reward_out = self.reward_std*torch.mean(rewards) + self.reward_mean
        else:
            reward_out = torch.mean(rewards).unsqueeze(0)

        #print('reward', reward_out.shape)
        if(uncertain):
            reward_out[0] = self.uncertain_penalty
            #print('UNCERTAIN!!', uncertain)
        reward_out = torch.squeeze(reward_out)
        #print('reward', reward_out.shape)

        self.steps_elapsed += 1
        # print("reward {}\tuncertain{}".format(reward_out, uncertain))
        # input()

        #print('next_obs', next_obs)

        return next_obs, reward_out, (uncertain or self.steps_elapsed > self.timeout_steps), {"HALT" : uncertain}
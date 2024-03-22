import os
import random
import time
import pickle
import numpy as np
import pandas as pd
from typing import Tuple
import sys

import wandb
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import equinox as eqx
import optax

import GPyOpt
from GPyOpt.methods import BayesianOptimization


import stable_baselines3 as sb3
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


from env_models import make_env, QNetwork, make_network, linear_schedule
from config import Args, parse_args




global_args = parse_args()


def objective_function_with_args(global_args):

    def objective_function(params):
        param_values = params[0]
        
        print("RUN objective function!!")
        bo_args = Args(
            learning_rate=param_values[0],
            buffer_size=int(param_values[1]),
            batch_size=int(param_values[2]),
            tau=param_values[3],
            exploration_fraction=param_values[4],
            track=global_args.track
        )

        if global_args.track:
            unique_id = f"bo_run_{int(time.time())}"
            wandb.init(project="DeepRL_DQN_Lunar_BO_global_step", name=unique_id, config={
                "learning_rate": param_values[0],
                "buffer_size": int(param_values[1]),
                "batch_size": int(param_values[2]),
                "tau": param_values[3],
                "exploration_fraction": param_values[4]
            })

        final_global_step = run_experiment(bo_args)

        if global_args.track:
            wandb.log({"final_global_step": final_global_step})
            wandb.finish()
        
        return final_global_step
    return objective_function

def run_experiment(args: Args):


    learning_rate = args.learning_rate
    buffer_size = args.buffer_size
    batch_size = args.batch_size
    tau = args.tau
    exploration_fraction = args.exploration_fraction

    print('learning_rate', learning_rate)
    print('buffer_size', buffer_size)
    print('tau', tau)
    print('exploration_fraction', exploration_fraction)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )


    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, q_key = jax.random.split(key, 2)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    #assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    obs, _ = envs.reset(seed=args.seed)
    print('obs', obs.shape)
    #q_network = make_network(action_dim=envs.single_action_space.n, key=q_key, input_shape=obs.shape)
    #print('obs', obs.shape)
    q_network = make_network(action_dim=envs.single_action_space.n, key=q_key, input_shape=obs.shape[1:])


    optimizer = optax.adam(learning_rate=args.learning_rate)
    optimizer_state = optimizer.init(q_network)

    dummy_obs = jnp.ones((args.batch_size,) + envs.single_observation_space.shape)
    #print('dummy_obs', dummy_obs[0:1].shape)
    
    dummy_q_values = q_network(dummy_obs)
    #print(f"Dummy Q-values shape: {dummy_q_values.shape}")
    #print('envs.single_action_space.n', envs.single_action_space.n)
    assert dummy_q_values.shape == (args.batch_size, envs.single_action_space.n), "Incorrect Q-values shape"


    # Target network
    #target_network = make_network(action_dim=envs.single_action_
    # space.n, key=q_key, input_shape=obs.shape)
    target_network = make_network(action_dim=envs.single_action_space.n, key=q_key, input_shape=obs.shape[1:])
    target_network = tree_map(lambda x, y: y, target_network, q_network)


    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        handle_timeout_termination=False,
    )
    #print(f"Replay buffer size: {rb.size()}")
    assert rb.size() <= args.buffer_size, "Replay buffer size exceeded"


    
    @jax.jit
    def update(model, target_model, optimizer_state, observations, actions, next_observations, rewards, dones):
        # Calculate next Q-values using the target network
        q_next_target = target_model(next_observations)  # (batch_size, num_actions)
        q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
        next_q_value = rewards + (1 - dones) * args.gamma * q_next_target

        # Loss function uses the current Q-network
        def mse_loss(model, observations, actions, next_observations, rewards, dones):
            # Compute the Q-values for the next observations using the target network
            q_next_target = target_model(next_observations)  # (batch_size, num_actions)
            q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
            # Compute the target Q-values
            next_q_value = rewards + (1 - dones) * args.gamma * q_next_target
            # Compute the Q-values for the current observations
            q_pred = model(observations)  # (batch_size, num_actions)
            q_pred = q_pred[jnp.arange(q_pred.shape[0]), actions.squeeze()]  # (batch_size,)
            # Return the loss and Q-values
            loss = ((q_pred - next_q_value) ** 2).mean()
            return loss


        # Calculate gradients and perform the optimization step
        loss_value = mse_loss(model, observations, actions, next_observations, rewards, dones)
        grads = jax.grad(mse_loss, argnums=0)(model, observations, actions, next_observations, rewards, dones)
        
        #print(f"Loss: {loss_value}, Gradient shapes: {[g.shape for g in jax.tree_leaves(grads)]}")
        assert isinstance(loss_value, jnp.ndarray) and loss_value.shape == (), "Loss should be a scalar"
        
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        model = eqx.apply_updates(model, updates)

        q_pred = model(observations)
        q_pred = q_pred[jnp.arange(q_pred.shape[0]), actions.squeeze()]

        return loss_value, q_pred, model, optimizer_state

    start_time = time.time()

    # Start the game
    obs, _ = envs.reset(seed=args.seed)
    #print(f"Initial observation shape: {obs.shape}")  # It should match the environment's observation space
    #print("envs.single_observation_space.shape", envs.single_observation_space.shape)
    assert obs.shape[1:] == envs.single_observation_space.shape, "Mismatch in observation shape"


    episodic_returns_list = []
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(obs)
            actions = q_values.argmax(axis=-1)
            actions = jax.device_get(actions)
        
        #print(f"Actions: {actions}")
        assert actions.shape == (envs.num_envs,), "Incorrect actions shape"
        assert (actions >= 0).all() and (actions < envs.single_action_space.n).all(), "Actions out of bounds"


        # Execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)


        # Record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                    value = info['episode']['r'][0]
                    episodic_returns_list.append(value)
                    #print('episodic_returns_list', episodic_returns_list)

        if len(episodic_returns_list) >= 200:
            successful_returns = np.sum(np.array(episodic_returns_list[-200:]) >= 200)
            if successful_returns >= 180:
                print("Termination condition met.")
                average_returns = np.mean(episodic_returns_list)
                envs.close()
                writer.close()
                return global_step
            

            elif np.mean(episodic_returns_list[-200:]) >= 180:
                print("Termination condition met.")
                average_returns = np.mean(episodic_returns_list)
                envs.close()
                writer.close()
                return global_step


        # Save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # important!
        obs = next_obs

        # Training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)

                #print(f"Sampled data shapes - observations: {data.observations.shape}, actions: {data.actions.shape}")
                assert data.observations.shape == (args.batch_size,) + envs.single_observation_space.shape, "Incorrect shape of sampled observations"
                assert data.actions.shape == (args.batch_size, 1), "Incorrect shape of sampled actions"

                # perform a gradient-descent step
                loss, old_val, q_network, optimizer_state = update(
                    q_network,
                    target_network,
                    optimizer_state,
                    data.observations.numpy(),
                    data.actions.numpy(),
                    data.next_observations.numpy(),
                    data.rewards.flatten().numpy(),
                    data.dones.flatten().numpy(),
                )

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", jax.device_get(loss), global_step)
                    writer.add_scalar("losses/q_values", jax.device_get(old_val).mean(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # update target network
            if global_step % args.target_network_frequency == 0:
                target_network = jax.tree_map(
                    lambda x, y: args.tau * x + (1 - args.tau) * y,
                    q_network,
                    target_network,
                )
                print("Updated target network parameters.")



    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.equinox_model"
        with open(model_path, 'wb') as f:
            pickle.dump(q_network, f)

        print(f"model saved to {model_path}")

    average_returns = np.mean(episodic_returns_list)
    print('episodic_returns_list', episodic_returns_list)

    envs.close()
    writer.close()
    
    return global_step

if __name__ == "__main__":
    sys.stdout = open('bo_output_2.txt', 'w')
    global_args = parse_args()  

    if global_args.enable_bo:
        'BO enabled!!'
        domain = [
            {"name": "learning_rate", "type": "continuous", "domain": (1e-5, 1e-3)},
            {"name": "buffer_size", "type": "discrete", "domain": np.array([50000, 100000, 150000])},
            {"name": "batch_size", "type": "discrete", "domain": np.array([32, 64, 128])},
            {"name": "tau", "type": "continuous", "domain": (0.001, 1)},
            {"name": "exploration_fraction", "type": "continuous", "domain": (0.1, 0.8)}
        ]
        
        objective_function = objective_function_with_args(global_args)

        
        print("BEGIN OPTIMISATION test 1!!")
        optimizer = BayesianOptimization(
            f=objective_function, 
            domain=domain,
            acquisition_type="EI",
            exact_feval=True,
            initial_design_numdata=15,
            maximize=False 
        )
        
        #optimizer.X = X
        #optimizer.Y = Y
        print("BEGIN OPTIMISATION test 2!!")

        optimizer.run_optimization(max_iter=15)
    
        print("BEGIN OPTIMISATION test 3 !!")

        gp_model = optimizer.model.model
        kernel = gp_model.kern
        print("Kernel used by the GP model:", kernel)
        print("Kernel parameters:")
        print("Variance:", kernel.variance.values)
        print("Lengthscale:", kernel.lengthscale.values)

        
        for i, (params, value) in enumerate(zip(optimizer.X, optimizer.Y)):
            print(f"Iteration {i+1}: Params={params}, Global_step={value[0]}")

        print("Optimal parameters found:\n", optimizer.x_opt)
        print("Max  average return achieved:\n", optimizer.fx_opt)
    
    else:
        run_experiment(global_args)
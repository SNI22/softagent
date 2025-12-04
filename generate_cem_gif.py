#!/usr/bin/env python

import argparse
import os
import os.path as osp
import pickle

from cem.visualize_cem import cem_make_gif
from envs.env import Env
from softgym.registered_env import env_arg_dict

def generate_gif(pkl_path, env_name, output_dir):
    # Load the pickle file
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    initial_states = data['initial_states']
    action_trajs = data['action_trajs']
    configs = data['configs']

    # Construct env kwargs from env registry
    env_kwargs = env_arg_dict[env_name]
    env_kwargs['render'] = True

    # Build the rendering environment
    env_render = Env(
        env=env_name,
        symbolic=False,
        seed=0,
        max_episode_length=200,
        action_repeat=1,
        bit_depth=8,
        image_dim=None,
        env_kwargs=env_kwargs,
    )

    # Generate the GIF
    gif_path = osp.join(output_dir, f"{env_name}_replay.gif")
    cem_make_gif(env_render, initial_states, action_trajs, configs, output_dir, osp.basename(gif_path))
    print(f"Saved GIF to: {gif_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_path', type=str, required=True, help='Path to cem_traj.pkl')
    parser.add_argument('--env_name', type=str, required=True, help='Environment name (e.g., PassWater)')
    parser.add_argument('--output_dir', type=str, default='.', help='Where to save the output GIF')
    args = parser.parse_args()

    generate_gif(args.pkl_path, args.env_name, args.output_dir)

#!/usr/bin/env python
"""
Run CEM / MPC‑CEM on SoftGym tasks with timing prints for
iterations, steps, episodes, and a device banner.  Allows
--num_envs to cap rollout parallelism.
"""

import argparse
import copy
import json
import multiprocessing as mp
import os
import os.path as osp
import pickle
import time
import ctypes
import numpy as np
import torch

from chester import logger
from cem.cem import CEMPolicy
from cem.visualize_cem import cem_make_gif
from envs.env import Env
from experiments.planet.train import update_env_kwargs
from planet.utils import transform_info
from softgym.registered_env import env_arg_dict

# ------------------------------------------------------------------ #
# Helper: show once which CUDA + GL device we are on
# ------------------------------------------------------------------ #
def show_devices_once():
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        print(f"[Torch]   CUDA device {idx}: {torch.cuda.get_device_name(idx)}")
    else:
        print("[Torch]   CUDA NOT available")

    try:
        # query GL renderer via GLX
        from OpenGL import GL
        renderer = GL.glGetString(GL.GL_RENDERER)
        vendor   = GL.glGetString(GL.GL_VENDOR)
        if renderer and vendor:
            print(f"[OpenGL]  {vendor.decode()} | {renderer.decode()}")
    except Exception as e:
        print(f"[OpenGL]  Could not query renderer ({e})")

# ------------------------------------------------------------------ #
# Per‑environment default planning horizon
# ------------------------------------------------------------------ #
cem_plan_horizon = {
    'PassWater': 7,
    'PourWater': 40,
    'PourWaterAmount': 40,
    'ClothFold': 15,
    'ClothFoldCrumpled': 30,
    'ClothFoldDrop': 30,
    'ClothFlatten': 15,
    'ClothDrop': 15,
    'RopeFlatten': 15,
    'RopeConfiguration': 15,
    'ReachTraj': 12, 
}

# ------------------------------------------------------------------ #
def run_task(vv, log_dir, exp_name):
    # ----- multiprocessing setup -----
    mp.set_start_method('spawn', force=True)
    os.environ["OMP_NUM_THREADS"] = "1"      # keep cpu libs single‑threaded
    torch.set_num_threads(1)

    # show device banner (once in main proc)
    show_devices_once()

    env_name = vv['env_name']
    vv['algorithm'] = 'CEM'
    vv['env_kwargs'] = env_arg_dict[env_name]
    vv['plan_horizon'] = cem_plan_horizon[env_name]

    # derive population / elites
    vv['population_size'] = vv['timestep_per_decision'] // vv['max_iters']
    if vv['use_mpc']:
        vv['population_size'] //= vv['plan_horizon']
    vv['num_elites'] = max(1, vv['population_size'] // 10)
    vv = update_env_kwargs(vv)

    # ---------- logger ----------
    logger.configure(dir=log_dir, exp_name=exp_name)
    os.makedirs(logger.get_dir(), exist_ok=True)
    with open(osp.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=2, sort_keys=True)

    # ---------- env builder ----------
    env_symbolic = vv['env_kwargs']['observation_mode'] != 'cam_rgb'
    base_kwargs = dict(
        env=vv['env_name'],
        symbolic=env_symbolic,
        seed=vv['seed'],
        max_episode_length=200,
        action_repeat=1,
        bit_depth=8,
        image_dim=None,
        env_kwargs=vv['env_kwargs'],
    )
    env = Env(**base_kwargs)

    render_kwargs = copy.deepcopy(base_kwargs)
    render_kwargs['env_kwargs']['render'] = True
    env_render = Env(**render_kwargs)

    # ---------- policy ----------
    policy = CEMPolicy(
        env, Env, base_kwargs,
        vv['use_mpc'],
        plan_horizon=vv['plan_horizon'],
        max_iters=vv['max_iters'],
        population_size=vv['population_size'],
        num_elites=vv['num_elites'],
    )

    # ---------- rollout loop ----------
    initial_states, action_trajs, configs, all_infos = [], [], [], []
    for ep in range(vv['test_episodes']):
        logger.log(f'episode {ep}')
        ep_start = time.time()

        obs = env.reset(); policy.reset()
        initial_state = env.get_state()
        action_traj, infos = [], []

        for t in range(env.horizon):
            logger.log(f'episode {ep}, step {t}')
            step_start = time.time()
            action = policy.get_action(obs)
            logger.log(f'  step runtime: {time.time()-step_start:.3f} s')
            action_traj.append(action.copy())
            obs, reward, _, info = env.step(action)
            infos.append(info)

        logger.log(f'episode {ep} finished in {time.time()-ep_start:.2f} s')

        # record
        all_infos.append(infos)
        initial_states.append(initial_state.copy())
        action_trajs.append(action_traj.copy())
        configs.append(env.get_current_config().copy())

        # tabular metrics
        tr = transform_info([infos])
        for k, arr in tr.items():
            logger.record_tabular('info_final_'+k,   arr[0, -1])
            logger.record_tabular('info_average_'+k, np.mean(arr[0]))
            logger.record_tabular('info_sum_'+k,     np.sum(arr[0]))
        logger.dump_tabular()

    # ---------- save ----------
    with open(osp.join(log_dir, 'cem_traj.pkl'), 'wb') as f:
        pickle.dump(dict(
            initial_states=initial_states,
            action_trajs=action_trajs,
            configs=configs,
        ), f)
    cem_make_gif(env_render, initial_states, action_trajs,
                 configs, logger.get_dir(), vv['env_name'] + '.gif')

# ------------------------------------------------------------------ #
def main():
    p = argparse.ArgumentParser()
    # experiment
    p.add_argument('--exp_name', default='cem')
    p.add_argument('--env_name', default='PassWater')
    p.add_argument('--log_dir',  default='./data/cem/PassWater')
    p.add_argument('--test_episodes', type=int, default=10)
    p.add_argument('--seed', type=int, default=100)
    # CEM
    p.add_argument('--max_iters', type=int, default=10)
    p.add_argument('--timestep_per_decision', type=int, default=21000)
    p.add_argument('--use_mpc', type=bool, default=True)
    # parallelism
    p.add_argument('--num_envs', type=int, default=2,
                   help='rollout workers used by CEM')
    args = p.parse_args()

    run_task(vars(args), args.log_dir, args.exp_name)

# ------------------------------------------------------------------ #
if __name__ == '__main__':
    main()


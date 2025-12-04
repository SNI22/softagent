#!/usr/bin/env python3
import argparse
import csv
import json
import os.path as osp
import pickle
import numpy as np
import inspect

from envs.env import Env

WRAP_ATTRS = ["env", "_env", "unwrapped", "_wrapped_env", "_gym_env", "gym_env", "sim_env"]

def unwrap_chain(env):
    chain = [env]
    for _ in range(8):
        last = chain[-1]
        nxt = None
        for n in WRAP_ATTRS:
            if hasattr(last, n):
                cand = getattr(last, n)
                if cand is not None and cand is not last:
                    nxt = cand
                    break
        if nxt is None or nxt in chain:
            break
        chain.append(nxt)
    return chain

def load_env_from_variant(variant_path, force_keypoint=False, force_action_mode=None):
    with open(variant_path, "r") as f:
        vv = json.load(f)
    if force_keypoint:
        vv["env_kwargs"]["observation_mode"] = "key_point"  # replay-time only
    if force_action_mode is not None:
        vv["env_kwargs"]["action_mode"] = force_action_mode # e.g., "picker"
    vv["env_kwargs"]["render"] = True
    env_symbolic = vv["env_kwargs"]["observation_mode"] != "cam_rgb"
    env = Env(
        env=vv["env_name"],
        symbolic=env_symbolic,
        seed=vv["seed"],
        max_episode_length=200,
        action_repeat=1,
        bit_depth=8,
        image_dim=None,
        env_kwargs=vv["env_kwargs"],
    )
    return env, vv

def load_traj(pkl_path):
    with open(pkl_path, "rb") as f:
        traj = pickle.load(f)
    for k in ["initial_states", "action_trajs", "configs"]:
        if k not in traj:
            raise KeyError(f"Missing '{k}' in {pkl_path}")
    return traj["initial_states"], traj["action_trajs"], traj["configs"]

# ---------- robust un-nesting helpers ----------
def is_numeric_array(x):
    try:
        a = np.asarray(x, dtype=np.float64)
        return np.isfinite(a).all()
    except Exception:
        return False

def extract_xyz_rows(candidate, want_rows=2):
    """
    Try to coerce arbitrarily nested returns like _get_pos()[0][0][0] into an (N,3) array.
    We:
      - walk down lists/tuples until reaching a numeric array
      - if 1D with >=3, take first 3
      - if 2D with last dim >=3, take first N rows and first 3 cols
      - if deeply nested (e.g., [[[ [x,y,z], ... ]]]), flatten until 2D
    """
    # Step 1: descend through containers until we hit something numeric
    obj = candidate
    for _ in range(6):  # avoid infinite loops
        if is_numeric_array(obj):
            break
        if isinstance(obj, (list, tuple)) and len(obj) > 0:
            obj = obj[0]
        else:
            break

    # Step 2: now try to coerce into ndarray and shape to (N,3)
    try:
        arr = np.array(candidate, dtype=np.float64)
    except Exception:
        try:
            arr = np.array(obj, dtype=np.float64)
        except Exception:
            return None

    if arr.ndim == 1 and arr.size >= 3:
        arr = arr[:3][None, :]
    elif arr.ndim >= 2:
        # ensure columns >= 3, and take first want_rows
        if arr.shape[-1] < 3:
            return None
        arr = np.reshape(arr, (-1, arr.shape[-1]))  # flatten leading dims
        arr = arr[:max(want_rows,1), :3]
    else:
        return None

    if not np.isfinite(arr).all():
        return None
    return arr

# ---------- action tool pose readers ----------
def find_action_tool(env):
    for obj in unwrap_chain(env):
        if hasattr(obj, "action_tool"):
            return getattr(obj, "action_tool")
    return None

def read_live_picker_rows(tool):
    """
    Try multiple sources on the tool, in a robust order.
    Returns (rows(N,3), source_string) or (None, reason).
    """
    if tool is None:
        return None, "no action_tool"

    # 1) private/legacy method used in your env comments: _get_pos()
    if hasattr(tool, "_get_pos") and callable(tool._get_pos):
        try:
            rows = extract_xyz_rows(tool._get_pos())
            if rows is not None:
                return rows, "action_tool._get_pos()"
        except Exception:
            pass

    # 2) public-ish variants
    for name in ["get_pos", "_get_target_pos", "get_picker_world_pos", "get_picker_pos"]:
        if hasattr(tool, name) and callable(getattr(tool, name)):
            try:
                rows = extract_xyz_rows(getattr(tool, name)())
                if rows is not None:
                    return rows, f"action_tool.{name}()"
            except Exception:
                pass

    # 3) dynamic attributes that some builds update each step
    for name in ["current_pos", "pos", "picker_pos", "picker_positions"]:
        if hasattr(tool, name):
            try:
                rows = extract_xyz_rows(getattr(tool, name))
                if rows is not None:
                    return rows, f"action_tool.{name}"
            except Exception:
                pass

    # 4) final fallback: static anchors (useful for debugging but not live)
    for name in ["picker_high", "picker_low"]:
        if hasattr(tool, name):
            try:
                rows = extract_xyz_rows(getattr(tool, name))
                if rows is not None:
                    return rows, f"STATIC {name}"
            except Exception:
                pass

    return None, "no pose-like source on tool"

# ---------- main replay ----------
def replay_and_export(pkl_path, csv_out=None, max_episodes=None, max_steps=None,
                      force_keypoint=False, force_action_mode=None,
                      closed_thresh=0.0, grip_invert=False):
    traj_dir = osp.dirname(pkl_path)
    variant_path = osp.join(traj_dir, "variant.json")
    env, vv = load_env_from_variant(variant_path,
                                    force_keypoint=force_keypoint,
                                    force_action_mode=force_action_mode)

    env_name = vv["env_name"]
    obs_mode = vv["env_kwargs"].get("observation_mode", "<unknown>")
    action_mode = vv["env_kwargs"].get("action_mode", "<unknown>")

    inits, actions_all, configs = load_traj(pkl_path)
    N = len(actions_all)
    if max_episodes is not None:
        N = min(N, max_episodes)

    writer = None
    fh = None
    if csv_out:
        fh = open(csv_out, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(["episode", "step", "picker_id", "x", "y", "z", "grip_raw", "grip_closed", "source"])

    print(f"Env: {env_name} | obs_mode: {obs_mode} | action_mode: {action_mode} | episodes={len(actions_all)} (running {N})")

    for epi in range(N):
        env.reset(config=configs[epi], initial_state=inits[epi])

        tool = find_action_tool(env)
        num_picker = getattr(tool, "num_picker", None)
        print(f"\n=== Episode {epi} ===  action_tool={type(tool).__name__ if tool else None}  num_picker={num_picker}")

        actions = actions_all[epi]
        S = len(actions) if max_steps is None else min(len(actions), max_steps)

        prev_rows = None
        for t in range(S):
            act = np.asarray(actions[t], dtype=np.float64)
            move = act[:-1] if act.size >= 4 else act
            grip_raw = float(act[-1]) if act.size >= 4 else np.nan
            grip_eval = -grip_raw if grip_invert and not np.isnan(grip_raw) else grip_raw
            grip_closed = int((grip_eval if not np.isnan(grip_eval) else -1e9) > closed_thresh)

            # advance the sim
            env.step(act)

            # read live pose rows
            rows, source = read_live_picker_rows(tool)
            if rows is not None:
                # compute a max-row delta for quick sanity
                d = None
                if prev_rows is not None and prev_rows.shape == rows.shape:
                    d = float(np.max(np.linalg.norm(rows - prev_rows, axis=1)))
                prev_rows = rows.copy()

                # print and write
                print(f"  step {t:03d} | move_norm={np.linalg.norm(move):.6f}  grip={grip_raw:.3f} closed={grip_closed}  [{source}]"
                      + (f"  d={d:.3e}" if d is not None else ""))
                for pid in range(rows.shape[0]):
                    x, y, z = rows[pid, 0], rows[pid, 1], rows[pid, 2]
                    print(f"           picker{pid}: ({x:.3f}, {y:.3f}, {z:.3f})")
                    if writer:
                        writer.writerow([epi, t, pid, x, y, z, grip_raw, grip_closed, source])
            else:
                print(f"  step {t:03d} | move_norm={np.linalg.norm(move):.6f}  grip={grip_raw:.3f} closed={grip_closed}  [no live source: {source}]")
                if writer:
                    writer.writerow([epi, t, -1, np.nan, np.nan, np.nan, grip_raw, grip_closed, source])

    if fh:
        fh.close()
        print(f"\nSaved CSV: {csv_out}")

def main():
    ap = argparse.ArgumentParser(description="Replay cem_traj.pkl and export LIVE picker poses using tool._get_pos() / friends.")
    ap.add_argument("--pkl", required=True, help="Path to cem_traj.pkl")
    ap.add_argument("--csv-out", type=str, default=None, help="Optional CSV for positions + grip")
    ap.add_argument("--force-keypoint", action="store_true", help="Use key_point observation at replay (not required).")
    ap.add_argument("--force-action-mode", type=str, default=None, help="Override env_kwargs.action_mode at replay (e.g., 'picker').")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--closed-thresh", type=float, default=0.0)
    ap.add_argument("--grip-invert", action="store_true")
    args = ap.parse_args()

    replay_and_export(
        pkl_path=args.pkl,
        csv_out=args.csv_out,
        max_episodes=args.max_episodes,
        max_steps=args.max_steps,
        force_keypoint=args.force_keypoint,
        force_action_mode=args.force_action_mode,
        closed_thresh=args.closed_thresh,
        grip_invert=args.grip_invert,
    )

if __name__ == "__main__":
    main()

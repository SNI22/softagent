#!/usr/bin/env python3
# Re-export first dict in initial_state.pkl to shaped CSVs
import pickle, json
from pathlib import Path
import numpy as np
import pandas as pd
import argparse

def json_safe(o):
    import numpy as np
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(x) for x in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="initial_state.pkl", help="path to PKL")
    ap.add_argument("--idx", type=int, default=0, help="which entry in the PKL list")
    ap.add_argument("--out", default=None, help="output folder (default: <pklstem>_IDX_shaped_csvs)")
    args = ap.parse_args()

    pkl_path = Path(args.pkl)
    with pkl_path.open("rb") as f:
        L = pickle.load(f)

    entry = L[args.idx]
    # Some dumps use {"state": {...}, "config": ...}
    state = entry["state"] if (isinstance(entry, dict) and "state" in entry) else entry

    out_dir = Path(args.out) if args.out else Path(f"{pkl_path.stem}_{args.idx}_shaped_csvs")
    out_dir.mkdir(exist_ok=True, parents=True)

    # Discover N from phase if present; fallback to guess via divisibility
    N = None
    for key in ["particle_phase", "phase"]:
        if key in state and isinstance(state[key], (np.ndarray, list)):
            N = int(np.asarray(state[key]).shape[0])
            break

    def to_2d_known(name, arr):
        a = np.asarray(arr)
        if name in {"particle_pos", "positions"} and N is not None:
            assert a.size % (N * 4) == 0 or a.size == N * 4, "pos not divisible by N*4"
            return a.reshape(-1, 4), ["x", "y", "z", "inv_mass"]
        if name in {"particle_vel", "velocities"} and N is not None:
            assert a.size % (N * 3) == 0 or a.size == N * 3, "vel not divisible by N*3"
            return a.reshape(-1, 3), ["vx", "vy", "vz"]
        if name in {"shape_state", "shape_pos"}:
            assert a.size % 14 == 0, "shape_state not divisible by 14"
            k = a.size // 14
            return a.reshape(k, 14), [f"s{j}" for j in range(14)]
        if name in {"particle_phase", "phase"}:
            return a.reshape(-1, 1), ["phase"]
        # Generic: keep original rank if <=2; otherwise flatten to 2D (… x last)
        if a.ndim == 0:
            return a.reshape(1, 1), ["value"]
        if a.ndim == 1:
            return a.reshape(-1, 1), ["c0"]
        if a.ndim == 2:
            return a, [f"c{j}" for j in range(a.shape[1])]
        # >2 dims: squash to 2D with last axis as columns
        return a.reshape(-1, a.shape[-1]), [f"c{j}" for j in range(a.shape[-1])]

    for k, v in state.items():
        # Dicts → JSON
        if isinstance(v, dict):
            (out_dir / f"{k}.json").write_text(json.dumps(json_safe(v), indent=2))
            print(f"Wrote {k}.json (dict)")          
            print(f"Wrote {k}.json (dict with {len(v)} keys)")
            continue
        # Scalars → one-cell CSV
        if np.isscalar(v):
            pd.DataFrame({"value": [v]}).to_csv(out_dir / f"{k}.csv", index=False)
            print(f"Wrote {k}.csv (scalar)")
            continue

        a2, cols = to_2d_known(k, v)
        # Dtype normalization (keeps your original)
        a2 = np.asarray(a2)
        if "phase" in k.lower():
            a2 = a2.astype(np.int32, copy=False)
        else:
            # keep float arrays float32 if they were
            if np.asarray(v).dtype == np.float32:
                a2 = a2.astype(np.float32, copy=False)

        df = pd.DataFrame(a2, columns=cols)
        df.to_csv(out_dir / f"{k}.csv", index=False)
        print(f"Wrote {k}.csv shape={tuple(a2.shape)}")

    print(f"Done. Output in: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
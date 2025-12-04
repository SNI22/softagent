#!/usr/bin/env python3
"""
Convert per-step XYZ deltas (+ grip flag) into absolute XYZ positions per episode.

Input CSV rows look like:
episode,step,[ dx dy dz grip ]

Output CSV rows look like:
episode,step,x,y,z,grip

By default, each episode starts at [0.4, -0.2, 0.4] (meters). Override with:
  --start "x,y,z"
Or provide per-episode starts with:
  --starts-file episode_starts.csv   (columns: episode,x,y,z)
"""

import argparse
import csv
import re
from collections import defaultdict

FLOATS_IN_BRACKETS = re.compile(r"\[([^\]]+)\]")  # capture the content inside brackets
WHITESPACE_SPLIT = re.compile(r"[,\s]+")

def parse_bracketed_four(s: str):
    """
    Parse a string like "[ 0.12 -0.34 0.56 0.78]" into four floats.
    Handles arbitrary spaces and/or commas between numbers.
    """
    m = FLOATS_IN_BRACKETS.search(s)
    if not m:
        raise ValueError(f"Could not find bracketed values in: {s!r}")
    inside = m.group(1).strip()
    parts = [p for p in WHITESPACE_SPLIT.split(inside) if p]
    if len(parts) != 4:
        raise ValueError(f"Expected 4 numbers in bracketed array, got {len(parts)}: {parts}")
    dx, dy, dz, grip = map(float, parts)
    return dx, dy, dz, grip

def load_per_episode_starts(path: str):
    """
    Read per-episode start positions from a CSV with columns:
      episode,x,y,z
    Returns dict[int] -> (x,y,z)
    """
    starts = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"episode", "x", "y", "z"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{path} must have header with columns: {sorted(required)}")
        for row in reader:
            ep = int(row["episode"])
            starts[ep] = (float(row["x"]), float(row["y"]), float(row["z"]))
    return starts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_csv", required=True, help="Path to input CSV with delta rows")
    ap.add_argument("--out", dest="output_csv", required=True, help="Path to output CSV with absolute rows")
    ap.add_argument("--start", default="0.4,-0.2,0.4",
                    help='Default episode start "x,y,z" (meters). Used when no per-episode start is given.')
    ap.add_argument("--starts-file", default=None,
                    help='Optional CSV of per-episode starts with columns: episode,x,y,z')
    args = ap.parse_args()

    # Parse default start
    try:
        sx, sy, sz = [float(x) for x in args.start.split(",")]
    except Exception as e:
        raise SystemExit(f"--start must be 'x,y,z' floats. Got {args.start!r}: {e}")

    per_episode_start = defaultdict(lambda: (sx, sy, sz))
    if args.starts_file:
        per_episode_start.update(load_per_episode_starts(args.starts_file))

    # Running accumulation per episode
    accum = {}  # episode -> [x, y, z] current absolute
    initialized = set()

    # Buffer rows to sort by (episode, step) if input isn't already ordered
    rows = []
    with open(args.input_csv, newline="") as f:
        # This file might not have headers; we’ll treat it as generic CSV:
        # expected columns: episode, step, bracketed_string
        reader = csv.reader(f)
        for raw in reader:
            if not raw or all(not c.strip() for c in raw):
                continue
            # If a header line sneaks in, skip it
            try:
                ep = int(raw[0])
                st = int(raw[1])
            except Exception:
                # not a data line
                continue
            # The third column may include commas or spaces; rejoin everything after index 1
            bracket_payload = ",".join(raw[2:]) if len(raw) > 3 else raw[2]
            rows.append((ep, st, bracket_payload))

    # Sort rows to ensure correct accumulation order
    rows.sort(key=lambda r: (r[0], r[1]))

    with open(args.output_csv, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["episode", "step", "x", "y", "z", "grip"])

        for ep, st, payload in rows:
            dx, dy, dz, grip = parse_bracketed_four(payload)

            if ep not in initialized:
                # Initialize episode absolute position to its start
                accum[ep] = list(per_episode_start[ep])
                initialized.add(ep)

            # Apply delta to absolute
            ax, ay, az = accum[ep]
            ax += dx
            ay += dy
            az += dz
            accum[ep] = [ax, ay, az]

            writer.writerow([ep, st, ax, ay, az, grip])

if __name__ == "__main__":
    main()

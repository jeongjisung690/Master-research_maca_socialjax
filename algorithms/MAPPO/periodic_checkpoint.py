"""
Periodic checkpointing utility for the SocialJax training scripts.

The training loop is a single ``jax.lax.scan`` over ``NUM_UPDATES`` wrapped in
``jax.vmap`` over seeds, so nothing is written to disk until the whole run
finishes.  To avoid losing progress when a long run is interrupted, we piggy-back
on the per-update host callback that already exists for wandb logging and dump an
evaluation-ready parameter snapshot every ``CHECKPOINT_INTERVAL_STEPS`` env steps.

The snapshots have the exact same on-disk format as ``save_params`` (a pickled
``params`` pytree of numpy arrays), so they load with the existing ``load_params``
/ ``evaluate`` tooling.  They are NOT resumable training states (no optimizer
state / rng / env state) -- they are for evaluation/inference only.

Config keys (optional, read with defaults):
  CHECKPOINT_INTERVAL_STEPS : env-step interval between snapshots (default 50e6)
  CHECKPOINT_DIR            : base directory (default "./checkpoints/periodic")
"""

import os
import pickle
import numpy as np
import jax

DEFAULT_INTERVAL_STEPS = 50_000_000
DEFAULT_DIR = "./checkpoints/periodic"


def _to_numpy(tree):
    """Move every leaf to numpy.

    NOTE: this callback runs *inside* ``jax.vmap`` and ``jax.debug.callback``
    fires once per seed with the natural (un-batched) leaf shape -- there is NO
    leading seed axis to slice here.  The per-boundary guard below means only the
    first seed to reach each boundary is written (matching ``save_params`` which
    keeps a single seed), so for the common NUM_SEEDS=1 case this is exactly the
    seed-0 snapshot.
    """
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def make_periodic_saver(config, filename, per_agent=False, subdir=""):
    """Build a host callback for ``jax.debug.callback``.

    Args:
      config: the (plain dict) run config; uses NUM_STEPS, NUM_ENVS and the
        optional CHECKPOINT_INTERVAL_STEPS / CHECKPOINT_DIR keys.
      filename: base name, e.g. ``"{ENV}_seed{SEED}_reward_{REWARD}"``.
      per_agent: if True, ``params`` is expected to be a sequence of per-agent
        pytrees (IPPO without parameter sharing) and is saved as
        ``{filename}_step{N}_{i}.pkl``; otherwise a single pytree saved as
        ``{filename}_step{N}.pkl``.
      subdir: optional sub-directory under CHECKPOINT_DIR.

    Returns a function ``save(update_steps, params)`` to be invoked from inside
    the scan via ``jax.debug.callback``.  It writes a snapshot the first time the
    env-step count crosses each ``CHECKPOINT_INTERVAL_STEPS`` boundary.
    """
    interval = float(config.get("CHECKPOINT_INTERVAL_STEPS", DEFAULT_INTERVAL_STEPS))
    per_update = int(config["NUM_STEPS"]) * int(config["NUM_ENVS"])
    base = os.path.join(config.get("CHECKPOINT_DIR", DEFAULT_DIR), subdir)
    progress = {"last_boundary": 0}

    def _dump(obj, env_step, suffix=""):
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, f"{filename}_step{env_step}{suffix}.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return path

    def save(update_steps, params):
        step = int(update_steps)
        env_step = step * per_update
        boundary = int(env_step // interval)
        if boundary <= progress["last_boundary"]:
            return
        progress["last_boundary"] = boundary
        if per_agent:
            paths = []
            for i, agent_params in enumerate(params):
                paths.append(_dump(_to_numpy(agent_params), env_step, f"_{i}"))
            print(f"[periodic-checkpoint] env_step={env_step} -> {len(paths)} agent files in {base}")
        else:
            path = _dump(_to_numpy(params), env_step)
            print(f"[periodic-checkpoint] env_step={env_step} -> {path}")

    return save

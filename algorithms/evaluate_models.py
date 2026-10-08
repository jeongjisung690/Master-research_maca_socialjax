#!/usr/bin/env python3
"""Evaluate saved .pkl policies and compare mean episode return across methods.

Loads one or more trained checkpoints (the policy / actor parameters that the
IPPO / MAPPO / MACA training scripts pickle into ``checkpoints/``), rolls each
one out for ``--episodes`` full episodes in the SAME environment, and reports
the mean (+/- std / sem) collective episode return per method.

Why this is a fair comparison
------------------------------
The metric is the *collective ground-truth return* = sum over the episode of
``info["original_rewards"]`` (the un-shared, un-shaped per-agent environment
reward) summed across agents. This equals the total number of reward events
the team produced (apples eaten / cleaned, gifts, territory tiles ...). It is
independent of how a model was trained (individual vs. shared rewards) because:

  * the observation a policy sees does NOT depend on the reward mode, so a
    policy's behaviour is identical regardless of the env's ``shared_rewards``;
  * we therefore force ``shared_rewards=True`` for every model so the env always
    exposes ``info["original_rewards"]`` and we measure the same physical
    quantity for all methods.

Network architecture (small MAPPO/MACA Actor vs. the larger IPPO ActorCritic) is
auto-detected from the pickled parameter tree, so you can mix methods freely.

Examples
--------
    # Compare the three top-level harvest checkpoints, 32 episodes each
    python algorithms/evaluate_models.py --env harvest_common_open --episodes 32

    # Explicit checkpoints with custom labels
    python algorithms/evaluate_models.py --env clean_up --episodes 50 \
        --ckpt "IPPO(ind)=checkpoints/clean_up_seed30_reward_individual.pkl" \
        --ckpt "MAPPO=checkpoints/clean_up_seed30_reward_MAPPO.pkl" \
        --ckpt "MACA=checkpoints/clean_up_seed30_reward_MACA_JNT.pkl"

    # Greedy (argmax) actions instead of sampling, save a CSV + bar plot
    python algorithms/evaluate_models.py --env gift --greedy --csv out.csv --plot
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

# Make the local `socialjax` package importable when run as a script from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax

import socialjax


# --------------------------------------------------------------------------- #
# Networks. Copied verbatim from the training scripts so the pickled params
# load with identical structure.
# --------------------------------------------------------------------------- #
class CNNSmall(nn.Module):
    """CNN used by MAPPO / MACA Actor."""
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh
        x = nn.Conv(16, (3, 3), kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0))(x)
        x = act(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(16, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = act(x)
        return x


class Actor(nn.Module):
    """Policy network used by MAPPO and MACA (saved checkpoints = these params)."""
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs):
        act = nn.relu if self.activation == "relu" else nn.tanh
        # name="CNN_0" matches the original class (named `CNN`) so pickled keys bind.
        embedding = CNNSmall(self.activation, name="CNN_0")(obs)
        x = nn.Dense(16, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(embedding)
        x = act(x)
        x = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                     bias_init=constant(0.0))(x)
        return distrax.Categorical(logits=x)


class CNNLarge(nn.Module):
    """CNN used by IPPO ActorCritic."""
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh
        x = nn.Conv(32, (5, 5), kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0))(x)
        x = act(x)
        x = nn.Conv(32, (3, 3), kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0))(x)
        x = act(x)
        x = nn.Conv(32, (3, 3), kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0))(x)
        x = act(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = act(x)
        return x


class ActorCritic(nn.Module):
    """Combined policy+value network used by IPPO (the 'individual' checkpoints)."""
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh
        embedding = CNNLarge(self.activation, name="CNN_0")(x)
        actor = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)),
                         bias_init=constant(0.0))(embedding)
        actor = act(actor)
        actor = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                         bias_init=constant(0.0))(actor)
        pi = distrax.Categorical(logits=actor)
        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)),
                          bias_init=constant(0.0))(embedding)
        critic = act(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0),
                          bias_init=constant(0.0))(critic)
        return pi, jnp.squeeze(critic, axis=-1)


# The "partnership" variant of harvest is the SAME registered env
# (`harvest_common_open`) trained on a custom map; we must pass the same map at
# eval time so the layout/dynamics match the trained policies.
HARVEST_PARTNERSHIP_MAP = [
    "AAA    A      A    AAA",
    "AA    AAA    AAA    AA",
    "A    AAAAA  AAAAA    A",
    "      AAA    AAA      ",
    "       A      A       ",
    "  A                A  ",
    " AAA  Q   WW   Q  AAA ",
    "AAAAA            AAAAA",
    " AAA  WWWWWWWWWW  AAA ",
    "  A       WW       A  ",
    "WWWWWWWWW WW WWWWWWWWW",
    "          WW          ",
    "  WWWWWWWWWWWWWWWWWW  ",
    "  P P P P P P P P P   ",
    " P P P P P P P P P P  ",
]

# Per-task config (must match how the models were trained; only `shared_rewards`
# is forced to True for a fair, mode-independent metric). `env_kwargs` carries any
# extra task-specific kwargs (e.g. a custom map).
TASKS = {
    "harvest_common_open": dict(env_name="harvest_common_open", num_agents=7),
    "clean_up":            dict(env_name="clean_up",            num_agents=7),
    "gift":                dict(env_name="gift",                num_agents=5),
    "territory_open":      dict(env_name="territory_open",      num_agents=7),
    "harvest_partnership": dict(env_name="harvest_common_open", num_agents=7,
                                env_kwargs=dict(map_ASCII=HARVEST_PARTNERSHIP_MAP)),
}


def make_env(env_name, num_agents, num_inner_steps, env_kwargs=None):
    return socialjax.make(
        env_name,
        num_agents=num_agents,
        num_inner_steps=num_inner_steps,
        shared_rewards=True,   # forced: exposes info["original_rewards"]
        cnn=True,
        jit=True,
        **(env_kwargs or {}),
    )


def _shapes(tree):
    return [tuple(np.shape(x)) for x in jax.tree_util.tree_leaves(tree)]


def select_network(params, env, activation="relu"):
    """Pick the network whose init params match the pickled tree exactly."""
    dummy = jnp.zeros((1, *env.observation_space()[0].shape))
    target_struct = jax.tree_util.tree_structure(params)
    target_shapes = _shapes(params)
    candidates = [
        ("Actor", Actor(env.action_space().n, activation)),
        ("ActorCritic", ActorCritic(env.action_space().n, activation)),
    ]
    for name, net in candidates:
        p = net.init(jax.random.PRNGKey(0), dummy)
        if (jax.tree_util.tree_structure(p) == target_struct
                and _shapes(p) == target_shapes):
            return name, net
    raise ValueError(
        "Checkpoint params do not match any known architecture.\n"
        f"  loaded shapes: {target_shapes}\n"
        f"  Actor shapes:  {_shapes(candidates[0][1].init(jax.random.PRNGKey(0), dummy))}\n"
        f"  AC shapes:     {_shapes(candidates[1][1].init(jax.random.PRNGKey(0), dummy))}"
    )


def load_params(path):
    with open(path, "rb") as f:
        params = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), params)


def build_evaluator(env, net, net_name, num_steps, greedy):
    obs_shape = env.observation_space()[0].shape

    def apply_pi(params, obs_batch):
        out = net.apply(params, obs_batch)
        return out[0] if net_name == "ActorCritic" else out

    def run_episode(params, rng):
        rng, rk = jax.random.split(rng)
        obs, state = env.reset(rk)

        def step(carry, _):
            rng, obs, state = carry
            obs_batch = obs.reshape(-1, *obs_shape)
            pi = apply_pi(params, obs_batch)
            rng, ak = jax.random.split(rng)
            actions = pi.mode() if greedy else pi.sample(seed=ak)
            rng, sk = jax.random.split(rng)
            obs, state, reward, done, info = env.step(sk, state, actions)
            team_r = jnp.sum(info["original_rewards"])
            return (rng, obs, state), team_r

        (_, _, _), team_rs = jax.lax.scan(step, (rng, obs, state), None, num_steps)
        return jnp.sum(team_rs)

    @jax.jit
    def eval_episodes(params, keys):
        return jax.vmap(lambda k: run_episode(params, k))(keys)

    return eval_episodes


def default_checkpoints(task, ckpt_dir):
    """Best-guess default checkpoints for a task, if present on disk."""
    ckpt_dir = Path(ckpt_dir)
    # Partnership models live in dedicated sub-dirs with step-tagged names.
    if task == "harvest_partnership":
        candidates = [
            ("MAPPO",  ckpt_dir / "mappo/harvest_partnership/_step100000000.pkl"),
            ("MACA",   ckpt_dir / "maca_mlae/harvest_partnership/_step100000000.pkl"),
        ]
        return [(label, str(p)) for label, p in candidates if p.exists()]

    # The plain tasks have top-level {individual, MAPPO, MACA_JNT} checkpoints.
    env_name = TASKS[task]["env_name"]
    seed_tag = f"{env_name}_seed30_reward_"
    mapping = [("IPPO(ind)", "individual"), ("MAPPO", "MAPPO"), ("MACA", "MACA_JNT")]
    out = []
    for label, suffix in mapping:
        p = ckpt_dir / f"{seed_tag}{suffix}.pkl"
        if p.exists():
            out.append((label, str(p)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True, choices=list(TASKS),
                    help="task / environment name")
    ap.add_argument("--ckpt", action="append", default=[], metavar="LABEL=PATH",
                    help="checkpoint to evaluate; repeatable. If omitted, the "
                         "top-level {individual,MAPPO,MACA_JNT} checkpoints are used.")
    ap.add_argument("--episodes", type=int, default=32,
                    help="number of episodes per method (default 32)")
    ap.add_argument("--num-agents", type=int, default=None,
                    help="override num_agents (default: per-task value)")
    ap.add_argument("--num-steps", type=int, default=1000,
                    help="steps per episode = num_inner_steps (default 1000)")
    ap.add_argument("--seed", type=int, default=0,
                    help="base PRNG seed (same eval seeds reused for every method)")
    ap.add_argument("--greedy", action="store_true",
                    help="take argmax actions instead of sampling")
    ap.add_argument("--ckpt-dir", default="checkpoints",
                    help="dir for default checkpoint lookup (default ./checkpoints)")
    ap.add_argument("--csv", default=None, help="write per-episode returns to CSV")
    ap.add_argument("--plot", action="store_true",
                    help="save a bar chart of mean returns to figures/")
    args = ap.parse_args()

    task = TASKS[args.env]
    num_agents = args.num_agents or task["num_agents"]
    env = make_env(task["env_name"], num_agents, args.num_steps,
                   task.get("env_kwargs"))

    # Resolve checkpoint list.
    if args.ckpt:
        ckpts = []
        for spec in args.ckpt:
            if "=" not in spec:
                raise SystemExit(f"--ckpt expects LABEL=PATH, got: {spec}")
            label, path = spec.split("=", 1)
            ckpts.append((label, path))
    else:
        ckpts = default_checkpoints(args.env, args.ckpt_dir)
        if not ckpts:
            raise SystemExit(
                f"No default checkpoints found for {args.env} in {args.ckpt_dir}. "
                "Pass --ckpt LABEL=PATH explicitly.")

    print(f"Environment : {args.env}  (num_agents={num_agents}, "
          f"steps/episode={args.num_steps}, shared_rewards=True)")
    print(f"Episodes    : {args.episodes}   Action: "
          f"{'greedy(argmax)' if args.greedy else 'sample'}   Seed: {args.seed}")
    print(f"Metric      : collective episode return = sum_t sum_i "
          f"info['original_rewards']\n")

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)

    results = {}
    for label, path in ckpts:
        if not Path(path).exists():
            print(f"[skip] {label}: file not found: {path}")
            continue
        params = load_params(path)
        net_name, net = select_network(params, env)
        evaluator = build_evaluator(env, net, net_name, args.num_steps, args.greedy)
        t0 = time.time()
        returns = np.asarray(jax.block_until_ready(evaluator(params, keys)))
        dt = time.time() - t0
        results[label] = returns
        mean, std = returns.mean(), returns.std()
        sem = std / np.sqrt(len(returns))
        per_agent = mean / num_agents
        print(f"{label:14s} [{net_name:11s}] "
              f"mean={mean:9.2f} +/- {std:7.2f} (std)  sem={sem:6.2f}  "
              f"per-agent={per_agent:7.2f}  "
              f"min={returns.min():.0f} max={returns.max():.0f}  ({dt:.1f}s)")

    if not results:
        raise SystemExit("No checkpoints were evaluated.")

    # Ranked summary.
    print("\n=== Ranking (by mean collective return) ===")
    for rank, (label, r) in enumerate(
            sorted(results.items(), key=lambda kv: kv[1].mean(), reverse=True), 1):
        print(f"  {rank}. {label:14s} {r.mean():9.2f} +/- "
              f"{r.std()/np.sqrt(len(r)):.2f} (sem)")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "episode", "collective_return"])
            for label, r in results.items():
                for i, v in enumerate(r):
                    w.writerow([label, i, float(v)])
        print(f"\nWrote per-episode returns -> {args.csv}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = list(results)
        means = [results[k].mean() for k in labels]
        sems = [results[k].std() / np.sqrt(len(results[k])) for k in labels]
        fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2, 4))
        ax.bar(labels, means, yerr=sems, capsize=5,
               color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"][:len(labels)])
        ax.set_ylabel("Mean collective episode return")
        ax.set_title(f"{args.env}  ({args.episodes} episodes, "
                     f"{'greedy' if args.greedy else 'sampled'})")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out = Path("figures") / f"eval_{args.env}.png"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"Saved bar chart -> {out}")


if __name__ == "__main__":
    main()

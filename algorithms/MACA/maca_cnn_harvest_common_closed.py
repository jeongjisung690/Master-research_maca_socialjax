""" 
Based on PureJaxRL & jaxmarl Implementation of PPO
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import socialjax
from socialjax.wrappers.baselines import MAPPOWorldStateWrapper, LogWrapper
import hydra
from omegaconf import OmegaConf
import wandb
import copy
import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from periodic_checkpoint import make_periodic_saver

class CNN(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        # x = nn.Conv(
        #     features=32,
        #     kernel_size=(3, 3),
        #     kernel_init=orthogonal(np.sqrt(2)),
        #     bias_init=constant(0.0),
        # )(x)
        # x = activation(x)
        # x = nn.Conv(
        #     features=32,
        #     kernel_size=(3, 3),
        #     kernel_init=orthogonal(np.sqrt(2)),
        #     bias_init=constant(0.0),
        # )(x)
        # x = activation(x)
        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(
            features=16, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = activation(x)

        return x


class Actor(nn.Module):
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        embedding = CNN(self.activation)(obs)
        actor_mean = nn.Dense(
            16, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)

        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        return pi


class Critic(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        world_state = x

        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(world_state)
        hidden = nn.Dense(
            features=16,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)

        hidden = activation(hidden)

        value = nn.Dense(
            features=1,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(hidden)
        # squeeze 去除最后一个维度
        return jnp.squeeze(value, axis=-1)
    

class MACACritic(nn.Module):
    action_dim: int
    num_agents: int
    activation: str = "relu"
    n_embd: int = 64
    zs_dim: int = 256
    attention_temperature: float = 1.0
    q_head_type: str = "linear"
    pairwise_rank: int = 8
    pairwise_coef: float = 0.1
    fuse_hidden_dim: int = 128
    pair_hidden_dim: int = 128
    target_agent_conditioning: bool = False

    @nn.compact
    def __call__(self, obs_seq, pi_bar, agent_id=None):
        if self.activation == "relu":
            activation = nn.relu
        elif self.activation == "gelu":
            activation = nn.gelu
        else:
            activation = nn.tanh

        batch_size = obs_seq.shape[0]
        obs_flat = obs_seq.reshape((-1,) + obs_seq.shape[-3:])
        token = CNN(self.activation)(obs_flat)
        token = nn.Dense(
            self.n_embd,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(token)
        token = activation(token)
        token = token.reshape((batch_size, self.num_agents, self.n_embd))

        query = nn.Dense(self.n_embd, use_bias=False, kernel_init=orthogonal(np.sqrt(2)))(token)
        key = nn.Dense(self.n_embd, use_bias=False, kernel_init=orthogonal(np.sqrt(2)))(token)
        value = nn.Dense(self.n_embd, use_bias=False, kernel_init=orthogonal(np.sqrt(2)))(token)
        att_logits = jnp.einsum("bid,bjd->bij", query, key) / jnp.sqrt(self.n_embd)
        temperature = jnp.maximum(jnp.asarray(self.attention_temperature, dtype=att_logits.dtype), 1e-6)
        attention = nn.softmax(att_logits / temperature, axis=-1)
        attended = jnp.einsum("bij,bjd->bid", attention, value)
        token = nn.LayerNorm()(token + attended)

        mlp = nn.Dense(
            self.n_embd * 2,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(token)
        mlp = activation(mlp)
        mlp = nn.Dense(
            self.n_embd,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(mlp)
        token = nn.LayerNorm()(token + mlp)

        zs = token.reshape((batch_size, -1))
        zs = nn.Dense(
            self.zs_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(zs)
        zs = activation(zs)

        if self.target_agent_conditioning or self.q_head_type == "agent_action_fuse":
            if agent_id is None:
                agent_id = jnp.zeros((batch_size,), dtype=jnp.int32)
            agent_id = agent_id.astype(jnp.int32)
            target_token = token[jnp.arange(batch_size), agent_id]
            agent_onehot = jax.nn.one_hot(agent_id, self.num_agents)
            agent_emb = nn.Dense(
                self.n_embd,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(agent_onehot)
            target_context = jnp.concatenate([target_token, agent_emb], axis=-1)
            target_context = nn.Dense(
                self.zs_dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(target_context)
            target_context = activation(target_context)
            zs = nn.LayerNorm()(zs + target_context)
            token_context = nn.Dense(
                self.n_embd,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(target_context)
            token = nn.LayerNorm()(token + token_context[:, None, :])

        state_bias = nn.Dense(
            1,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(zs)
        state_bias = jnp.squeeze(state_bias, axis=-1)
        if self.q_head_type == "pairwise_fuse":
            unary_table = nn.Dense(
                self.action_dim,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0),
            )(token)
            unary_value = jnp.sum(unary_table * pi_bar, axis=(1, 2))

            rank = max(int(self.pairwise_rank), 1)
            left_table = nn.Dense(
                self.action_dim * rank,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(token)
            right_table = nn.Dense(
                self.action_dim * rank,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(token)
            left_table = left_table.reshape((batch_size, self.num_agents, self.action_dim, rank))
            right_table = right_table.reshape((batch_size, self.num_agents, self.action_dim, rank))
            left_action = jnp.einsum("bna,bnar->bnr", pi_bar, left_table)
            right_action = jnp.einsum("bna,bnar->bnr", pi_bar, right_table)
            pair_energy = jnp.einsum("bnr,bmr->bnm", left_action, right_action) / jnp.sqrt(rank)
            pair_mask = 1.0 - jnp.eye(self.num_agents, dtype=pair_energy.dtype)[None, :, :]
            pair_attention = attention.astype(pair_energy.dtype) * pair_mask
            pair_norm = jnp.sum(pair_attention, axis=(1, 2)) + 1e-6
            pair_value = jnp.sum(pair_attention * pair_energy, axis=(1, 2)) / pair_norm
            q_value = state_bias + unary_value + self.pairwise_coef * pair_value
        elif self.q_head_type == "pairwise_full":
            unary_table = nn.Dense(
                self.action_dim,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0),
            )(token)
            unary_value = jnp.sum(unary_table * pi_bar, axis=(1, 2))

            token_i = jnp.broadcast_to(
                token[:, :, None, :],
                (batch_size, self.num_agents, self.num_agents, self.n_embd),
            )
            token_j = jnp.broadcast_to(
                token[:, None, :, :],
                (batch_size, self.num_agents, self.num_agents, self.n_embd),
            )
            pair_feat = jnp.concatenate([token_i, token_j], axis=-1)
            pair_hidden = nn.Dense(
                self.pair_hidden_dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(pair_feat)
            pair_hidden = activation(pair_hidden)
            pair_w_flat = nn.Dense(
                self.action_dim * self.action_dim,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0),
            )(pair_hidden)
            pair_W = pair_w_flat.reshape(
                (batch_size, self.num_agents, self.num_agents, self.action_dim, self.action_dim)
            )
            pair_energy = jnp.einsum("bnp,bnmpq,bmq->bnm", pi_bar, pair_W, pi_bar)
            pair_mask = 1.0 - jnp.eye(self.num_agents, dtype=pair_energy.dtype)[None, :, :]
            pair_value = jnp.sum(
                attention.astype(pair_energy.dtype) * pair_mask * pair_energy,
                axis=(1, 2),
            )
            q_value = state_bias + unary_value + pair_value
        elif self.q_head_type == "agent_action_fuse":
            agent_id = agent_id.astype(jnp.int32)
            target_token = token[jnp.arange(batch_size), agent_id]
            attention_row = jnp.take_along_axis(
                attention,
                agent_id[:, None, None],
                axis=1,
            ).squeeze(axis=1)
            action_emb = nn.Dense(
                self.n_embd,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(pi_bar)
            target_token_tiled = jnp.tile(target_token[:, None, :], (1, self.num_agents, 1))
            fuse_input = jnp.concatenate(
                [token, action_emb, target_token_tiled, attention_row[..., None]],
                axis=-1,
            )
            fuse_hidden = nn.Dense(
                self.fuse_hidden_dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(fuse_input)
            fuse_hidden = activation(fuse_hidden)
            fuse_hidden = nn.Dense(
                self.fuse_hidden_dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(fuse_hidden)
            fuse_hidden = activation(fuse_hidden)
            fused_action = jnp.sum(attention_row[..., None] * fuse_hidden, axis=1)
            q_input = jnp.concatenate([zs, target_token, fused_action], axis=-1)
            q_hidden = nn.Dense(
                self.fuse_hidden_dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(q_input)
            q_hidden = activation(q_hidden)
            q_hidden = nn.Dense(
                self.fuse_hidden_dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(q_hidden)
            q_hidden = activation(q_hidden)
            q_residual = nn.Dense(
                1,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0),
            )(q_hidden)
            q_value = state_bias + jnp.squeeze(q_residual, axis=-1)
        else:
            pi_flat = pi_bar.reshape((batch_size, self.num_agents * self.action_dim))
            action_coef = nn.Dense(
                self.num_agents * self.action_dim,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0),
            )(zs)
            q_value = state_bias + jnp.sum(action_coef * pi_flat, axis=-1)

        psi_logits = nn.Dense(
            3,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(zs)
        psi = nn.softmax(psi_logits, axis=-1)

        return q_value, attention, psi


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    critic_obs: jnp.ndarray
    joint_action_onehot: jnp.ndarray
    joint_action_probs: jnp.ndarray
    agent_id: jnp.ndarray
    info: jnp.ndarray


class MACANextBatch(NamedTuple):
    critic_obs: jnp.ndarray
    joint_action_onehot: jnp.ndarray
    joint_action_probs: jnp.ndarray
    agent_id: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[str(a)] for a in agent_list])
    return x.reshape((num_actors, -1))

def batchify_numpy(x: dict, agent_list, num_actors):
    x = jnp.stack([x[:, a] for a in agent_list])
    return x.reshape((num_actors, -1))

def batchify_dict(x: dict, agent_list, num_actors):
    x = jnp.stack([x[str(a)] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


MACA_Q_MODES = {"maca_jnt", "jnt", "maca_ind", "ind", "maca_cor", "cor", "maca_full", "full"}


def is_maca_q_mode(maca_mode):
    return str(maca_mode).lower() in MACA_Q_MODES


def actor_major_to_env_agent(x, num_envs, num_agents):
    return jnp.swapaxes(x.reshape((num_agents, num_envs) + x.shape[1:]), 0, 1)


def env_agent_to_actor_major(x, num_envs, num_agents):
    return jnp.swapaxes(x, 0, 1).reshape((num_envs * num_agents,) + x.shape[2:])


def duplicate_env_data_for_actors(x, num_envs, num_agents):
    x = jnp.expand_dims(x, axis=0)
    x = jnp.tile(x, (num_agents,) + (1,) * (x.ndim - 1))
    return x.reshape((num_envs * num_agents,) + x.shape[2:])


def make_maca_actor_inputs(obs, action, action_probs, num_envs, num_agents, action_dim):
    action_onehot = jax.nn.one_hot(action, action_dim)
    action_onehot_env = actor_major_to_env_agent(action_onehot, num_envs, num_agents)
    action_probs_env = actor_major_to_env_agent(action_probs, num_envs, num_agents)
    critic_obs = duplicate_env_data_for_actors(obs, num_envs, num_agents)
    joint_action_onehot = duplicate_env_data_for_actors(action_onehot_env, num_envs, num_agents)
    joint_action_probs = duplicate_env_data_for_actors(action_probs_env, num_envs, num_agents)
    agent_id = jnp.repeat(jnp.arange(num_agents), num_envs)
    return critic_obs, joint_action_onehot, joint_action_probs, agent_id


def build_maca_pi_bars(joint_action_onehot, joint_action_probs, attention, agent_id, sigma):
    num_agents = joint_action_onehot.shape[1]
    agent_mask = jnp.arange(num_agents)[None, :] == agent_id[:, None]

    pi_bar_taken = joint_action_onehot
    pi_bar_jnt = joint_action_probs
    pi_bar_ind = jnp.where(agent_mask[..., None], joint_action_probs, joint_action_onehot)

    attention_row = jnp.take_along_axis(
        attention,
        agent_id[:, None, None],
        axis=1,
    ).squeeze(axis=1)
    corr_mask = jnp.logical_or(attention_row >= sigma, agent_mask)
    pi_bar_cor = jnp.where(corr_mask[..., None], joint_action_probs, joint_action_onehot)

    return pi_bar_taken, pi_bar_jnt, pi_bar_ind, pi_bar_cor, corr_mask


def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    maca_config = config.get("MACA", {})
    maca_mode = str(maca_config.get("MODE", "joint")).lower()
    use_maca_q = is_maca_q_mode(maca_mode)
    if maca_mode not in {"joint"} and not use_maca_q:
        raise NotImplementedError(
            "Supported MACA.MODE values are 'joint', 'maca_jnt', 'maca_ind', 'maca_cor', and 'maca_full'."
        )
    maca_sigma = float(maca_config.get("ATT_SIGMA", 1.0 / env.num_agents))
    maca_attention_temperature = float(maca_config.get("ATT_TEMPERATURE", 1.0))
    maca_q_value_loss_coef = float(maca_config.get("Q_VALUE_LOSS_COEF", 0.5))
    maca_v_value_loss_coef = float(maca_config.get("V_VALUE_LOSS_COEF", 1.0))
    maca_critic_loss_type = str(maca_config.get("CRITIC_LOSS_TYPE", "td")).lower()
    maca_mla_baseline_loss_coef = float(maca_config.get("MLA_BASELINE_LOSS_COEF", 0.0))
    maca_ind_baseline_loss_coef = float(maca_config.get("IND_BASELINE_LOSS_COEF", 0.0))
    maca_cor_baseline_loss_coef = float(maca_config.get("COR_BASELINE_LOSS_COEF", 0.0))
    maca_fixed_psi = jnp.array(maca_config.get("PSI", [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]))
    maca_advantage_source = str(maca_config.get("ADVANTAGE_SOURCE", "lambda_mla")).lower()
    default_psi_type = (
        "learned"
        if maca_advantage_source
        in {"full_maca_gae", "baseline_maca_gae", "maca_baseline_gae"}
        else "fixed"
    )
    maca_psi_type = str(maca_config.get("PSI_TYPE", default_psi_type)).lower()
    maca_q_head_type = str(maca_config.get("Q_HEAD_TYPE", "linear")).lower()
    maca_pairwise_rank = int(maca_config.get("PAIRWISE_RANK", 8))
    maca_pairwise_coef = float(maca_config.get("PAIRWISE_COEF", 0.1))
    maca_fuse_hidden_dim = int(maca_config.get("FUSE_HIDDEN_DIM", 128))
    maca_pair_hidden_dim = int(maca_config.get("PAIR_HIDDEN_DIM", 128))
    maca_target_agent_conditioning = bool(maca_config.get("TARGET_AGENT_CONDITIONING", False))
    maca_q_adv_mix_coef = float(maca_config.get("Q_ADV_MIX_COEF", 0.1))
    maca_q_adv_warmup_updates = int(maca_config.get("Q_ADV_WARMUP_UPDATES", 1000))
    maca_q_adv_min_std = float(maca_config.get("Q_ADV_MIN_STD", 1e-3))

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]

    env = MAPPOWorldStateWrapper(env)
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):

        # INIT NETWORK
        action_dim = env.action_space().n
        actor_network = Actor(action_dim, activation=config["ACTIVATION"])
        if use_maca_q:
            critic_network = MACACritic(
                action_dim=action_dim,
                num_agents=env.num_agents,
                activation=maca_config.get("CRITIC_ACTIVATION", "gelu"),
                n_embd=int(maca_config.get("N_EMBD", 64)),
                zs_dim=int(maca_config.get("ZS_DIM", 256)),
                attention_temperature=maca_attention_temperature,
                q_head_type=maca_q_head_type,
                pairwise_rank=maca_pairwise_rank,
                pairwise_coef=maca_pairwise_coef,
                fuse_hidden_dim=maca_fuse_hidden_dim,
                pair_hidden_dim=maca_pair_hidden_dim,
                target_agent_conditioning=maca_target_agent_conditioning,
            )
        else:
            critic_network = Critic(activation=config["ACTIVATION"])

        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        ac_init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))
            
        actor_network_params = actor_network.init(_rng_actor, ac_init_x)

        obs_shape = env.observation_space()[0].shape
        if use_maca_q:
            cr_init_obs = jnp.zeros((1, env.num_agents, *obs_shape))
            cr_init_pi_bar = jnp.zeros((1, env.num_agents, action_dim))
            cr_init_agent_id = jnp.zeros((1,), dtype=jnp.int32)
            critic_network_params = critic_network.init(_rng_critic, cr_init_obs, cr_init_pi_bar, cr_init_agent_id)
        else:
            global_shape = (1, *obs_shape[:-1], obs_shape[-1] * env.num_agents)
            cr_init_x = jnp.zeros(global_shape)
            # cr_init_x = jnp.zeros((1, *(env.observation_space()[0]).shape)) 
            critic_network_params = critic_network.init(_rng_critic, cr_init_x)

        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)


                obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)
                # obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(-1, *env.observation_space().shape)

                print("input_obs_shape", obs_batch.shape)

                pi = actor_network.apply(train_states[0].params, obs_batch)
                action = pi.sample(seed=_rng)
                action_probs = pi.probs
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )

                # env_act = {k: v.flatten() for k, v in env_act.items()}
                env_act = [v for v in env_act.values()]

                #VALUE
                world_state = jnp.transpose(last_obs, (0,2,3,1,4)).reshape(config["NUM_ENVS"], *(env.observation_space()[0]).shape[:-1], -1)
                world_state = jnp.expand_dims(world_state, axis=0)
                world_state = jnp.tile(world_state, (env.num_agents, 1, 1, 1, 1))
                world_state = jnp.reshape(world_state, (-1, *(world_state.shape[2:])))
                # world_state = last_obs["world_state"].swapaxes(0,1)  
                # world_state = world_state.reshape((config["NUM_ACTORS"],-1))
                critic_obs, joint_action_onehot, joint_action_probs, agent_id = make_maca_actor_inputs(
                    last_obs,
                    action,
                    action_probs,
                    config["NUM_ENVS"],
                    env.num_agents,
                    action_dim,
                )
                if use_maca_q:
                    value, _, _ = critic_network.apply(
                        train_states[1].params,
                        critic_obs,
                        joint_action_probs,
                        agent_id,
                    )
                else:
                    value = critic_network.apply(train_states[1].params, world_state)
                # expanded_value = jnp.expand_dims(value, axis=0)
                # value = jnp.tile(expanded_value, (env.num_agents, 1))
                value = value.reshape(config["NUM_ACTORS"])


                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                info = jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action,
                    value,
                    batchify_numpy(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    world_state,
                    critic_obs,
                    joint_action_onehot,
                    joint_action_probs,
                    agent_id,
                    info
                )
                runner_state = (train_states, env_state, obsv, done_batch, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, last_done, rng = runner_state

            # last_world_state = last_obs["world_state"].swapaxes(0,1)
            # last_world_state = last_world_state.reshape((config["NUM_ACTORS"],-1))
            if use_maca_q:
                last_obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)
                last_pi = actor_network.apply(train_states[0].params, last_obs_batch)
                rng, _rng_last_action = jax.random.split(rng)
                last_action = last_pi.sample(seed=_rng_last_action)
                last_critic_obs, last_joint_action_onehot, last_joint_action_probs, last_agent_id = make_maca_actor_inputs(
                    last_obs,
                    last_action,
                    last_pi.probs,
                    config["NUM_ENVS"],
                    env.num_agents,
                    action_dim,
                )
                last_val, _, _ = critic_network.apply(
                    train_states[1].params,
                    last_critic_obs,
                    last_joint_action_probs,
                    last_agent_id,
                )
            else:
                last_world_state = jnp.transpose(last_obs, (0,2,3,1,4)).reshape(config["NUM_ENVS"], *(env.observation_space()[0]).shape[:-1], -1)
                last_val = critic_network.apply(train_states[1].params, last_world_state)
                last_val = jnp.expand_dims(last_val, axis=0)
                last_val = jnp.tile(last_val, (env.num_agents, 1))
                last_val = last_val.reshape((config["NUM_ACTORS"],-1))
                last_val = last_val.squeeze()

            if use_maca_q:
                def _append_last(xs, last_x):
                    return jnp.concatenate([xs[1:], last_x[None]], axis=0)

                next_maca_batch = MACANextBatch(
                    critic_obs=_append_last(traj_batch.critic_obs, last_critic_obs),
                    joint_action_onehot=_append_last(traj_batch.joint_action_onehot, last_joint_action_onehot),
                    joint_action_probs=_append_last(traj_batch.joint_action_probs, last_joint_action_probs),
                    agent_id=_append_last(traj_batch.agent_id, last_agent_id),
                )
            else:
                next_maca_batch = MACANextBatch(
                    critic_obs=traj_batch.critic_obs,
                    joint_action_onehot=traj_batch.joint_action_onehot,
                    joint_action_probs=traj_batch.joint_action_probs,
                    agent_id=traj_batch.agent_id,
                )

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            def _calculate_gae_split_baseline(traj_batch, last_val, baseline):
                # GAE where the *bootstrap* term keeps V (traj_batch.value / last_val)
                # but the *subtracted baseline* is a separate action-conditioned signal
                # (b^MACA). Used by "baseline_maca_gae":
                #   delta_t = r_t + gamma * V(s_{t+1}) - b^MACA_i(s_t, a_t)
                #   A_t     = delta_t + gamma * lambda * (1 - done) * A_{t+1}
                # NB: because the subtracted term (baseline) differs from the bootstrap
                # term (V), the usual telescoping does not hold -> this is a biased
                # advantage estimator unless b^MACA ~ V.
                def _get_advantages(gae_and_next_value, transition_and_baseline):
                    gae, next_value = gae_and_next_value
                    transition, base = transition_and_baseline
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - base
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    # Carry V (value), not the baseline, so the next (earlier) step
                    # bootstraps on V(s_{t+1}).
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    (traj_batch, baseline),
                    reverse=True,
                    unroll=16,
                )
                # Lambda-return that this advantage bootstraps on, used as the
                # self-consistency target for b^MACA: R_t = A_t + b^MACA(s_t, a_t).
                return advantages, advantages + baseline

            def _compute_maca_values(critic_params, critic_obs, joint_action_onehot, joint_action_probs, agent_id):
                q_taken, attention, psi_learned = critic_network.apply(
                    critic_params,
                    critic_obs,
                    joint_action_onehot,
                    agent_id,
                )
                _, pi_bar_jnt, pi_bar_ind, pi_bar_cor, corr_mask = build_maca_pi_bars(
                    joint_action_onehot,
                    joint_action_probs,
                    attention,
                    agent_id.astype(jnp.int32),
                    maca_sigma,
                )
                q_jnt, _, _ = critic_network.apply(critic_params, critic_obs, pi_bar_jnt, agent_id)
                q_ind, _, _ = critic_network.apply(critic_params, critic_obs, pi_bar_ind, agent_id)
                q_cor, _, _ = critic_network.apply(critic_params, critic_obs, pi_bar_cor, agent_id)

                psi_fixed = jnp.tile(maca_fixed_psi[None, :], (q_taken.shape[0], 1))
                psi_full = jnp.where(
                    jnp.array(maca_psi_type == "learned"),
                    psi_learned,
                    psi_fixed,
                )
                psi_jnt = jnp.tile(jnp.array([[1.0, 0.0, 0.0]]), (q_taken.shape[0], 1))
                psi_ind = jnp.tile(jnp.array([[0.0, 1.0, 0.0]]), (q_taken.shape[0], 1))
                psi_cor = jnp.tile(jnp.array([[0.0, 0.0, 1.0]]), (q_taken.shape[0], 1))

                if maca_mode in {"maca_jnt", "jnt"}:
                    baseline = q_jnt
                    psi = psi_jnt
                elif maca_mode in {"maca_ind", "ind"}:
                    baseline = q_ind
                    psi = psi_ind
                elif maca_mode in {"maca_cor", "cor"}:
                    baseline = q_cor
                    psi = psi_cor
                else:
                    baseline = (
                        psi_full[:, 0] * q_jnt
                        + psi_full[:, 1] * q_ind
                        + psi_full[:, 2] * q_cor
                    )
                    psi = psi_full

                q_advantage = q_taken - baseline
                return q_advantage, baseline, q_taken, q_jnt, q_ind, q_cor, psi, corr_mask

            advantages, targets = _calculate_gae(traj_batch, last_val)
            if use_maca_q:
                maca_values = _compute_maca_values(
                    train_states[1].params,
                    traj_batch.critic_obs.reshape((-1,) + traj_batch.critic_obs.shape[2:]),
                    traj_batch.joint_action_onehot.reshape((-1,) + traj_batch.joint_action_onehot.shape[2:]),
                    traj_batch.joint_action_probs.reshape((-1,) + traj_batch.joint_action_probs.shape[2:]),
                    traj_batch.agent_id.reshape((-1,)),
                )
                q_maca_advantages = maca_values[0].reshape(traj_batch.action.shape)
                mla_baseline = maca_values[1].reshape(traj_batch.action.shape)
                q_taken_values = maca_values[2].reshape(traj_batch.action.shape)
                jnt_baseline = maca_values[3].reshape(traj_batch.action.shape)
                ind_baseline = maca_values[4].reshape(traj_batch.action.shape)
                cor_baseline = maca_values[5].reshape(traj_batch.action.shape)
                lambda_mla_advantages = targets - mla_baseline
                lambda_jnt_advantages = targets - jnt_baseline
                q_taken_jnt_delta = q_taken_values - jnt_baseline
                ind_jnt_delta = ind_baseline - jnt_baseline
                cor_jnt_delta = cor_baseline - jnt_baseline
                cor_ind_delta = cor_baseline - ind_baseline
                adv_var_eps = jnp.array(1e-8, dtype=lambda_mla_advantages.dtype)
                q_adv_std = q_maca_advantages.std()
                mixed_q_coef = jnp.clip(
                    jnp.asarray(maca_q_adv_mix_coef, dtype=lambda_mla_advantages.dtype),
                    0.0,
                    1.0,
                )
                # Full MACA-GAE: replace V everywhere in standard GAE with b^MACA_i.
                # delta_t = r_t + gamma * b^MACA_i(s_{t+1}, a_{t+1}) - b^MACA_i(s_t, a_t)
                if maca_advantage_source == "full_maca_gae":
                    last_maca_values = _compute_maca_values(
                        train_states[1].params,
                        last_critic_obs,
                        last_joint_action_onehot,
                        last_joint_action_probs,
                        last_agent_id,
                    )
                    last_mla_baseline = last_maca_values[1]
                    traj_batch_for_maca_gae = traj_batch._replace(value=mla_baseline)
                    full_maca_gae_advantages, full_maca_gae_targets = _calculate_gae(
                        traj_batch_for_maca_gae, last_mla_baseline
                    )
                    # Self-consistency target for b^MACA: the lambda-return that full
                    # MACA-GAE itself bootstraps on (R^MACA_t = A_t + b^MACA(s_t,a_t)).
                    maca_baseline_targets = full_maca_gae_targets
                elif maca_advantage_source in {"baseline_maca_gae", "maca_baseline_gae"}:
                    # GAE with only the subtracted baseline swapped to b^MACA; the
                    # bootstrap term keeps the state value V (traj_batch.value / last_val).
                    # delta_t = r_t + gamma * V(s_{t+1}) - b^MACA_i(s_t, a_t)
                    baseline_maca_gae_advantages, baseline_maca_gae_targets = (
                        _calculate_gae_split_baseline(traj_batch, last_val, mla_baseline)
                    )
                    full_maca_gae_advantages = lambda_mla_advantages
                    maca_baseline_targets = baseline_maca_gae_targets
                else:
                    full_maca_gae_advantages = lambda_mla_advantages
                    maca_baseline_targets = targets
                if maca_advantage_source == "q":
                    use_q_advantage = jnp.array(True)
                    q_adv_weight = jnp.array(1.0, dtype=lambda_mla_advantages.dtype)
                elif maca_advantage_source == "gae":
                    use_q_advantage = jnp.array(False)
                    q_adv_weight = jnp.array(0.0, dtype=lambda_mla_advantages.dtype)
                elif maca_advantage_source == "q_warmup":
                    use_q_advantage = jnp.logical_and(
                        update_steps >= maca_q_adv_warmup_updates,
                        q_adv_std >= maca_q_adv_min_std,
                    )
                    q_adv_weight = use_q_advantage.astype(lambda_mla_advantages.dtype)
                elif maca_advantage_source in {"mixed_q", "mix_q"}:
                    use_q_advantage = mixed_q_coef > 0.0
                    q_adv_weight = mixed_q_coef
                elif maca_advantage_source == "full_maca_gae":
                    use_q_advantage = jnp.array(False)
                    q_adv_weight = jnp.array(0.0, dtype=lambda_mla_advantages.dtype)
                elif maca_advantage_source in {"baseline_maca_gae", "maca_baseline_gae"}:
                    use_q_advantage = jnp.array(False)
                    q_adv_weight = jnp.array(0.0, dtype=lambda_mla_advantages.dtype)
                else:
                    use_q_advantage = jnp.array(False)
                    q_adv_weight = jnp.array(0.0, dtype=lambda_mla_advantages.dtype)
                if maca_advantage_source == "gae":
                    maca_advantages = advantages
                elif maca_advantage_source in {"q", "q_warmup"}:
                    maca_advantages = jnp.where(use_q_advantage, q_maca_advantages, advantages)
                elif maca_advantage_source in {"mixed_q", "mix_q"}:
                    maca_advantages = (1.0 - q_adv_weight) * lambda_mla_advantages + q_adv_weight * q_maca_advantages
                elif maca_advantage_source == "full_maca_gae":
                    maca_advantages = full_maca_gae_advantages
                elif maca_advantage_source in {"baseline_maca_gae", "maca_baseline_gae"}:
                    maca_advantages = baseline_maca_gae_advantages
                else:
                    maca_advantages = lambda_mla_advantages

                # --- psi-sensitivity diagnostic for the full MACA-GAE advantage ---
                # Recompute the full_maca_gae advantage under vertex / uniform psi and
                # measure how much the (normalized) advantage the actor consumes changes.
                # corr ~ 1 across psi  =>  psi has no leverage on A^full (mixing inert);
                # corr clearly < 1     =>  psi genuinely moves the actor signal.
                if maca_advantage_source == "full_maca_gae":
                    psi_diag_last = last_maca_values
                else:
                    psi_diag_last = _compute_maca_values(
                        train_states[1].params,
                        last_critic_obs,
                        last_joint_action_onehot,
                        last_joint_action_probs,
                        last_agent_id,
                    )

                def _full_maca_gae_adv_for_psi(psi_vec):
                    b_psi = (
                        psi_vec[0] * jnt_baseline
                        + psi_vec[1] * ind_baseline
                        + psi_vec[2] * cor_baseline
                    )
                    last_b_psi = (
                        psi_vec[0] * psi_diag_last[3]
                        + psi_vec[1] * psi_diag_last[4]
                        + psi_vec[2] * psi_diag_last[5]
                    )
                    adv, _ = _calculate_gae(traj_batch._replace(value=b_psi), last_b_psi)
                    return adv

                def _adv_corr(a, b):
                    a = a.reshape(-1)
                    b = b.reshape(-1)
                    a = a - a.mean()
                    b = b - b.mean()
                    denom = jnp.sqrt((a * a).mean() * (b * b).mean()) + adv_var_eps
                    return (a * b).mean() / denom

                adv_full_jnt = _full_maca_gae_adv_for_psi(jnp.array([1.0, 0.0, 0.0]))
                adv_full_ind = _full_maca_gae_adv_for_psi(jnp.array([0.0, 1.0, 0.0]))
                adv_full_cor = _full_maca_gae_adv_for_psi(jnp.array([0.0, 0.0, 1.0]))
                adv_full_unif = _full_maca_gae_adv_for_psi(
                    jnp.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
                )

                # --- coherence diagnostic for the full-vs-lambda_mla difference ---
                # prop = A^full - A^mla = ((1-lambda)/lambda) * sum_m (gamma*lambda)^m e_{t+m},
                # the discounted future counterfactual correction that full propagates and
                # lambda_mla does not. e_t = b^MACA - V is the per-step correction.
                # "small but coherent" = small std(prop)/std(A^mla) YET e is temporally
                # persistent (autocorr > 0) and prop systematically steers the advantage.
                maca_e = mla_baseline - jnt_baseline
                adv_full_actual, _ = _calculate_gae(
                    traj_batch._replace(value=mla_baseline), psi_diag_last[1]
                )
                maca_prop = adv_full_actual - lambda_mla_advantages

                maca_advantage_delta = jnp.array(0.0)
                maca_debug = {
                    "maca_q_taken_mean": maca_values[2].mean(),
                    "maca_b_mla_mean": maca_values[1].mean(),
                    "maca_b_jnt_mean": maca_values[3].mean(),
                    "maca_b_ind_mean": maca_values[4].mean(),
                    "maca_b_cor_mean": maca_values[5].mean(),
                    "maca_abs_b_ind_jnt_mean": jnp.abs(ind_jnt_delta).mean(),
                    "maca_abs_b_cor_jnt_mean": jnp.abs(cor_jnt_delta).mean(),
                    "maca_abs_b_cor_ind_mean": jnp.abs(cor_ind_delta).mean(),
                    "maca_abs_q_taken_b_jnt_mean": jnp.abs(q_taken_jnt_delta).mean(),
                    "maca_std_q_taken_b_jnt": q_taken_jnt_delta.std(),
                    "maca_std_b_ind_jnt": ind_jnt_delta.std(),
                    "maca_std_b_cor_jnt": cor_jnt_delta.std(),
                    "maca_std_b_cor_ind": cor_ind_delta.std(),
                    "maca_q_adv_mean": maca_values[0].mean(),
                    "maca_q_adv_std": maca_values[0].std(),
                    "maca_lambda_jnt_adv_mean": lambda_jnt_advantages.mean(),
                    "maca_lambda_jnt_adv_std": lambda_jnt_advantages.std(),
                    "maca_lambda_jnt_adv_var": lambda_jnt_advantages.var(),
                    "maca_lambda_mla_adv_mean": lambda_mla_advantages.mean(),
                    "maca_lambda_mla_adv_std": lambda_mla_advantages.std(),
                    "maca_lambda_mla_adv_var": lambda_mla_advantages.var(),
                    "maca_mla_jnt_adv_var_ratio": lambda_mla_advantages.var() / (lambda_jnt_advantages.var() + adv_var_eps),
                    "maca_abs_adv_mla_jnt_mean": jnp.abs(lambda_mla_advantages - lambda_jnt_advantages).mean(),
                    "maca_adv_mean": maca_advantages.mean(),
                    "maca_adv_std": maca_advantages.std(),
                    "maca_selected_adv_mean": maca_advantages.mean(),
                    "maca_selected_adv_std": maca_advantages.std(),
                    "maca_used_q_advantage": use_q_advantage.astype(jnp.float32),
                    "maca_q_adv_weight": q_adv_weight,
                    "maca_psi_jnt_mean": maca_values[6][:, 0].mean(),
                    "maca_psi_ind_mean": maca_values[6][:, 1].mean(),
                    "maca_psi_cor_mean": maca_values[6][:, 2].mean(),
                    "maca_corrset_size_mean": maca_values[7].sum(axis=-1).mean(),
                    "maca_psi_diag_corr_jnt_unif": _adv_corr(adv_full_jnt, adv_full_unif),
                    "maca_psi_diag_corr_jnt_ind": _adv_corr(adv_full_jnt, adv_full_ind),
                    "maca_psi_diag_corr_jnt_cor": _adv_corr(adv_full_jnt, adv_full_cor),
                    "maca_psi_diag_corr_ind_cor": _adv_corr(adv_full_ind, adv_full_cor),
                    "maca_psi_diag_adv_std_jnt": adv_full_jnt.std(),
                    "maca_psi_diag_adv_std_ind": adv_full_ind.std(),
                    "maca_psi_diag_adv_std_cor": adv_full_cor.std(),
                    "maca_psi_diag_adv_std_unif": adv_full_unif.std(),
                    "maca_e_mean": maca_e.mean(),
                    "maca_e_std": maca_e.std(),
                    "maca_prop_rel_std": maca_prop.std() / (lambda_mla_advantages.std() + adv_var_eps),
                    "maca_e_autocorr_lag1": _adv_corr(maca_e[:-1], maca_e[1:]),
                    "maca_prop_corr_amla": _adv_corr(maca_prop, lambda_mla_advantages),
                    "maca_prop_corr_e": _adv_corr(maca_prop, maca_e),
                }
            else:
                maca_advantages = targets - traj_batch.value
                maca_advantage_delta = jnp.max(jnp.abs(maca_advantages - advantages))
                maca_baseline_targets = targets
                maca_debug = {
                    "maca_q_taken_mean": jnp.array(0.0),
                    "maca_b_mla_mean": jnp.array(0.0),
                    "maca_b_jnt_mean": jnp.array(0.0),
                    "maca_b_ind_mean": jnp.array(0.0),
                    "maca_b_cor_mean": jnp.array(0.0),
                    "maca_abs_b_ind_jnt_mean": jnp.array(0.0),
                    "maca_abs_b_cor_jnt_mean": jnp.array(0.0),
                    "maca_abs_b_cor_ind_mean": jnp.array(0.0),
                    "maca_abs_q_taken_b_jnt_mean": jnp.array(0.0),
                    "maca_std_q_taken_b_jnt": jnp.array(0.0),
                    "maca_std_b_ind_jnt": jnp.array(0.0),
                    "maca_std_b_cor_jnt": jnp.array(0.0),
                    "maca_std_b_cor_ind": jnp.array(0.0),
                    "maca_q_adv_mean": jnp.array(0.0),
                    "maca_q_adv_std": jnp.array(0.0),
                    "maca_lambda_jnt_adv_mean": maca_advantages.mean(),
                    "maca_lambda_jnt_adv_std": maca_advantages.std(),
                    "maca_lambda_jnt_adv_var": maca_advantages.var(),
                    "maca_lambda_mla_adv_mean": maca_advantages.mean(),
                    "maca_lambda_mla_adv_std": maca_advantages.std(),
                    "maca_lambda_mla_adv_var": maca_advantages.var(),
                    "maca_mla_jnt_adv_var_ratio": jnp.array(1.0),
                    "maca_abs_adv_mla_jnt_mean": jnp.array(0.0),
                    "maca_adv_mean": maca_advantages.mean(),
                    "maca_adv_std": maca_advantages.std(),
                    "maca_selected_adv_mean": maca_advantages.mean(),
                    "maca_selected_adv_std": maca_advantages.std(),
                    "maca_used_q_advantage": jnp.array(0.0),
                    "maca_q_adv_weight": jnp.array(0.0),
                    "maca_psi_jnt_mean": jnp.array(1.0),
                    "maca_psi_ind_mean": jnp.array(0.0),
                    "maca_psi_cor_mean": jnp.array(0.0),
                    "maca_corrset_size_mean": jnp.array(0.0),
                }

            # UPDATE NETWORK
            def _actor_loss_fn(actor_params, traj_batch, gae):
                # RERUN NETWORK
                pi = actor_network.apply(
                    actor_params,
                    jnp.reshape(traj_batch.obs, (-1, *(traj_batch.obs).shape[-3:])),
                ) #.reshape(traj_batch.action.shape)

                log_prob = pi.log_prob(traj_batch.action.reshape(-1))

                # CALCULATE ACTOR LOSS
                logratio = log_prob - jnp.reshape(traj_batch.log_prob, (-1,))
                logratio = jnp.reshape(logratio, traj_batch.action.shape)
                log_prob = jnp.reshape(log_prob, traj_batch.action.shape)
                ratio = jnp.exp(log_prob - traj_batch.log_prob)
                gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                loss_actor1 = ratio * gae
                loss_actor2 = (
                    jnp.clip(
                        ratio,
                        1.0 - config["CLIP_EPS"],
                        1.0 + config["CLIP_EPS"],
                    )
                    * gae
                )
                loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                loss_actor = loss_actor.mean()
                entropy = pi.entropy().mean()
                # debug
                approx_kl = ((ratio - 1) - logratio).mean()
                clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                actor_loss = (
                    loss_actor
                    - config["ENT_COEF"] * entropy
                )
                return actor_loss, (loss_actor, entropy, ratio.mean(), approx_kl, clip_frac)

            def _critic_loss_fn(critic_params, traj_batch, targets, maca_baseline_targets, next_maca_batch):
                if use_maca_q:
                    (
                        _,
                        baseline,
                        q_taken,
                        q_jnt,
                        q_ind,
                        q_cor,
                        psi,
                        corr_mask,
                    ) = _compute_maca_values(
                        critic_params,
                        traj_batch.critic_obs,
                        traj_batch.joint_action_onehot,
                        traj_batch.joint_action_probs,
                        traj_batch.agent_id,
                    )
                    (
                        _,
                        next_baseline,
                        next_q_taken,
                        next_q_jnt,
                        _,
                        _,
                        _,
                        _,
                    ) = _compute_maca_values(
                        critic_params,
                        next_maca_batch.critic_obs,
                        next_maca_batch.joint_action_onehot,
                        next_maca_batch.joint_action_probs,
                        next_maca_batch.agent_id,
                    )
                    not_done = 1.0 - traj_batch.global_done.astype(q_jnt.dtype)
                    td_v_target = jax.lax.stop_gradient(
                        traj_batch.reward + config["GAMMA"] * next_q_jnt * not_done
                    )
                    td_q_target = jax.lax.stop_gradient(
                        traj_batch.reward + config["GAMMA"] * next_q_taken * not_done
                    )
                    td_q_loss = 0.5 * jnp.square(q_taken - td_q_target).mean()
                    td_v_loss = 0.5 * jnp.square(q_jnt - td_v_target).mean()
                    q_return_loss = 0.5 * jnp.square(q_taken - targets).mean()
                    v_return_loss = 0.5 * jnp.square(q_jnt - targets).mean()
                    # b^MACA self-consistency: fit the multi-level baseline that
                    # full MACA-GAE bootstraps on to its own return.
                    #   td:     b^MACA(s_t,a_t) <- r_t + gamma * b^MACA(s_{t+1},a_{t+1})
                    #   return: b^MACA(s_t,a_t) <- R^MACA_t (lambda-return on b^MACA)
                    td_mla_target = jax.lax.stop_gradient(
                        traj_batch.reward + config["GAMMA"] * next_baseline * not_done
                    )
                    td_mla_loss = 0.5 * jnp.square(baseline - td_mla_target).mean()
                    mla_return_loss = 0.5 * jnp.square(
                        baseline - jax.lax.stop_gradient(maca_baseline_targets)
                    ).mean()
                    if maca_critic_loss_type in {"gae_return", "lambda_return", "return"}:
                        q_loss = q_return_loss
                        v_loss = v_return_loss
                        mla_baseline_loss = mla_return_loss
                    else:
                        q_loss = td_q_loss
                        v_loss = td_v_loss
                        mla_baseline_loss = td_mla_loss
                    ind_loss = 0.5 * jnp.square(q_ind - targets).mean()
                    cor_loss = 0.5 * jnp.square(q_cor - targets).mean()
                    critic_loss = (
                        maca_q_value_loss_coef * q_loss
                        + maca_v_value_loss_coef * v_loss
                        + maca_mla_baseline_loss_coef * mla_baseline_loss
                        + maca_ind_baseline_loss_coef * ind_loss
                        + maca_cor_baseline_loss_coef * cor_loss
                    )
                    return critic_loss, {
                        "q_loss": q_loss,
                        "v_loss": v_loss,
                        "q_return_loss": q_return_loss,
                        "v_return_loss": v_return_loss,
                        "mla_baseline_loss": mla_baseline_loss,
                        "td_mla_loss": td_mla_loss,
                        "mla_return_loss": mla_return_loss,
                        "td_mla_target": td_mla_target.mean(),
                        "ind_loss": ind_loss,
                        "cor_loss": cor_loss,
                        "td_q_target": td_q_target.mean(),
                        "td_v_target": td_v_target.mean(),
                        "b_mla": baseline.mean(),
                        "q_taken": q_taken.mean(),
                        "b_jnt": q_jnt.mean(),
                        "b_ind": q_ind.mean(),
                        "b_cor": q_cor.mean(),
                        "abs_b_ind_jnt": jnp.abs(q_ind - q_jnt).mean(),
                        "abs_b_cor_jnt": jnp.abs(q_cor - q_jnt).mean(),
                        "abs_b_cor_ind": jnp.abs(q_cor - q_ind).mean(),
                        "abs_q_taken_b_jnt": jnp.abs(q_taken - q_jnt).mean(),
                        "std_b_ind_jnt": (q_ind - q_jnt).std(),
                        "std_b_cor_jnt": (q_cor - q_jnt).std(),
                        "std_b_cor_ind": (q_cor - q_ind).std(),
                        "std_q_taken_b_jnt": (q_taken - q_jnt).std(),
                        "psi_jnt": psi[:, 0].mean(),
                        "psi_ind": psi[:, 1].mean(),
                        "psi_cor": psi[:, 2].mean(),
                        "corrset_size": corr_mask.sum(axis=-1).mean(),
                    }
                else:
                    # RERUN NETWORK
                    value = critic_network.apply(critic_params, traj_batch.world_state.reshape(-1, *(traj_batch.world_state).shape[-3:]))
                    value = jnp.reshape(value, traj_batch.value.shape)
                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )
                    critic_loss = config["VF_COEF"] * value_loss
                    return critic_loss, {"v_loss": value_loss}

            def _update_minbatch(train_state, batch_info):
                actor_train_state, critic_train_state = train_state
                traj_batch, advantages, targets, maca_baseline_targets, next_maca_batch = batch_info

                actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                actor_loss, actor_grads = actor_grad_fn(
                    actor_train_state.params, traj_batch, advantages
                )
                critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                critic_loss, critic_grads = critic_grad_fn(
                    critic_train_state.params, traj_batch, targets, maca_baseline_targets, next_maca_batch
                )

                actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)

                total_loss = actor_loss[0] + critic_loss[0]
                loss_info = {
                    "total_loss": total_loss,
                    "actor_loss": actor_loss[0],
                    "value_loss": critic_loss[0],
                    "entropy": actor_loss[1][1],
                    "ratio": actor_loss[1][2],
                    "approx_kl": actor_loss[1][3],
                    "clip_frac": actor_loss[1][4],
                }
                if use_maca_q:
                    loss_info.update(
                        {
                            "critic_q_loss": critic_loss[1]["q_loss"],
                            "critic_v_loss": critic_loss[1]["v_loss"],
                            "critic_q_return_loss": critic_loss[1]["q_return_loss"],
                            "critic_v_return_loss": critic_loss[1]["v_return_loss"],
                            "critic_mla_baseline_loss": critic_loss[1]["mla_baseline_loss"],
                            "critic_td_mla_loss": critic_loss[1]["td_mla_loss"],
                            "critic_mla_return_loss": critic_loss[1]["mla_return_loss"],
                            "critic_td_mla_target": critic_loss[1]["td_mla_target"],
                            "critic_ind_loss": critic_loss[1]["ind_loss"],
                            "critic_cor_loss": critic_loss[1]["cor_loss"],
                            "critic_td_q_target": critic_loss[1]["td_q_target"],
                            "critic_td_v_target": critic_loss[1]["td_v_target"],
                            "critic_q_taken": critic_loss[1]["q_taken"],
                            "critic_b_mla": critic_loss[1]["b_mla"],
                            "critic_b_jnt": critic_loss[1]["b_jnt"],
                            "critic_b_ind": critic_loss[1]["b_ind"],
                            "critic_b_cor": critic_loss[1]["b_cor"],
                            "critic_abs_b_ind_jnt": critic_loss[1]["abs_b_ind_jnt"],
                            "critic_abs_b_cor_jnt": critic_loss[1]["abs_b_cor_jnt"],
                            "critic_abs_b_cor_ind": critic_loss[1]["abs_b_cor_ind"],
                            "critic_abs_q_taken_b_jnt": critic_loss[1]["abs_q_taken_b_jnt"],
                            "critic_std_b_ind_jnt": critic_loss[1]["std_b_ind_jnt"],
                            "critic_std_b_cor_jnt": critic_loss[1]["std_b_cor_jnt"],
                            "critic_std_b_cor_ind": critic_loss[1]["std_b_cor_ind"],
                            "critic_std_q_taken_b_jnt": critic_loss[1]["std_q_taken_b_jnt"],
                            "critic_corrset_size": critic_loss[1]["corrset_size"],
                        }
                    )
                else:
                    loss_info.update(
                        {
                            "critic_v_loss": critic_loss[1]["v_loss"],
                        }
                    )

                return (actor_train_state, critic_train_state), loss_info

            def _update_epoch(update_state, unused):
                (
                    train_states,
                    traj_batch,
                    maca_advantages,
                    targets,
                    maca_baseline_targets,
                    next_maca_batch,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, maca_advantages, targets, maca_baseline_targets, next_maca_batch)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )

                train_states, loss_info = jax.lax.scan(
                    _update_minbatch, train_states, minibatches
                )
                update_state = (
                    train_states,
                    traj_batch,
                    maca_advantages,
                    targets,
                    maca_baseline_targets,
                    next_maca_batch,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_states,
                traj_batch,
                maca_advantages,
                targets,
                maca_baseline_targets,
                next_maca_batch,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_states = update_state[0]
            metric = traj_batch.info
            loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
            rng = update_state[-1]

            def callback(metric):
                wandb.log(metric)
            update_steps = update_steps + 1
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            metric["update_steps"] = update_steps
            metric["env_step"] = update_steps * config["NUM_STEPS"] * config["NUM_ENVS"]
            metric["maca_advantage_delta"] = maca_advantage_delta
            metric["maca_joint_only"] = jnp.array(0.0 if use_maca_q else 1.0)
            metric["maca_q_mode"] = jnp.array(1.0 if use_maca_q else 0.0)
            metric.update(maca_debug)
            metric.update(loss_info)

            # jax.experimental.io_callback(callback, None, metric)

            jax.debug.callback(callback, metric)
            jax.debug.callback(_periodic_saver, update_steps, train_states[0].params)
            runner_state = (train_states, env_state, last_obs, last_done, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            _rng,
        )
        _periodic_saver = make_periodic_saver(
            config, f'{config["ENV_NAME"]}_seed{config["SEED"]}_reward_{config["REWARD"]}'
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train

def single_run(config):
    config = OmegaConf.to_container(config)
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["MACA", "MACA_JOINT", "FF"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f'maca_cnn_harvest_common_closed'
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    # train_jit = jax.jit(make_train(config))

    train_jit = make_train(config)
    
    out = jax.vmap(train_jit)(rngs)

    print("** Saving Results **")
    filename = f'{config["ENV_NAME"]}_seed{config["SEED"]}_reward_{config["REWARD"]}'
    train_state = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0][0][0])
    save_path = f'{config.get("CHECKPOINT_DIR", "./checkpoints")}/{filename}.pkl'
    save_params(train_state, save_path)
    params = load_params(save_path)
    print("** Evaluating Results **")
    evaluate(params, socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]), save_path, config)
    # state_seq = get_rollout(train_state.params, config)
    # viz = OvercookedVisualizer()
    # agent_view_size is hardcoded as it determines the padding around the layout.
    # viz.animate(state_seq, agent_view_size=5, filename=f"{filename}.gif")

def save_params(train_state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    params = jax.tree_util.tree_map(lambda x: np.array(x), train_state.params)

    with open(save_path, 'wb') as f:
        pickle.dump(params, f)

def load_params(load_path):
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.array(x), params)

def evaluate(params, env, save_path, config):
    rng = jax.random.PRNGKey(0)
    
    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)
    done = False
    
    pics = []
    img = env.render(state)
    pics.append(img)
    root_dir = f"evaluation/harvest_common_closed"
    path = Path(root_dir + "/state_pics")
    path.mkdir(parents=True, exist_ok=True)

    for o_t in range(config["GIF_NUM_FRAMES"]):
        # 获取所有智能体的观察
        print(o_t)
        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *env.observation_space()[0].shape)

        # 使用模型选择动作
        network = Actor(action_dim=env.action_space().n, activation=config["ACTIVATION"])  # 使用与训练时相同的参数
        pi= network.apply(params, obs_batch)
        rng, _rng = jax.random.split(rng)
        actions = pi.sample(seed=_rng)
        
        # 转换动作格式
        env_act = {k: v.squeeze() for k, v in unbatchify(
            actions, env.agents, 1, env.num_agents
        ).items()}
        
        # 执行动作
        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, [v.item() for v in env_act.values()])
        done = done["__all__"]
        
        # 记录结果
        # episode_reward += sum(reward.values())
        
        # 渲染
        img = env.render(state)
        pics.append(img)
        
        print('###################')
        print(f'Actions: {env_act}')
        print(f'Reward: {reward}')
        print(f'State: {state.agent_locs}')
        print("###################")
    
    # 保存GIF
    print(f"Saving Episode GIF")
    pics = [Image.fromarray(img) for img in pics]
    pics[0].save(
    f"{root_dir}/state_outer_step_{o_t+1}.gif",
    format="GIF",
    save_all=True,
    optimize=False,
    append_images=pics[1:],
    duration=200,
    loop=0,
    )
        
        # print(f"Episode {episode} total reward: {episode_reward}")
def tune(default_config):
    """
    Hyperparameter sweep with wandb, including logic to:
    - Initialize wandb
    - Train for each hyperparameter set
    - Save checkpoint
    - Evaluate and log GIF
    """
    import copy

    default_config = OmegaConf.to_container(default_config)

    sweep_config = {
        "name": "harvest",
        "method": "grid",
        "metric": {
            "name": "returned_episode_original_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {"values": [0.001, 0.0005, 0.0001, 0.00005]},
            "ACTIVATION": {"values": ["relu", "tanh"]},
            "UPDATE_EPOCHS": {"values": [2, 4, 8]},
            "NUM_MINIBATCHES": {"values": [4, 8, 16, 32]},
            "CLIP_EPS": {"values": [0.1, 0.2, 0.3]},
            "ENT_COEF": {"values": [0.001, 0.01, 0.1]},
            "NUM_STEPS": {"values": [64, 128, 256]},
        },
    }

    def wrapped_make_train():


        wandb.init(project=default_config["PROJECT"])
        config = copy.deepcopy(default_config)
        # only overwrite the single nested key we're sweeping
        for k, v in dict(wandb.config).items():
            if "." in k:
                parent, child = k.split(".", 1)
                config[parent][child] = v
            else:
                config[k] = v


        # Rename the run for clarity
        run_name = f"sweep_maca_{config['ENV_NAME']}_seed{config['SEED']}"
        wandb.run.name = run_name
        print("Running experiment:", run_name)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))
        train_state = jax.tree_util.tree_map(lambda x: x[0], outs["runner_state"][0])

        # Evaluate and log
        # params = load_params(train_state.params)
        # test_env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        # evaluate(params, test_env, config)

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


@hydra.main(version_base=None, config_path="config", config_name="maca_cnn_harvest_common_closed")
def main(config):
    if config["TUNE"]:
        tune(config)
    else:
        single_run(config)

if __name__ == "__main__":
    main()

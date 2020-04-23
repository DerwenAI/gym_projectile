#!/usr/bin/env python
# encoding: utf-8

from gym_projectile.envs.projectile_env import Projectile_v0
from ray.tune.registry import register_env
import gym
import pprint
import ray
import ray.rllib.agents.sac as sac
import sys
# NB: SACTrainer requires tensorflow_probability so test it here
import tensorflow as tf
import tensorflow_probability as tfp


CHECKPOINT_PATH = "/tmp/sac/proj"
SELECT_ENV = "projectile-v0"
N_ITER = 10


def train_policy (agent, path, debug=True, n_iter=N_ITER):
    reward_history = []

    for _ in range(n_iter):
        result = agent.train()

        max_reward = result["episode_reward_max"]
        reward_history.append(max_reward)

        checkpoint_path = agent.save(path)

        if debug:
            pprint.pprint(result)

    return checkpoint_path, reward_history


def rollout_actions (agent, env, debug=True, render=True, max_steps=1000, episode_interval=20):
    for step in range(max_steps):
        if step % episode_interval == 0:
            state = env.reset()

        last_state = state
        print("state", state)
        action = agent.compute_action(state)
        state, reward, done, info = env.step(action)

        if debug:
            print("state", last_state, "action", action, "reward", reward)
            print(info)

        if render:
            env.render()

        if done == 1 and reward > 0:
            break


if __name__ == "__main__":
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    config = sac.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"

    register_env("projectile-v0", lambda config: Projectile_v0())

    # train a policy with RLlib using SAC

    agent = sac.SACTrainer(config, env=SELECT_ENV)
    checkpoint_path, reward_history = train_policy(agent, CHECKPOINT_PATH)

    print(reward_history)

    # apply the trained policy in a use case

    agent.restore(checkpoint_path)
    env = gym.make(SELECT_ENV)

    rollout_actions(agent, env)

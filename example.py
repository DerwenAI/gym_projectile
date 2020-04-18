#!/usr/bin/env python
# encoding: utf-8

import gym
import gym_projectile

N_ITER = 100
           

if __name__ == "__main__":
    env = gym.make("projectile-v0")

    for i in range(N_ITER):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

        print(state, reward, done, info)

        if done == 1:
            print(f"done @ step {i}")
            break

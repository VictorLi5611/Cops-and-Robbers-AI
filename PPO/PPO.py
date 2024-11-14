import gymnasium as gym
from stable_baselines3 import PPO
import gymnasium as gym
from GridWorld import GridWorld
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import numpy as np
import os, os.path 
import matplotlib.pyplot as plt

env = GridWorld(render_mode=None)
env = DummyVecEnv([lambda: env])
model = PPO("MultiInputPolicy", env, verbose=1)
returns = []
num_episodes = 200
evaluate_every = 20
n_eval = num_episodes // 20

for episode in range(num_episodes):
    model.learn(total_timesteps=100)
    mean_return = np.mean(evaluate_policy(model, env, n_eval_episodes=evaluate_every))
    returns.append(mean_return)

# Can uncomment to show graphs

# plt.plot(np.arange(num_episodes), returns)
# plt.xlabel("Number of evaluation steps")
# plt.ylabel("Average Returns")
# plt.savefig("PPO_plot.png")

model.save("ppo_cops_and_robbers")

# Can uncomment below to show saving and loading

# model = PPO.load("ppo_cops_and_robbers")

# newenv = GridWorld(render_mode="human")
# newenv = DummyVecEnv([lambda: newenv])
# obs = newenv.reset()

# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, done, info = newenv.step(action)
#     newenv.render("human")
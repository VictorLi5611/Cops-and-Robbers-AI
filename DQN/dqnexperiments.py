
import gymnasium as gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import imageio
import os, os.path

from GridWorld import GridWorld
import pygame
from DQN import DQN
from GridWorldConstants import *

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
# BATCH_SIZE = 128
BATCH_SIZE = 256
GAMMA = 0.90
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 50000
TAU = 0.02
LR = 1e-4

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#clear the playback_gifs folder
directory_path = "playback_gifs"
for f in os.listdir(directory_path):
    file_path = os.path.join(directory_path, f)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)





Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)




env = GridWorld(render_mode=None, num_cops=NUM_COPS)
# Get number of actions from gym action space
n_actions = env.action_space.n
observation, info = env.reset()
# state = np.concatenate((observation["vision"], observation["dist"] , observation["bank_pos"]))
state = np.concatenate((observation["vision"] , observation["robber_pos"]))




n_observations = len(state)


policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(1000)

steps_done = 0

frames = []

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

episode_durations = []
episode_rewards = []

def plot_durations(show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Steps')
    plt.ylabel('Rewards')
    # plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    # if less than 100 episodes, plot average of all episodes
    else:
        means = rewards_t.unfold(0, len(rewards_t), 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(len(rewards_t)-1), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def count_files(directory_path):
    # Get the list of files in the directory using relative paths
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    
    # Count the number of files
    num_files = len(files)

    return num_files

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    # print(batch)

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # print("Finished!")

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 12000

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    # change to human for the last 50 episodes
    # if (i_episode == num_episodes - 50):
    #     env.render_mode = "human"
    observation, info = env.reset()
    # state = np.concatenate((observation["vision"], observation["dist"] , observation["bank_pos"]))
    state = np.concatenate((observation["vision"] , observation["robber_pos"]))
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        r = reward
        # state = np.concatenate((observation["vision"], observation["dist"] , observation["bank_pos"]))
        state = np.concatenate((observation["vision"] , observation["robber_pos"]))
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0), action, next_state, reward)
        # print(state)
        # print(next_state)
        # print("End")
        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        
        

        


        if done:
            episode_durations.append(t + 1)
            episode_rewards.append(r)
            r = 0
            plot_durations()
            break
    
    # if env.render_mode == "human":
    #         directory_path = "playback_gifs"
    #         num_files = count_files(directory_path)
    #         file_name = "playback-" + str(num_files) + ".gif"
    #         output_path = os.path.join(directory_path, file_name)

    #         imageio.mimsave(output_path, env.frames)


#frames = [np.rot90(env.frames) for frame in env.frames]
# print(env.frames)
# frames = np.rot90(env.frames, 1)
# frames = np.flipud(env.frames)

# 

#save results to results folder
directory_path = "results"
num_files = count_files(directory_path)
file_name = "result-Tau" + str(TAU) + ".txt"
output_path = os.path.join(directory_path, file_name)
with open(output_path, 'w') as f:
    for item in episode_rewards:
        f.write("%s\n" % item)


print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
env.close()



# import libraries needed
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import csv

env = gym.make("LunarLander-v2")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyper Parameters
state_len = 8
action_len = 4

LearningRate = 1e-4
MaxEpisodeSteps = 300
NumEpisodes = 5000

class DQN(nn.Module):
    def __init__(self, dims):
        super().__init__()
  	
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Create memory and Q-Net
dqn = DQN([8,128,128,4]).to(device)

def get_policy(q_net, state):
    return Categorical(logits = q_net(torch.FloatTensor(state).to(device)))

def select_action(state, q_net):
    action = get_policy(q_net, state).sample().item()
    return action

def calculate_loss(q_net, actions, states, rewards):
    rewards  = torch.FloatTensor(rewards).to(device)
    actions = torch.FloatTensor(actions).to(device)
    probs = get_policy(q_net, states).log_prob(actions)
    loss = -probs * rewards

    return loss.mean()

def backpropagation(loss, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def record_episode(model, episode_idx):
    num_record_per_episode = 1

    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    env = RecordVideo(env, video_folder="luna-lander-pg", name_prefix="training-{}".format(episode_idx), 
                      episode_trigger=lambda x: True)
    env = RecordEpisodeStatistics(env)
    
    for episode_num in range(num_record_per_episode):
        obs, info = env.reset()
    
        episode_over = False
        while not episode_over:
            action = select_action(obs, model)
            obs, reward, terminated, truncated, info = env.step(action)
    
            episode_over = terminated or truncated
    env.close()

# Training Process
optimizer = optim.Adam(dqn.parameters(), lr=LearningRate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NumEpisodes)

# Record for finel evaluation
episode_scores = []
episode_steps = []

# Train one batch
for i in tqdm(range(1, NumEpisodes+1)):
    # collect episode info
    # reset for one new episode
    epi_states = []
    epi_actions = []
    epi_rewards = []

    state, info = env.reset()
    epi_score = 0
    epi_step = 0

    # Train one episode
    while epi_step < MaxEpisodeSteps: 
        # select an action
        action = select_action(state, dqn)
        # receive env update
        new_state, reward, done, truncated, info = env.step(action)
        state = new_state
        epi_step += 1

        epi_score += reward
        epi_states.append(state)
        epi_actions.append(action)

        # if episode done, start over
        if done or epi_step == MaxEpisodeSteps:
            episode_scores.append(epi_score)
            episode_steps.append(epi_step)
            # expand epi_score
            epi_rewards.extend([epi_score] * epi_step)

            # calculate loss
            loss = calculate_loss(dqn, epi_actions, epi_states, epi_rewards)
            # backpropagation
            backpropagation(loss, optimizer)

            break

    scheduler.step()

    if i % 100 == 0:
        # print info of each batch
        print('-' * 80)
        print("Score of episode {}: {}".format(i, np.mean(episode_scores[-100:])))
        print("Lr of batch {}: {}".format(i, optimizer.param_groups[0]['lr']))

        record_episode(dqn, i)


#print(episode_scores)
#print(episode_steps)
torch.save(dqn.state_dict(), './trained_lunar-pg.pth')

with open('lunar-pg-scores.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(episode_scores)

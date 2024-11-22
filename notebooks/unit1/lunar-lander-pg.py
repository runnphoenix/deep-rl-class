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

LearningRate = 2e-3
NumBatches = 300
EpisodesPerBatch = 100
Gamma = 0.99

class PG(nn.Module):
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
pg = PG([8,64,128,32,4]).to(device)

def get_policy(model, state):
    prob = model(torch.FloatTensor(state).to(device))
    return Categorical(logits = prob)

def select_action(state, model):
    action = get_policy(model, state).sample().item()
    return action

def calculate_loss(model, actions, states, rewards):
    rewards  = torch.FloatTensor(rewards).to(device)
    actions = torch.FloatTensor(actions).to(device)
    probs = get_policy(model, states).log_prob(actions)
    loss = -probs * rewards

    return loss.mean()

def backpropagation(loss, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def record_episode(model, episode_idx):
    num_record_per_episode = 3

    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    env = RecordVideo(env, video_folder="lunar-lander-pg", name_prefix="training-{}".format(episode_idx), 
                      episode_trigger=lambda x: True)
    env = RecordEpisodeStatistics(env)
    scores = []
    
    for episode_num in range(num_record_per_episode):
        obs, info = env.reset()
        score = 0
    
        episode_over = False
        while not episode_over:
            action = select_action(obs, model)
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward
    
            episode_over = terminated or truncated
        scores.append(score)

    env.close()
    print("Score evaluated is: {} {}".format(np.mean(scores), scores))
    print("=" * 80)

# Training Process
optimizer = optim.Adam(pg.parameters(), lr=LearningRate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NumBatches)

# Record for final evaluation
episode_scores = []
episode_steps = []

# Train one batch
for i in tqdm(range(1, NumBatches+1)):
    # collect episode info
    # reset for one new episode
    batch_states = []
    batch_actions = []
    batch_rewards = []
    terminate_count = 0

    for j in range(EpisodesPerBatch):
        state, info = env.reset()
        epi_score = 0
        epi_step = 0
        epi_rewards = []
        epi_states = []
        epi_actions = []

        # Train one episode
        while True: 
            # select an action
            action = select_action(state, pg)
            epi_states.append(state)
            epi_actions.append(action)
            # receive env update
            state, reward, terminated, truncated, info = env.step(action)
            epi_score += reward
            epi_rewards.append(reward)

            epi_step += 1

            #print("Truncated") if truncated else None
            done = terminated or truncated
            # if episode done, start over
            if terminated:# Consider Only Complete Trajectories 
                terminate_count += 1
                episode_scores.append(epi_score)
                episode_steps.append(epi_step)

                batch_states.extend(epi_states)
                batch_actions.extend(epi_actions)
                # expand epi_score
                #epi_rewards.extend([epi_score] * epi_step)
                # only consider rewards from now on
                k = epi_step - 2
                while True:
                    epi_rewards[k] = epi_rewards[k] + Gamma * epi_rewards[k+1] 
                    k -= 1
                    if k < 0:
                        break
                batch_rewards.extend(epi_rewards)

                break
            elif truncated:
                break

        if len(batch_actions) == 0:
            continue

    print("Terminated in Batch: {} of {}".format(terminate_count, EpisodesPerBatch))
    # calculate loss
    loss = calculate_loss(pg, batch_actions, batch_states, batch_rewards)
    # backpropagation
    backpropagation(loss, optimizer)

    scheduler.step()

    if i % 10 == 0:
        # print info of each batch
        print('-' * 80)
        print("Score of episode {}: {}".format(i, np.mean(episode_scores[-10 * EpisodesPerBatch:])))
        print("Lr of batch {}: {}".format(i, optimizer.param_groups[0]['lr']))

        record_episode(pg, i)


#print(episode_scores)
#print(episode_steps)
torch.save(pg.state_dict(), './trained_lunar-pg.pth')

with open('lunar-pg-scores.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(episode_scores)

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

env = gym.make("CarRacing-v2", continuous=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyper Parameters
action_len = 5

LearningRate = 1e-5
NumEpisodes = 3000
NumEval = 10
Gamma = 0.99

class Actor(nn.Module):   
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 8, 5, stride=2)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2)
        self.fc1 = nn.Linear(7744, 256)
        self.fc2 = nn.Linear(256, action_len)
 
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((x.shape[0], -1))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 8, 5, stride=2)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2)
        self.fc_v1 = nn.Linear(7744, 256)
        self.fc_v2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((x.shape[0], -1))

        v = F.relu(self.fc_v1(x))
        v = self.fc_v2(v)

        return v

# Create memory and Q-Net
actor = Actor().to(device)
critic = Critic().to(device)

def get_policy(model, state):
    prob = model(torch.FloatTensor(np.array(state)).reshape(-1,3,96,96).to(device))
    return Categorical(logits = prob)

def select_action(state, model):
    action = get_policy(model, state).sample().item()
    return action

def calculate_loss_actor(model, action, state, reward):
    rewards  = torch.FloatTensor([reward]).to(device)
    actions = torch.FloatTensor([action]).to(device)
    probs = get_policy(model, state).log_prob(actions)
    loss = -probs * reward

    return loss.mean()

def calculate_loss_critic(model, state, new_state, reward, done):
    state = torch.FloatTensor(np.array(state)).reshape(-1,3,96,96).to(device)
    new_state = torch.FloatTensor(np.array(new_state)).reshape(-1,3,96,96).to(device)
    loss =  0.5 * F.mse_loss(reward + Gamma * model(new_state) * (1-done),  model(state))

    return loss

def record_episode(actor, episode_idx):
    num_record_per_episode = 1

    env = gym.make("CarRacing-v2", render_mode="rgb_array", continuous=False)
    env = RecordVideo(env, video_folder="car-racing-actor-critic", name_prefix="training-{}".format(episode_idx), 
                      episode_trigger=lambda x: True)
    env = RecordEpisodeStatistics(env)
    scores = []
    
    for episode_num in range(num_record_per_episode):
        obs, info = env.reset()
        score = 0
    
        episode_over = False
        while not episode_over:
            action = select_action(obs, actor)
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward
    
            episode_over = terminated or truncated
        scores.append(score)

    env.close()
    print("Score evaluated is: {} {}".format(np.mean(scores), scores))
    print("=" * 80)

# Training Process
actor_optimizer = optim.Adam(actor.parameters(), lr=LearningRate)
actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(actor_optimizer, T_max=NumEpisodes)
critic_optimizer = optim.Adam(critic.parameters(), lr=LearningRate)
critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(critic_optimizer, T_max=NumEpisodes)

# Record for final evaluation
episode_scores = []
episode_steps = []

# Train one batch
for i in tqdm(range(NumEpisodes)):
    state, info = env.reset()
    epi_score = 0
    epi_step = 0

    # Train one episode
    while True: 
        # select an action
        action = select_action(state, actor)

        # receive env update
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # calculate losses and update 
        critic_loss = calculate_loss_critic(critic, state, new_state, reward, done)
        c_l_d = critic_loss.detach()
        critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        critic_optimizer.step()

        #critic_loss = calculate_loss_critic(critic, state, new_state, reward, done)
        actor_loss = calculate_loss_actor(actor, action, state, c_l_d)
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # update new step
        epi_step += 1
        state = new_state
        epi_score += reward

        # if Done
        if done:# Consider Only Complete Trajectories 
            episode_scores.append(epi_score)
            episode_steps.append(epi_step)

            break

    critic_scheduler.step()
    actor_scheduler.step()

    if i % NumEval == 0:
        # print info of each batch
        print('-' * 80)
        print("Score of episode {}: {}".format(i, np.mean(episode_scores[-NumEval:])))
        print("Lr of batch {}: {}".format(i, actor_optimizer.param_groups[0]['lr']))

        record_episode(actor, i)


torch.save(pg.state_dict(), './trained_car-actor-critic.pth')

with open('car-actor-critic-scores.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(episode_scores)

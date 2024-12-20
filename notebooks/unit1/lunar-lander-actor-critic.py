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

env = gym.make("LunarLander-v2", continuous=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyper Parameters
state_len = 8
action_len = 4

LearningRateActor  = 5e-4
LearningRateCritic = 5e-3
NumEpisodes = 2000
NumEval = 100
BatchSize = 3
Gamma = 0.99

class Net(nn.Module):
    def __init__(self, dims):
        super().__init__()
  	
        layers = []
        for i in range(len(dims)-1):
            linear = nn.Linear(dims[i], dims[i+1])
            torch.nn.init.xavier_uniform_(linear.weight)
            layers.append(linear)
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Create memory and Q-Net
actor = Net([8,64,32,4]).to(device)
critic = Net([8,64,8,1]).to(device)

def get_policy(model, state):
    prob = model(torch.FloatTensor(np.array(state)).to(device))
    return Categorical(logits = prob)

def select_action(state, model):
    action = get_policy(model, state).sample().item()
    return action

def calculate_loss_actor(model, actions, states, delta_v):
    delta_v  = torch.FloatTensor(np.array(delta_v.cpu())).to(device)
    actions = torch.FloatTensor(actions).to(device)
    probs = get_policy(model, states).log_prob(actions)
    loss = -probs * delta_v

    return loss.mean()

def calculate_loss_critic(model, states, new_states, rewards, dones):
    states = torch.FloatTensor(np.array(states)).to(device)
    new_states = torch.FloatTensor(np.array(new_states)).to(device)
    rewards = torch.FloatTensor(np.array(rewards)).to(device)

    current_v = model(states).squeeze()
    dones = torch.LongTensor(np.array(dones)).to(device)
    new_v = rewards + Gamma * model(new_states).squeeze() * (1-dones)

    delta_v = new_v - current_v
    loss =  F.mse_loss(current_v, new_v)

    return delta_v, loss

def record_episode(actor, episode_idx):
    num_record_per_episode = 1

    env = gym.make("LunarLander-v2", render_mode="rgb_array", continuous=False)
    env = RecordVideo(env, video_folder="lunar-lander-actor-critic", name_prefix="training-{}".format(episode_idx), 
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
actor_optimizer = optim.Adam(actor.parameters(), lr=LearningRateActor)
actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(actor_optimizer, T_max=NumEpisodes)
critic_optimizer = optim.Adam(critic.parameters(), lr=LearningRateCritic)
critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(critic_optimizer, T_max=NumEpisodes)

# Record for final evaluation
episode_scores = []
episode_steps = []

# Train
#TODO: Decide which modification works: 1. diff lr for actor and critic 2. new value for critic
for i in tqdm(range(1, NumEpisodes+1)):
    state, info = env.reset()
    epi_score = 0
    epi_step = 0

    # store info and train at the end of an episode
    states = []
    new_states = []
    actions = []
    rewards = []
    dones = []

    # Train one episode
    while True: 
        # select an action
        action = select_action(state, actor)

        # receive env update
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        states.append(state)
        new_states.append(new_state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

        # update new step
        epi_step += 1
        state = new_state
        epi_score += reward

        # if Done
        if done or len(states) == BatchSize:# Consider Only Complete Trajectories 
            # calculate losses and update 
            delta_v, critic_loss = calculate_loss_critic(critic, states, new_states, rewards, dones)
            delta_v_d = delta_v.detach()
            critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            critic_optimizer.step()

            actor_loss = calculate_loss_actor(actor, actions, states, delta_v_d)
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            if done:
                episode_scores.append(epi_score)
                episode_steps.append(epi_step)
                break
            if len(states) == BatchSize:
                states, new_states, actions, rewards, dones = [],[],[],[],[]

    critic_scheduler.step()
    actor_scheduler.step()

    if i % NumEval == 0:
        # print info of each batch
        print('-' * 80)
        print("Score of episode {}: {}".format(i, np.mean(episode_scores[-NumEval:])))
        print("Lr of batch {}: {}".format(i, actor_optimizer.param_groups[0]['lr']))

        record_episode(actor, i)


torch.save(actor.state_dict(), './trained_lunar-actor-critic.pth')

with open('lunar-actor-critic-scores.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(episode_scores)

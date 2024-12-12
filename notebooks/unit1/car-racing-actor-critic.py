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
NumEpisodes = 300
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
        self.fc_v2 = nn.Linear(256, action_len)

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

def calculate_q(model, state, action):
    state = torch.FloatTensor(state).reshape(-1,3,96,96).to(device)
    action = torch.LongTensor([action]).reshape(-1,1).to(device)
    
    return model(state).gather(1, action)

def calculate_loss_critic(model, state, action, reward, new_state, new_action):
    current_q = calculate_q(model, state, action)
    target_q = calculate_q(model, new_state, new_action)

    target = target_q - current_q

    return target


def record_episode(actor, episode_idx):
    num_record_per_episode = 2

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
    epi_rewards = []

    # Train one episode
    while True: 
        # select an action
        action = select_action(state, actor)
        current_q = calculate_q(critic, state, action)

        # receive env update
        new_state, reward, terminated, truncated, info = env.step(action)

        #print("Truncated") if truncated else None
        done = terminated or truncated
        if done:# Consider Only Complete Trajectories 
            episode_scores.append(epi_score)
            episode_steps.append(epi_step)

            break

        # update parameters of actor
        loss_actor = calculate_loss_actor(actor, action, state, current_q)
        actor_optimizer.zero_grad()
        loss_actor.backward(retain_graph=True)
        actor_optimizer.step()

        # select a new action by actor
        new_action = select_action(new_state, actor)

        # update critic
        next_q = calculate_q(critic, new_state, new_action)
        loss_critic = next_q - current_q

        critic_optimizer.zero_grad()
        loss_critic.backward()
        critic_optimizer.step()

        # update new step
        epi_step += 1
        state = new_state
        epi_score += reward
        epi_rewards.append(reward)


    critic_scheduler.step()
    actor_scheduler.step()

    if i % 1 == 0:
        # print info of each batch
        print('-' * 80)
        print("Score of episode {}: {}".format(i, np.mean(episode_scores[-1])))
        print("Lr of batch {}: {}".format(i, actor_optimizer.param_groups[0]['lr']))

        record_episode(actor, i)


#print(episode_scores)
#print(episode_steps)
torch.save(pg.state_dict(), './trained_car-actor-critic.pth')

with open('car-actor-critic-scores.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(episode_scores)

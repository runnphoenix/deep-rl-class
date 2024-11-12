# import libraries needed
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import csv

env = gym.make("CarRacing-v2", continuous=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyper Parameters
action_len = 5

Tau = 4
BatchSize = 128
MemoryCapacity = BatchSize * 20
LearningRate = 3e-4
NumEpisodes = 2000
MaxEpisodeSteps = 1000
EvalSteps = NumEpisodes // 200
Epsilon = 1
Gamma = 0.99

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2)
        self.fc1 = nn.Linear(5408, 256)
        self.fc2 = nn.Linear(256, action_len)
  	

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x)) # 128,342,13,13
        x = x.view((x.shape[0], -1))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #print(x.shape)
        return x

class Memory():
    def __init__(self, batch_size, capacity):
        self.states     = np.zeros([capacity, 96,96,3], dtype=np.float32)
        self.actions    = np.zeros([capacity], dtype=int)
        self.rewards    = np.zeros([capacity], dtype=np.float32)
        self.new_states = np.zeros([capacity, 96,96,3], dtype=np.float32)
        self.done       = np.zeros([capacity], dtype=int)

        self.batch_size = batch_size
        self.capacity = capacity
        self.ptr, self.size = 0, 0

    def length(self):
        return self.size

    def push(self, state, act, reward, new_state, done):
        self.states[self.ptr]     = state
        self.actions[self.ptr]    = act
        self.rewards[self.ptr]    = reward
        self.new_states[self.ptr] = new_state
        self.done[self.ptr]       = done

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self):
        idx = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(states     = self.states[idx],
                    actions    = self.actions[idx],
                    rewards    = self.rewards[idx],
                    new_states = self.new_states[idx],
                    done       = self.done[idx])


def select_action(epsilon, state, q_net, env, training):
    if np.random.random() < epsilon and training:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            action = q_net(torch.FloatTensor(state).reshape(1,3,96,96).to(device)).argmax().cpu().item()

    return action

def step(env, state, action, memory, training):
    new_state, reward, done, _, info = env.step(action)

    if training:
        memory.push(state, action, reward, new_state, done)

    return new_state, reward, done

def calculate_loss(q_net, target_q_net, samples, gamma):
    states =  torch.FloatTensor(samples['states']).reshape(-1,3,96,96).to(device)
    actions =  torch.LongTensor(samples['actions']).reshape(-1,1).to(device)
    rewards =  torch.FloatTensor(samples['rewards']).reshape(-1,1).to(device)
    new_states =  torch.FloatTensor(samples['new_states']).reshape(-1,3,96,96).to(device)
    dones =  torch.LongTensor(samples['done']).reshape(-1,1).to(device)

    current_q = q_net(states).gather(1, actions)
    #target_q = target_q_net(new_states).max(dim=1, keepdim=True)[0].detach()
    next_actions = dqn(states).argmax(1).reshape(-1,1)
    target_q = target_q_net(new_states).gather(1, next_actions)

    mask = 1 - dones
    target = (rewards + gamma * target_q * mask).to(device)
    #print([q.cpu().item() for q in current_q], [q.cpu().item() for q in target])

    return F.mse_loss(current_q, target)

def backpropagation(loss, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def record_episode(model, episode_idx):
    num_record_per_episode = 2

    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    env = RecordVideo(env, video_folder="luna-lander-agent", name_prefix="training-{}".format(episode_idx), 
                      episode_trigger=lambda x: True)
    env = RecordEpisodeStatistics(env)
    
    for episode_num in range(num_record_per_episode):
        obs, info = env.reset()
    
        episode_over = False
        while not episode_over:
            action = select_action(Epsilon, obs, model, env, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
    
            episode_over = terminated or truncated
    env.close()

# Create memory and Q-Net
memory = Memory(BatchSize, MemoryCapacity)
dqn = DQN().to(device)
target_dqn = DQN().to(device)

# Training Process
optimizer = optim.Adam(dqn.parameters(), lr=LearningRate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NumEpisodes)

# collect scores
scores = []
steps  = []

for i in tqdm(range(1, NumEpisodes+1)):
    state, info = env.reset()
    score = 0
    epi_step = 0

    while not epi_step > MaxEpisodeSteps:
        # select an action
        action = select_action(Epsilon, state, dqn, env, training=True)

        # receive env update
        # push to memory
        new_state, reward, done = step(env, state, action, memory, training=True)
        state = new_state
        score += reward

        # train if there are enough memories
        if memory.length() > BatchSize:
            # sample a batch
            train_samples = memory.sample()

            # calculate loss
            loss = calculate_loss(dqn, target_dqn, train_samples, Gamma)

            # backpropagation
            backpropagation(loss, optimizer)

            # update dqn_target
            if i % Tau == 0:
                target_dqn.load_state_dict(dqn.state_dict())

        # if episode done, start over
        if done or epi_step == MaxEpisodeSteps:
            scores.append(score)
            steps.append(epi_step)

            if i % EvalSteps == 0:
                print('-' * 80)
                print("Score of episode {}-{}: {}".format(i-EvalSteps, i, np.mean(scores[-EvalSteps:])))
                print("Lr of episode {}: {}".format(i, optimizer.param_groups[0]['lr']))
                print("Epsilon of episode {}: {}".format(i, Epsilon))

                record_episode(dqn, i)
            break

        epi_step += 1

    # decrease Epsilon
    Epsilon = 1 - (i / NumEpisodes)
    scheduler.step()

torch.save(dqn.state_dict(), './trained_dqn.pth')

with open('scores.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(scores)

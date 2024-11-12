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

#env = gym.make("LunarLander-v2")
env = gym.make("BipedalWalker-v3")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyper Parameters
state_len = 24
action_len = 4

Tau = 4
BatchSize = 128
MemoryCapacity = BatchSize * 20
LearningRate = 3e-4
NumEpisodes = 2000
MaxEpisodeSteps = 1000
EvalSteps = NumEpisodes // 20
Epsilon = 1
Gamma = 0.99

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

class Memory():
    def __init__(self, state_len, action_len, batch_size, capacity):
        self.states     = np.zeros([capacity, state_len], dtype=np.float32)
        self.actions    = np.zeros([capacity, action_len], dtype=np.float32)
        self.rewards    = np.zeros([capacity], dtype=np.float32)
        self.new_states = np.zeros([capacity, state_len], dtype=np.float32)
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
            action = q_net(torch.FloatTensor(state).to(device)).argmax().cpu().item()

    return action

def step(env, state, action, memory, training):
    new_state, reward, done, _, info = env.step(action)

    if training:
        memory.push(state, action, reward, new_state, done)

    return new_state, reward, done

def calculate_loss(q_net, target_q_net, samples, gamma):
    states =  torch.FloatTensor(samples['states']).to(device)
    actions =  torch.LongTensor(samples['actions']).reshape(-1,1).to(device)
    rewards =  torch.FloatTensor(samples['rewards']).reshape(-1,1).to(device)
    new_states =  torch.FloatTensor(samples['new_states']).to(device)
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
memory = Memory(state_shape, action_shape, BatchSize, MemoryCapacity)
dqn = DQN([8,64,64,32,4]).to(device)
target_dqn = DQN([8,64,64,32,4]).to(device)

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

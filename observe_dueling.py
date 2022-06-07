import os
import warnings
warnings.filterwarnings("ignore")
from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random
import time
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.vec_env import SubprocVecEnv
from baselines_wrappers import DummyVecEnv, Monitor
from pytorch_wrappers import make_atari_deepmind, BatchedPytorchFrameStack, PytorchLazyFrames

import msgpack
from msgpack_numpy import patch as msgpack_numpy_patch
msgpack_numpy_patch()

GAMMA=0.99
BATCH_SIZE=32
BUFFER_SIZE=int(1e6)
MIN_REPLAY_SIZE=50000
EPSILON_START=1.0
EPSILON_END=0.1
EPSILON_DECAY=int(1e6)
NUM_ENVS = 4    #create 4 env at start for training
TARGET_UPDATE_FREQ=10000 // NUM_ENVS
LR = 5e-5   #learning-rate
SAVE_INTERVAL = 10000   #save model after each 10000 steps
LOG_INTERVAL = 1000     #writing logs after each 1000 steps  

LOG_DIR = './logs/montezuma_dueling'

SAVE_PATH = './montezuma_dueling.pt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def nature_cnn(observation_space, depths=(32, 64, 64), final_layer=512):
    n_input_channels = observation_space.shape[0]

    cnn = nn.Sequential(
        nn.Conv2d(n_input_channels, depths[0], kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(depths[0], depths[1], kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(depths[1], depths[2], kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten())

    # Compute shape by doing one forward pass
    with torch.no_grad():
        n_flatten = cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

    out = nn.Sequential(cnn, nn.Linear(n_flatten, final_layer), nn.ReLU())

    return out

class Network(nn.Module):
    def __init__(self, env, device):
        super().__init__()
        self.device = device
        self.num_actions = env.action_space.n
        self.input_dim = env.observation_space.shape
        self.output_dim = self.num_actions
        
        
        self.net = nn.Sequential(
            nn.Conv2d(env.observation_space.shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc_input_dim = self.feature_size()
        
        self.value_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
        )

    def forward(self, state):
        features = self.net(state)
        features = features.view(features.size(0), -1)
        # features = torch.transpose(features, 0, 1)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())
        
        return qvals

    def feature_size(self):
        return self.net(torch.autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)

    def act(self, obses, epsilon):

        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=self.device)
        q_values = self(obses_t)

        max_q_indices = torch.argmax(q_values, dim=1)
        actions = max_q_indices.detach().tolist()

        # Epsilon greedy policy
        for i in range(len(actions)):
            random_sample = random.random()
            if random_sample <= epsilon:
                actions[i] = random.randint(0, self.num_actions - 1)

        return actions


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

make_env = lambda: make_atari_deepmind('Breakout-v0', scale_values=True)

vec_env = DummyVecEnv([make_env for _ in range(1)])

env = BatchedPytorchFrameStack(vec_env, k=4)

net = Network(env, device)
net = net.to(device)

# net.load('./atari_model.pt')
net.load_state_dict(torch.load('models/breakout_dueling.pt'))
net.eval()
print("Load xong r ne")
obs = env.reset()
beginning_episode = True
for t in itertools.count():
    if isinstance(obs[0], PytorchLazyFrames):
        act_obs = np.stack([o.get_frames() for o in obs])
        action = net.act(act_obs, 0.0)
    else:
        action = net.act(obs, 0.0)

    if beginning_episode:
        action = [1]
        beginning_episode = False

    obs, rew, done, _ = env.step(action)
    # env.render()
    time.sleep(0.02)

    if done[0]:
        obs = env.reset()
        beginning_episode = True
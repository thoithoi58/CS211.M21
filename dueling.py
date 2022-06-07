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
    def __init__(self, env):
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

    def compute_loss(self, transitions, target_net):
        """
        Function to get loss from online network
        """
        obses = [t[0] for t in transitions]
        actions = np.asarray([t[1] for t in transitions])
        rewards = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_obses = [t[4] for t in transitions]

        if isinstance(obses[0], PytorchLazyFrames):
            obses = np.stack([o.get_frames() for o in obses])
            new_obses = np.stack([o.get_frames() for o in new_obses])
        else:
            obses = np.asarray(obses)
            new_obses = np.asarray(new_obses)

        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(-1)
        rews_t = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(-1)
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32, device=device)

        # Compute targets
        target_q_values = target_net(new_obses_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        targets = rews_t + GAMMA * max_target_q_values * (1 - dones_t)

        # Compute loss
        q_values = self(obses_t)

        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
        
        loss = nn.functional.smooth_l1_loss(action_q_values, targets)
        
        return loss

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)
    
    def load(self, load_path):
        Network.load_state_dict(torch.load(load_path))
        Network.eval()

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    make_env = lambda: Monitor(make_atari_deepmind('MontezumaRevenge-v0', scale_values=True), allow_early_resets=True)

    # vec_env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])
    vec_env = SubprocVecEnv([make_env for _ in range(NUM_ENVS)])

    env = BatchedPytorchFrameStack(vec_env, k=4)

    replay_buffer = deque(maxlen=BUFFER_SIZE)
    epinfos_buffer = deque([], maxlen=100)

    episode_count = 0

    summary_writer = SummaryWriter(LOG_DIR)

    online_net = Network(env)
    target_net = Network(env)

    if os.path.exists(SAVE_PATH):
        print('Co ne!')
        online_net.load_state_dict(torch.load(SAVE_PATH))
        online_net.eval()

    online_net = online_net.to(device)
    target_net = target_net.to(device)

    target_net.load_state_dict(online_net.state_dict())

    optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)

    # Initialize Replay Buffer
    obses = env.reset()
    for _ in range(MIN_REPLAY_SIZE):
        actions = [env.action_space.sample() for _ in range(NUM_ENVS)]

        new_obses, rews, dones, _ = env.step(actions)

        for obs, action, rew, done, new_obs in zip(obses, actions, rews, dones, new_obses):
            transition = (obs, action, rew, done, new_obs)
            replay_buffer.append(transition)

        obses = new_obses

    # Main Training Loop
    obses = env.reset()

    for step in itertools.count(110000):
        epsilon = np.interp(step * NUM_ENVS, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

        rnd_sample = random.random()

        if isinstance(obses[0], PytorchLazyFrames):
            act_obses = np.stack([o.get_frames() for o in obses])
            actions = online_net.act(act_obses, epsilon)
        else:
            actions = online_net.act(obses, epsilon)

        new_obses, rews, dones, infos = env.step(actions)

        for obs, action, rew, done, new_obs, info in zip(obses, actions, rews, dones, new_obses, infos):
            transition = (obs, action, rew, done, new_obs)
            replay_buffer.append(transition)

            if done:
                epinfos_buffer.append(info['episode'])
                episode_count += 1

        obses = new_obses

        # Start Gradient Step
        transitions = random.sample(replay_buffer, BATCH_SIZE)
        loss = online_net.compute_loss(transitions, target_net)

        # Gradient Descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update Target Network
        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())

        # Logging
        if step % LOG_INTERVAL == 0:
            rew_mean = np.mean([e['r'] for e in epinfos_buffer]) or 0
            len_mean = np.mean([e['l'] for e in epinfos_buffer]) or 0

            print()
            print('Step:', step)
            print('Avg Rew:', rew_mean)
            print('Avg Ep. Len:', len_mean)
            print('Episodes', episode_count)
            summary_writer.add_scalar('AvgRew', rew_mean, global_step=step)
            summary_writer.add_scalar('Avg Ep. Len', len_mean, global_step=step)
            summary_writer.add_scalar('Episode', episode_count, global_step=step)

        # Save
        if step % SAVE_INTERVAL == 0 and step !=0:
            print('Saving...')
            online_net.save(SAVE_PATH)
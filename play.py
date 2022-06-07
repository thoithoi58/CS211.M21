import gym
from gym.utils.play import play

env = gym.make('Breakout-v0', render_mode='human')
for _ in range(100):
    env.reset()
    action = env.action_space.sample()
    obs, rew, done, _ = env.step(action)
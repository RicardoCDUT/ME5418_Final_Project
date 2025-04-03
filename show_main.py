import random
import time

import gym
import numpy as np
import torch
from gym import register

import memory_buffer
from agent import Agent
from qNetwork import QNetwork

register(
    id='2denv-v0', # Environment ID.
    entry_point='environment:envGym', # Environment ID.
)
# Create an instance of the environment.
env = gym.make('2denv-v0', render="rgb_array")
state_space = env.observation_space # Get the observation space of the environment.
action_space = env.action_space  # Get the action space of the environment.
# Create an instance of the memory buffer to store experiences, using CUDA as the device.
memory = memory_buffer.MemoryBuffer(100000, "cuda")
# List to store frames for rendering.
frames = []
t = time.time() # Record the current time.
# Create an agent for the environment, passing state space and action space sizes, along with the memory buffer.
agent = Agent(state_space.shape[0], action_space.shape[0], memory)
# Load the pre-trained agent parameters from a saved model file.
agent.load_state_dict(torch.load('weight/train1/epoch816.448reward.agent.pth'))
agent = agent.cuda()
scores = []
# List to store the rewards from each test episode.
for i in range(10):
    done = False
    state = env.reset()
    # Reset the 'done' flag for each episode.
    score = 0
    # Initialize the score for the current episode.
    while not done:
        env.render(mode='human')
        # Render the environment to visualize it.
        action_prob = agent.get_action(torch.tensor(state).float().reshape(1, -1).to("cuda"))
        # Render the environment to visualize it.
        action = [int(x.max(1)[1]) for x in action_prob]
        # Perform the selected action in the environment.
        next_state, reward, done, info = env.step(np.array([np.linspace(-1., 1., 4)[x] for x in action]))
        # Accumulate the reward received from the environment.
        score += reward
        # Update the state with the new state obtained.
        state = next_state
    scores.append(score)
    print(f"test {i +1} : reward {score}")

print(f"max score: {max(scores)}")
print(f"mean score: {np.mean(scores)}")
env.close()

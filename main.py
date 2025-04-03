import datetime
import pickle
import random
import time

import gym
import imageio
import numpy as np
import torch
from gym import register
from matplotlib import pyplot as plt
import math
import memory_buffer
from agent import Agent



def save_model(model, path):
    torch.save(model.state_dict(), path)
    # Save the parameters of the model to the specified path.


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    # Load model parameters from the specified path and return the loaded model.
    return model
def two_decimal(n):
    return round(n, 2)



register(
    id='2denv-v0',  # Environment ID.
    entry_point='environment:envGym',  # Entrance of environment.
)
save = True
if save:
    env = gym.make('2denv-v0', render="rgb_array")
else:
    env = gym.make('2denv-v0', render="speed")
state_space = env.observation_space
# Obtain the state space of the environment.
action_space = env.action_space
# Obtain the action space of the environment.
memory = memory_buffer.MemoryBuffer(100000, "cuda")

agent = Agent(state_space.shape[0], action_space.shape[0], memory, lr=1e-4).cuda()
# Initialize the agent with state space and action space.

max_score = -10000
# Initialize the maximum score to a smaller value.
max_game = 0
# The round that achieved the highest score during initialization is 0.

scores = []
# Used to store the scores for each round.

start = datetime.datetime.now()
# Record the start time of training.
actions = []
train_time = time.time()
times = []

for game in range(1000):
    # Play 500 rounds of games.
    done = False
    # At the beginning of each round, set the end flag to False.
    score = 0
    # The initial score for each round is 0.
    state = env.reset()
    # Reset the environment to obtain the initial state.
    game_actions = []
    # Used to store the action sequence for the current game round.
    episode_start = time.time()
    # Record the start time of the current round of the game.
    frames = []
    # Game Cycle
    while not done:
        if save:
            frame = env.render(mode='rgb_array')
            frames.append(frame)
        else:
            env.render(mode="speed")
        #The smaller the epsilon, the lower the probability of randomly selecting an action, but it is not zero
        # It can be said that the probability of the agent taking risks in the early stages will be higher,
        # and the probability of taking risks in the later stages will decrease.
        # However, in order to avoid falling into local optima, there still needs to be some random actions in the later stages
        # Continuously looping while the game is still ongoing.
        epsilon = max(0.01, 0.9 - 0.01 * (game / 10))
        if epsilon > random.random():
            action = random.sample(range(0, 4), 4)  # Randomly select actions (exploration).
        else:
            action_prob = agent.get_action(torch.tensor(state).float().reshape(1, -1).to("cuda"))
            action = [int(x.max(1)[1]) for x in action_prob] # Select actions based on the agent's policy (exploitation).
            actions.append(action)
        # Execute the action in the environment and observe the result.
        next_state, reward, done, info = env.step(np.array([np.linspace(-1., 1., 4)[x] for x in action]))
        # agent.step(state, action, reward, next_state, done)
        score += float(reward)  # Accumulate the reward for the game.
        done = 1 if done else 0  # Convert the done flag to an integer.
        memory.put(state, action, reward, next_state, done)  # Accumulate the reward for the game.
        if len(memory) > 1000:
            agent.train_mode(gamma=0.99) # Train the agent using the replay buffer when enough experiences are available.

        state = next_state  # Update the current state.
        if score <= -115.0: # Stop the game early if the score is too low.
            break
    scores.append(two_decimal(score)) # Stop the game early if the score is too low.
    if score == max(scores): # Save the model if the score is the highest so far.
        if save:
            imageio.mimsave(f'gif/epoch{game}.{int(score)}reward.mp4', frames, fps=60) # Save the gameplay.
        save_model(agent, f"weight/epoch{game}.{int(score)}reward.agent.pth")
    duration = two_decimal(time.time() - episode_start) # Record the time-normalized score.
    times.append(score / duration) # Record the time-normalized score.
    print(score / duration)
    print(game, ":", score, ":time:", duration)

# Drawing picture
def draw_pic(array, title, x_t, y_t):
    y = array
    x = list(range(len(array)))
    plt.figure(figsize=(7.2,4.8))
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(x_t)
    plt.ylabel(y_t)
    plt.show()

def plot_q_values(agent):
    train_num = range(1, len(agent.q_values) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(train_num, agent.q_values, label='Q Network')
    plt.plot(train_num, agent.target_q_values, label='Target Q Network')
    plt.xlabel('train_num')
    plt.ylabel('Q Values')
    plt.title('Q Values Over Time')
    plt.legend()
    plt.show()




print(f"train time: {time.time() - train_time}")
# agent = agent.cpu()
# loss_data = agent.loss_data
plot_q_values(agent)
draw_pic(scores, "dqn", "epoch", "reward")
draw_pic(times, "dqn", "epoch", "score / time(s)")
save_model(agent, "weight/agent.pth")

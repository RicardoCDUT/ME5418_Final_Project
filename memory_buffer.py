import collections
import random

import numpy as np
import torch
class MemoryBuffer:
    def __init__(self, capacity, device):
        # Set the capacity of the buffer (maximum number of stored experiences).
        self.capacity = capacity
        # Use deque as a buffer and set the maximum length to the capacity value.
        # Once the capacity is reached, old experiences will be automatically removed.
        self.buffer = collections.deque(maxlen=capacity)
        self.device = device
        # CPU or GPU

    def put(self, state, action, reward, next_state, done):
        # Adding experience.
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=32):
        mini_batch = random.sample(self.buffer, batch_size)
        # Randomly sample a batch of experiences from the buffer, with the specified batch size.
        state_lst, reward_lst, next_state_lst, done_mask_lst = [], [], [], []
        #  Initialize lists to store decomposed components of the sampled experiences.
        actions_lst = [[], [], [], []]
        for transition in mini_batch:
            # Iterate through the sampled batch.
            state, actions, reward, next_state, done_mask = transition
            # Decompose each transition tuple into its components.
            state_lst.append(state)
            # Iterate through action dimensions and store actions into separate lists.
            for transition in mini_batch:
                # Decompose each transition tuple into its components.
                state, actions, reward, next_state, done_mask = transition
                # Append the state to the state list.
                state_lst.append(state)
                for idx in range(len(actions_lst)):
                    # Append the reward, next state, and done flag to their respective lists.
                    actions_lst[idx].append(actions[idx])
                reward_lst.append([reward])
                next_state_lst.append(next_state)
                done_mask_lst.append([done_mask])
            # Convert the lists to NumPy arrays for efficient processing.
            state_array = np.array(state_lst)
            next_state_array = np.array(next_state_lst)
            # Convert the actions to PyTorch tensors, ensuring they are on the specified device.
            actions_lst = [torch.tensor(x, dtype=torch.float).to(self.device) for x in actions_lst]
            # Convert other components to PyTorch tensors and transfer them to the specified device.
            return torch.tensor(state_array, dtype=torch.float).to(self.device), \
                actions_lst, torch.tensor(reward_lst).to(self.device), \
                torch.tensor(next_state_array, dtype=torch.float).to(self.device), \
                torch.tensor(done_mask_lst).to(self.device)
        # sample = random.sample(self.buffer, batch_size)
        # states, actions, rewards, next_states, dones = zip(*sample)
        # # Decompress the sampled data separately.
        # states = torch.tensor(states).float().to(self.device)
        # # Convert the state, action, reward, next state, and end flag into PyTorch tensors and transfer them to the specified device.
        # actions = torch.stack(actions).long().to(self.device)
        # # Convert the state to a floating-point tensor and transfer it to the specified device.
        # rewards = torch.from_numpy(np.array(rewards, dtype=np.float32).reshape(-1, 1)).to(self.device)
        # # Stack actions into a long tensor and transfer it to a specified device.
        # next_states = torch.tensor(next_states).float().to(self.device)
        # # The next state is converted to a floating-point tensor and transferred to the device.
        # dones = torch.from_numpy(np.array(dones, dtype=np.uint8).reshape(-1, 1)).float().to(self.device)
        # # Convert the end flag to an unsigned integer, then to a floating-point tensor, and transfer it to the device.


        # Return the decomposed experience sample.

    def __len__(self):
        # Return the current number of experiences in the buffer.
        return len(self.buffer)
        # Return the buffer length to determine if the sampling requirements are met.

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from qNetwork import QNetwork

class Agent(nn.Module):
    def __init__(self, state_space, action_space, memory,lr=1e-4):
        # The size of the input state，
        # Number of executable actions，
        # Memory buffer for experience replay。
        super(Agent, self).__init__()
        self.device = "cuda"
        # Set the device to CUDA to use Cpu accelerated computing.
        self.q = QNetwork(state_space, action_space).to(self.device)
        # Instantiate the main Q network.
        self.target_q = QNetwork(state_space, action_space).to(self.device)
        # Instantiate the target network.
        self.target_q.load_state_dict(self.q.state_dict())
        # Copy the parameters of self. q to the target network.
        self.batch_size = 64
        # The batch size for training is 64.
        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        # Used to update the parameters of self. q, with a learning rate of 1e-4.
        self.memory = memory
        # Save experience replay buffer.
        self.epoch = 0
        # Used for counting training steps.
        self.loss_data = []
        self.q_values = []
        self.target_q_values = []

    def train_mode(self, gamma):
        # Defined the training process of Q network.
       # state, actions, reward, new_state, done_mask = self.memory.sample(self.batch_size)
        # Randomly sample a batch of empirical data from the buffer
        state, actions, reward, next_state, done_mask = self.memory.sample(self.batch_size)
        actions = torch.stack(actions).transpose(0, 1).unsqueeze(-1)
        # Actions are stacked and shaped to facilitate tensor operations.
        done_mask = torch.abs(done_mask - 1)
        # Indicates whether the current batch is in a terminated state.
        cur_actions = self.q(state)
        # Calculate the Q value of the current state using the main network..
        cur_actions = torch.stack(cur_actions).transpose(0, 1)
        # print(actions.shape,"\nasdasdasdasdasd\n", cur_actions)
        cur_actions = cur_actions.gather(2, actions.long()).squeeze(-1).float()
        # Delete the last dimension and ensure that the data type is consistent with .float()..

        target_cur_actions = self.target_q(next_state)
        # Use the target network to calculate the Q-value of NextStatus.
        target_cur_actions = torch.stack(target_cur_actions).transpose(0, 1)
        target_cur_actions = target_cur_actions.max(-1, keepdim=True)[0]
        target_action = (done_mask * gamma * target_cur_actions.mean(1) + reward)
        # Calculate the impact of future rewards.
        loss = F.mse_loss(cur_actions, target_action.repeat(1, 4).float())
        # Calculate the target Q value based on the Bellman equation,
        # and use the discount factor gamma to adjust the reward.
        self.q_values.append(cur_actions.mean().item())
        self.target_q_values.append(target_action.mean().item())
        self.optimizer.zero_grad()
        # Clear gradient.
        loss.backward()
        # Calculate gradient.
        self.optimizer.step()
        # Update network parameters based on gradients.
        self.loss_data.append(loss.cpu().item())

        self.epoch += 1
        if self.epoch % 1000 == 0:
            self.target_q.load_state_dict(self.q.state_dict())
            # Update the target network every two training sessions
        return loss
        # Return the loss value for each training step.

    def get_action(self, state):
        # Accept a state as input and output a Q value.
        return self.q(state)

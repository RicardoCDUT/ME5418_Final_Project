import torch.nn as nn
import torch.nn.functional as F
# input_dim = 10  # four joint angular velocities, four motor velocities, and the main body's horizontal and vertical velocities.
# hidden_dims = [128, 64]
# output_dim = 4 # four motor velocities.
class QNetwork(nn.Module):
    def __init__(self, state_space: int, action_num: int):
        # The input layer is 10 dimensions,
        # including four joint angular velocities, four motor velocities, and the main body's horizontal and vertical velocities.
        super(QNetwork, self).__init__()
        self.linear_1 = nn.Linear(state_space, 128)
        # The first layer is the linear fully that connected layer,
        # maps the original 10 dimensional data onto 128 dimensions.

        self.linear_2 = nn.Linear(128, 64)
        # The second layer reduces the data from 128 dimensions to 64 dimensions and extracts higher-level features.
        # The positive and negative values of the input data represent the motor speed's direction.
        # The Leaky ReLU mitigates the issue of neuron dead zones at the early training stages by handling negative values with a non-zero gradient set.

        # Stack layers using Sequential.
        self.actions = [nn.Sequential(nn.Linear(64, 32),
                                      nn.ReLU(),
        # The positive and negative values of the input data represent the motor speed's direction.
        # The Leaky ReLU mitigates the issue of neuron dead zones at the early training stages by handling negative values with a non-zero gradient set.
                                      nn.Linear(32, 4)
                                      ) for _ in range(action_num)]

        # # The last layer, Linear, directly outputs the result of linear combination.
        # Output the four motor velocities.

        self.actions = nn.ModuleList(self.actions)

        self.value = nn.Sequential(nn.Linear(64, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, 1)
                                )
        # In the early stages, the parameter update frequency of the Q-network is relatively low,
        # making it challenging to quickly adjust to a reasonable range.
        # The RELU is smooth in the negative region, helps alleviate the vanishing gradient problem and facilitates gradient flow.



    def forward(self, x):
        x = F.relu(self.linear_1(x))
        encoded = F.relu(self.linear_2(x))
        actions = [x(encoded) for x in self.actions]
        value = self.value(encoded)
        for i in range(len(actions)):
            actions[i] = actions[i] - actions[i].max(-1)[0].reshape(-1, 1)
            actions[i] += value
        return actions



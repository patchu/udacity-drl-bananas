import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, num_filters=[64, 128, 256], fc_layers=[64, 64]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed (not used)
            num_filters[0] (int): Number of nodes in first hidden layer
            num_filters[1] (int): Number of nodes in second hidden layer
            num_filters[2] (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()

        # if num_filters[2] > 0, then add a third layer, otherwise only use 2 layers
        self.third_layer = False
        self.last_filter_num = num_filters[1]

        # after 2 maxpool layers, the dimensions are now 22x22
        self.curr_dim = 84


        if num_filters[2] > 0:
            self.last_filter_num = num_filters[2]
            self.third_layer = True

        # disabling seed so we can get random behavior
        # self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, num_filters[0], kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(num_filters[0]),
                        nn.ReLU(),
                        # nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 1)
                    )

        self.conv2 = nn.Sequential(
                        nn.Conv2d(num_filters[0], num_filters[1], kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(num_filters[1]),
                        nn.ReLU(),
                        # nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 1)
                    )

        if self.third_layer:
            self.conv3 = nn.Sequential(
                            nn.Conv2d(num_filters[1], num_filters[2], kernel_size = 3, stride = 1, padding = 1),
                            nn.BatchNorm2d(num_filters[2]),
                            nn.ReLU()
                        )

        if fc_layers[1] > 0:
            self.fc = nn.Sequential(
                            nn.Linear(self.curr_dim * self.curr_dim * self.last_filter_num, fc_layers[0]),
                            # nn.BatchNorm1d(fc_layers[0]),
                            nn.ReLU(),
                            nn.Linear(fc_layers[0], fc_layers[1]),
                            nn.Linear(fc_layers[1], action_size),
                        )
        else:
            self.fc = nn.Sequential(
                            nn.Linear(self.curr_dim * self.curr_dim * self.last_filter_num, fc_layers[0]),
                            # nn.BatchNorm1d(fc_layers[0]),
                            nn.ReLU(),
                            nn.Linear(fc_layers[0], action_size),
                        )


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.conv1(state)
        x = self.conv2(x)

        if self.third_layer:
            x = self.conv3(x)

        # print('last filter num', self.last_filter_num)
        # print('x shape', x.shape)
        # # x = x.flatten()
        # # print('flatten shape', x.shape)

        # curr_dim = 22
        # x = nn.Linear(curr_dim * curr_dim * self.last_filter_num, 64)(x.flatten())
        # x = nn.ReLU()(x)

        x = x.reshape((-1, self.curr_dim * self.curr_dim * self.last_filter_num))
        # print('x shape', x.shape)

        x = self.fc(x)
        return x

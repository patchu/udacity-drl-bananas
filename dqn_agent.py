import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 6        # how often to update the network
NN_LOOPS = 1            # how many times to loop the NN for every update

# use Macbook M1 GPU if available
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")

print('torch device = ', device)

def convert_state_not_used(state):
    """Convert state from a 37-length vector to 11-length"""

    # state is a 37-length vector, consisting of 7 groups of 5 "rays" and 2 velocity floats
    # each ray group of 5 denotes a ray emanating outward from agent
    #       in degrees of [20, 90, 160, 45, 135, 70, 110]
    #    The 5 values are a one-hot encoding of [yellow banana, wall, blue banana, agent]
    #    The last value is a distance from agent to the object that's one-hot encoded
    # This function converts this set of 5 values into 2:
    #       [yellow banana, blue banana], and the values are the distances
    # Then, the 7 rays are reduced to 5: the 20 degree and 160 degree values are discarded
    # Lastly, only the forward velocity is saved, left/right velocity is discarded

    # indexes of the rays values to save
    # drop 20-degree and 160-degree indexes: 0 and 2 index
    take_indices = [1, 3, 4, 5, 6]

    # reshape rays into a 7x5 matrix
    rays = state.squeeze()[:-2].reshape(-1,5)

    # 16-length version, all velocities and angles
    return torch.Tensor(np.hstack((
        # yellow values are first column * distance
        rays[:,0] * rays[:,4],
        # blue values are first column * distance
        rays[:,2] * rays[:,4],
        # forward velocity
        state[-2:]
    )))

    # returned array (length=11):
    #       5x distance to yellow bananas in 5 directions
    #       5x distance to blue bananas in 5 directions
    #       forward velocity
    return torch.Tensor(np.hstack((
        # yellow values are first column * distance
        np.take(rays[:,0] * rays[:,4], take_indices),
        # blue values are first column * distance
        np.take(rays[:,2] * rays[:,4], take_indices),
        # forward velocity
        state[-1]
    )))


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size

        # disabling seed so we can get random behavior
        # self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(self.state_size, action_size, seed, 64, 64, 64).to(device)
        self.qnetwork_target = QNetwork(self.state_size, action_size, seed, 64, 64, 64).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                if NN_LOOPS > 1:
                    for _ in np.arange(NN_LOOPS-1):
                        experiences = self.memory.sample()
                        self.learn(experiences, GAMMA, True)

                experiences = self.memory.sample()
                self.learn(experiences, GAMMA, False)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, suppress_udpate):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        if suppress_udpate != True:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        # self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # e = self.experience(convert_state(state), action, reward, convert_state(next_state), done)
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)




| Time | episodes |     structure    |  batch size  |  update_every  |
|:----:|:--------:|:----------------:|:-------------:---------------:|
|      |          |                  |              |                |
|  638 |    559   |  512 / 512 / 256 |  128         |       6        |
|  348 |    434   |  256 / 256 / 256 |  128         |       6        |
|  526 |    567   |  256 / 256 / 64  |  32          |       2        |
|  343 |    563   |  256 / 256 / 64  | 128          |      12        |
|  393 |    599   |  256 / 256 / 64  | 256          |      12        |
|  964 |    513   | 2048 / 512 / 128 |  128         |       6        |
|  369 |    584   |  256 / 256 / 64  | 128          |      12        |
|  370 |    484   |  256 / 256 / 64  |  128         |       6        |
|  320 |    453   |  128 / 128 / 64  |  128         |       6        |
|  **310** |    **455**   |  **128 / 128 / 64**  | **128** | **6**   |
|  343 |    484   |   64 / 64 / 64   |  128         |       6        |
|  294 |    452   |   64 / 64 / 64   |  128         |       6        |


import torch
from torch import nn, Tensor
from torch.nn import functional as F

from gymnasium.spaces import Box

from copy import deepcopy


class Agent:
    """
    Base class for agents. Please inherit this when creating an agent
    """
    def __init__(self) -> None:
        super().__init__()
        self.prev_state = None
        self.action_taken = None

    def choose_action(self, action_space: Box, state: Box):
        self.prev_state = deepcopy(state)
        self.action_taken = deepcopy(action_space.sample())
        return self.action_taken
    
    def update(self, new_state: Box, reward: float, is_terminal: bool, is_trunc: bool):
        pass


class ValueApproximator(nn.Module):
    def __init__(self, in_dims, out_dims=1, width=8, depth=5, activation_function=F.relu) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.act = activation_function

        self.input_layer = nn.Linear(in_dims, width)
        self.output_layer = nn.Linear(width, out_dims)
        for i in range(depth - 2):
            self.layers.append(nn.Linear(width, width))
        
    def forward(self, x: Tensor):
        # x.shape = [n, in_dims]
        x = self.act(self.input_layer(x))

        for l in self.layers:
            x = self.act(l(x))
    
        return self.output_layer(l(x))

def example_model():
    f = ValueApproximator(in_dims=3, out_dims=3)
    optim = torch.optim.Adam(f.parameters(), lr=1e-3)

    x = torch.rand(size=(128, 3))
    y = torch.rand(size=(128, 3))

    for i in range(100):
        pred = f(x)
        optim.zero_grad()
        loss = F.mse_loss(pred, y)
        loss.backward()
        optim.step()
        print(loss)
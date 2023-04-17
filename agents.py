import torch
import numpy as np
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

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


class RoundingTDAgent(Agent):
    """
    Sarsa-Lambda Agent (Rounding TD Agent) ?
    """

    def __init__(
        self,
        state_space: Box,
        action_space: Box,
        discrete_actions: int,
        discrete_states: int,
        lambda_: float = 0.9,
        epsilon: float = 0.1,
        alpha: float = 0.1,
        gamma: float = 0.99,
    ) -> None:
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.discrete_actions = discrete_actions
        self.discrete_states = discrete_states

        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        # get dimensions of tensor
        multi_dim_array_shape = [discrete_states] * len(state_space.low) + [
            discrete_actions
        ]

        self.q_table = torch.zeros(multi_dim_array_shape)
        self.eligibility_traces = torch.zeros(multi_dim_array_shape)

    def discretize_state(self, state: Box):
        discrete_state = tuple(
            np.round(
                (state - self.state_space.low)
                / (self.state_space.high - self.state_space.low)
                * (self.discrete_states - 1)
            ).astype(int)
        )
        return discrete_state

    def choose_action(self, action_space: Box, state: Box):
        discrete_state = self.discretize_state(state)
        self.prev_state = discrete_state

        if np.random.uniform(0, 1) < self.epsilon:
            action = action_space.sample()
        else:
            action = torch.argmax(self.q_table[discrete_state[0]]).item()

        # Convert the discrete action to a continuous action
        continuous_action = (
            action
            / (self.discrete_actions - 1)
            * (action_space.high - action_space.low)
            + action_space.low
        )

        self.action_taken = continuous_action
        return self.action_taken

    def update(
        self,
        observation: Box,
        observation_next: Box,
        reward: float,
        is_terminal: bool,
        is_trunc: bool,
    ):
        # Convert the continuous states and actions to discrete states and actions
        next_discrete_state = self.discretize_state(observation_next)
        next_discrete_action = torch.argmax(self.q_table[next_discrete_state]).item()
        prev_discrete_state = self.discretize_state(observation)
        prev_discrete_action = torch.argmax(self.q_table[prev_discrete_state]).item()

        delta = (
            reward
            + self.gamma * self.q_table[next_discrete_state][next_discrete_action]
            - self.q_table[prev_discrete_state][prev_discrete_action]
        )

        self.eligibility_traces[prev_discrete_state][prev_discrete_action] += 1

        self.q_table += self.alpha * delta * self.eligibility_traces
        self.eligibility_traces *= self.gamma * self.lambda_

        if is_terminal or is_trunc:
            self.eligibility_traces *= 0

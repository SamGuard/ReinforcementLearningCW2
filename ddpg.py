import torch
from torch import nn, Tensor
from torch.nn import functional as F
import numpy as np
import math

from gymnasium.spaces import Box

from copy import deepcopy
from collections import deque, namedtuple
import random

from agents import ReplayMemory, Agent


class ValueApproximator(nn.Module):
    def __init__(
        self, in_dims, out_dims=1, width=8, depth=5, activation_function=F.relu
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.act = activation_function

        self.input_layer = nn.Linear(in_dims, width)
        self.output_layer = nn.Linear(width, out_dims)
        for i in range(depth - 2):
            self.layers.append(nn.Linear(width, width))

    def forward(self, x: Tensor):
        # x.shape = [n_data, in_dims]
        x = self.act(self.input_layer(x))

        for l in self.layers:
            x = self.act(l(x))

        return self.output_layer(x)


class ActionClamper(ValueApproximator):
    def __init__(
        self, in_dims, out_dims=1, width=8, depth=5, activation_function=F.relu
    ) -> None:
        super().__init__(in_dims, out_dims, width, depth, activation_function)

    def forward(self, x: Tensor):
        ans = torch.tanh(super().forward(x))
        return ans


class DDPG(Agent):
    def __init__(
        self,
        in_dim,
        out_dim,
        max_val,
        gamma=0.99,
        target_learn=0.001,
        batch_size=128,
        device="cpu",
    ):
        super().__init__()
        self.memory = ReplayMemory(in_dim, out_dim, device=device)
        self.actor = ActionClamper(
            in_dim, out_dim, width=256, depth=3, activation_function=F.relu
        ).to(device)
        self.critic = ValueApproximator(
            in_dim + out_dim, 1, width=256, depth=3, activation_function=F.relu
        ).to(device)

        self.actor_target = ActionClamper(
            in_dim, out_dim, width=256, depth=3, activation_function=F.relu
        ).to(device)
        self.critic_target = ValueApproximator(
            in_dim + out_dim, 1, width=256, depth=3, activation_function=F.relu
        ).to(device)

        self.actor_target.load_state_dict(deepcopy(self.actor.state_dict()))
        self.critic_target.load_state_dict(deepcopy(self.critic.state_dict()))

        # self.optim = torch.optim.Adam(param_list, 1e-3)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), 1e-4)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), 1e-3)

        self.gamma = gamma
        self.target_learn = target_learn
        self.batch_size = batch_size
        self.max_val = max_val
        self.device = device
        self.current_step = 0
        self.noise_level = 0.1

    def choose_action(
        self,
        state: np.ndarray,
        action_space: Box,
    ):
        self.prev_state = torch.tensor(state, device=self.device, dtype=torch.float32)
        N = self.noise_level * torch.randn(1).to(self.device)
        self.action_taken = torch.clamp(
            self.max_val * (self.actor(self.prev_state).detach() + N),
            -self.max_val,
            self.max_val,
        )
        return self.action_taken

    def update(
        self, new_state: np.ndarray, reward: float, is_terminal: bool, is_trunc: bool
    ):
        self.current_step += 1

        new_state = torch.tensor(new_state, dtype=torch.float32, device=self.device)
        self.memory.push(
            self.prev_state,
            self.action_taken,
            new_state,
            reward,
            is_terminal or is_trunc,
        )

        if self.memory.size < self.batch_size:
            return

        states, actions, next_states, rewards, terminal = self.memory.sample(
            self.batch_size
        )

        terminal_mask = terminal == 0

        next_states = next_states[terminal_mask]

        target_actions = self.actor_target(next_states)

        target_scores = self.critic_target(torch.cat((next_states, target_actions), 1))

        y = torch.zeros(size=(self.batch_size,), device=self.device)
        y[terminal_mask] = (self.gamma * target_scores).squeeze()
        y += rewards
        y = y.detach()

        current_scores = self.critic(torch.cat((states, actions), 1)).squeeze()

        loss_critic = F.mse_loss(current_scores, y)

        self.critic_optim.zero_grad()
        loss_critic.backward()
        self.critic_optim.step()

        loss_actor = -self.critic(torch.cat((states, self.actor(states)), 1)).mean()

        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()

        for target_weights, weights in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            target_weights.data.copy_(
                self.target_learn * weights.data
                + (1.0 - self.target_learn) * target_weights.data
            )

        for target_weights, weights in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_weights.data.copy_(
                self.target_learn * weights.data
                + (1.0 - self.target_learn) * target_weights.data
            )

    def get_learners(self):
        return deepcopy(self.actor),deepcopy(self.critic)
    
    def load(self, actor, critic, memory):
        self.actor = deepcopy(actor)
        self.critic = deepcopy(critic)
        self.memory = deepcopy(memory)

        self.actor_target.load_state_dict(deepcopy(self.actor.state_dict()))
        self.critic_target.load_state_dict(deepcopy(self.critic.state_dict()))

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), 1e-4)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), 1e-3)


import torch
from torch import nn, Tensor
from torch.nn import functional as F
import numpy as np
import math

from gymnasium.spaces import Box

from copy import deepcopy
from collections import deque, namedtuple
import random

MAX_MEMORY = 2**17

"""
Code taken from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
This code was used as it ReplayMemory is a very simple and useful class
"""
Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "terminal")
)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


"""
-----------------------------------------------------------------------------------------
"""

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

    def update(
        self, new_state: np.ndarray, reward: float, is_terminal: bool, is_trunc: bool
    ):
        pass


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

        return self.output_layer(l(x))
    

class ActionClamper(ValueApproximator):
    def __init__(self, in_dims, out_dims=1, width=8, depth=5, activation_function=F.relu) -> None:
        super().__init__(in_dims, out_dims, width, depth, activation_function)
    
    def forward(self, x: Tensor):
        #N = 0.1*torch.randn(1)
        #ans = torch.clamp(torch.tanh(super().forward(x)) + N, -1, 0.999)
        ans = torch.tanh(super().forward(x))

        #print(ans)
        return ans
        #return torch.clamp(torch.tanh(super().forward(x)) + N, -1, 1)
    
class DDPG(Agent):
    def __init__(self, state_dims, num_actions, target_learn=0.001, mini_batch_len=128, device="cpu") -> None:
        super().__init__()
        self.memory = ReplayMemory(MAX_MEMORY)
        self.actor = ActionClamper(state_dims, num_actions, width=32, depth=5, activation_function=torch.tanh)
        self.critic = ValueApproximator(state_dims + num_actions, 1, width=32, depth=5)
        
        self.actor_target = ActionClamper(state_dims, num_actions, width=32, depth=5, activation_function=torch.tanh)
        self.critic_target = ValueApproximator(state_dims + num_actions, 1, width=32, depth=5)
        
        self.actor_target.load_state_dict(deepcopy(self.actor.state_dict()))
        self.critic_target.load_state_dict(deepcopy(self.critic.state_dict()))
        
        #self.optim = torch.optim.Adam(param_list, 1e-3)
        
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), 0.0001)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), 0.001, weight_decay=0.001)

        self.state_dims = state_dims
        self.num_actions = num_actions
        self.gamma = 0.99
        self.target_learn = target_learn
        self.mini_batch_len = mini_batch_len
        self.device = device
    
    def choose_action(self, action_space: Box, state: np.ndarray):
        self.prev_state = torch.tensor(state, device=self.device, dtype=torch.float32)
        N = 0.1*torch.randn(1)
        self.action_taken = torch.clamp(self.actor(self.prev_state).detach() + N, -1, 1)
        return self.action_taken
    
    def update(self, new_state: np.ndarray, reward: float, is_terminal: bool, is_trunc: bool):
        new_state = torch.tensor(new_state, device=self.device, dtype=torch.float32)
        self.memory.push(
            self.prev_state, self.action_taken, new_state, reward, is_terminal or is_trunc
        )
        
        if len(self.memory.memory) < self.mini_batch_len:
            return
        batch = self.memory.sample(self.mini_batch_len)

        batch_states_0 = torch.zeros(
            size=(self.mini_batch_len, self.state_dims), device=self.device
        )
        batch_states_1 = torch.zeros(
            size=(self.mini_batch_len, self.state_dims), device=self.device
        )
        terminal_mask = torch.full(
            size=(self.mini_batch_len,),
            fill_value=False,
            dtype=torch.bool,
            device=self.device,
        )
        actions = torch.zeros(
            size=(self.mini_batch_len,self.num_actions), dtype=torch.long, device=self.device
        )
        rewards = torch.zeros(size=(self.mini_batch_len,))
        
        y = torch.zeros(size=(self.mini_batch_len,))

        target_actions = torch.zeros(
            size=(self.mini_batch_len, self.num_actions), device=self.device
        )
        
        target_scores = torch.zeros(
            size=(self.mini_batch_len,), device=self.device
        )

        for i, b in enumerate(batch):
            batch_states_0[i] = b.state
            batch_states_1[i] = b.next_state
            terminal_mask[i] = not b.terminal
            rewards[i] = b.reward
            actions[i] = b.action

        batch_states_1 = batch_states_1[terminal_mask]

        target_actions = self.actor_target(batch_states_1)

        target_scores = self.critic_target(torch.cat((batch_states_1, target_actions), 1))

        y[terminal_mask] = (self.gamma * target_scores).squeeze()
        y += rewards
        y = y.detach()

        current_scores = self.critic(torch.cat((batch_states_0, actions), 1)).squeeze()

        loss_critic = F.mse_loss(current_scores, y)

        self.critic_optim.zero_grad()
        loss_critic.backward()
        self.critic_optim.step()

        loss_actor = -self.critic(torch.cat((batch_states_0, self.actor(batch_states_0)), 1)).mean()

        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()

        for target_weights, weights in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_weights.data.copy_(self.target_learn * weights.data + (1.0 - self.target_learn) * target_weights.data)
       
        for target_weights, weights in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_weights.data.copy_(self.target_learn * weights.data + (1.0 - self.target_learn) * target_weights.data)
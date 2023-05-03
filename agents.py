import torch
from torch import nn, Tensor
from torch.nn import functional as F
import numpy as np
import math
import copy

from gymnasium.spaces import Box

from copy import deepcopy
from collections import deque, namedtuple
import random

MAX_MEMORY = 1000000

"""
Code taken from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
This code was used as it ReplayMemory is a very simple and useful class
"""
Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "terminal")
)

class ReplayMemory(object):
    def __init__(self, in_dim, out_dim):
        self.curr = 0
        self.size = 0

        self.state = np.zeros((MAX_MEMORY, in_dim))
        self.action = np.zeros((MAX_MEMORY, out_dim))
        self.next_state = np.zeros((MAX_MEMORY, in_dim))
        self.reward = np.zeros((MAX_MEMORY, 1))
        self.terminal = np.zeros((MAX_MEMORY, 1))

    def push(self, state, action, next_state, reward, terminal):
        self.state[self.curr] = state
        self.action[self.curr] = action
        self.next_state[self.curr] = next_state
        self.reward[self.curr] = reward
        self.terminal[self.curr] = terminal

        self.curr = (self.curr + 1) & MAX_MEMORY
        self.size = min(self.size + 1, MAX_MEMORY)
    
    def sample(self, batch_size):
        rand_elems = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.tensor(self.state[rand_elems], dtype=torch.float32),
            torch.tensor(self.action[rand_elems], dtype=torch.float32),
            torch.tensor(self.next_state[rand_elems], dtype=torch.float32),
            torch.tensor(self.reward[rand_elems], dtype=torch.float32),
            torch.tensor(self.terminal[rand_elems], dtype=torch.float32)
        )
        
"""
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        #Save a transition
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
"""

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

class Actor(nn.Module):
    def __init__(self, in_dim, out_dim, max_val):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(in_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 128)
        self.l4 = nn.Linear(128, out_dim)
        self.max_val = max_val
    
    def forward(self, state):
        out = F.relu(self.l1(state))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))
        return self.max_val * torch.tanh(self.l4(out))
    
class Critic(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Critic, self).__init__()
        
        # critic 1
        self.l1 = nn.Linear(in_dim + out_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 128)
        self.l4 = nn.Linear(128, out_dim)

        # critic 2
        self.l5 = nn.Linear(in_dim + out_dim, 128)
        self.l6 = nn.Linear(128, 128)
        self.l7 = nn.Linear(128, 128)
        self.l8 = nn.Linear(128, out_dim)

    def forward(self, state, action):
        return self.critic_1(state, action), self.critic_2(state, action)
    
    def critic_1(self, state, action):
        state_action = torch.cat([state, action], 1)
        out = F.relu(self.l1(state_action))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))
        out = self.l4(out)
        return out
    
    def critic_2(self, state, action):
        state_action = torch.cat([state, action], 1)
        out = F.relu(self.l5(state_action))
        out = F.relu(self.l6(out))
        out = F.relu(self.l7(out))
        out = self.l8(out)
        return out

class TD3(Agent):
    def __init__(self, in_dim, out_dim, max_val, gamma = 0.99, target_learn=0.001, batch_size=100):
        super().__init__()
        
        self.memory = ReplayMemory(in_dim, out_dim)

        self.actor = Actor(in_dim, out_dim, max_val)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(in_dim, out_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=3e-3)

        self.gamma = gamma
        self.target_learn = target_learn # tau
        self.batch_size = batch_size
        self.max_val = max_val

        self.current_step = 0


    def choose_action(self, state, action_space):
        self.prev_state = torch.tensor(state, dtype=torch.float32)
        if (self.memory.size < 10000):
            self.action_taken = torch.tensor(action_space.sample(), dtype=torch.float32)
        else:
            self.action_taken = self.actor(self.prev_state).detach()
        return self.action_taken
    
    def update(self, new_state: np.ndarray, reward: float, is_terminal: bool, is_trunc: bool):
        self.current_step += 1

        new_state = torch.tensor(new_state, dtype=torch.float32)
        self.memory.push(
            self.prev_state, self.action_taken, new_state, reward, is_terminal or is_trunc
        )

        if self.memory.size < self.batch_size:
            return
        
        state, action, next_state, reward, terminal = self.memory.sample(self.batch_size)


        with torch.no_grad():
            noise = (torch.randn_like(action) * 0.2).clamp(-0.5, 0.5)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_val, self.max_val)
            
            target_critic_1, target_critic_2 = self.critic_target(next_state, next_action)
            target_critic = torch.min(target_critic_1, target_critic_2)
            target_critic = reward + terminal * self.gamma * target_critic


        current_critic_1, current_critic_2 = self.critic(state, action)
        loss_critic = F.mse_loss(current_critic_1, target_critic) + F.mse_loss(current_critic_2, target_critic)

        self.critic_optim.zero_grad()
        loss_critic.backward()
        self.critic_optim.step()
       
        if self.current_step % 200 == 0:
            loss_actor = -self.critic.critic_1(state, self.actor(state)).mean()
            self.actor_optim.zero_grad()
            loss_actor.backward()
            self.actor_optim.step()

            for target_weights, weights in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_weights.data.copy_(self.target_learn * weights.data + (1.0 - self.target_learn) * target_weights.data)
        
            for target_weights, weights in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_weights.data.copy_(self.target_learn * weights.data + (1.0 - self.target_learn) * target_weights.data)

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import numpy as np
import copy

from gymnasium.spaces import Box

from copy import deepcopy

from replay import ReplayMemory

MAX_MEMORY = 2**17

class Agent:
    """
    Base class for agents. Please inherit this when creating an agent
    """

    def __init__(self) -> None:
        super().__init__()
        self.prev_state = None
        self.action_taken = None

    def choose_action(self, state: Box, action_space: Box):
        self.prev_state = deepcopy(state)
        self.action_taken = deepcopy(action_space.sample())
        return self.action_taken

    def update(
        self, new_state: np.ndarray, reward: float, is_terminal: bool, is_trunc: bool
    ):
        pass

# Actor & Critic class code inspired by https://towardsdatascience.com/td3-learning-to-run-with-ai-40dfc512f93
temp = 128
class Actor(nn.Module):
    def __init__(self, in_dim, out_dim, max_val):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(in_dim, temp)
        self.l2 = nn.Linear(temp, temp)
        self.l3 = nn.Linear(temp, out_dim)
        self.max_val = max_val
    
    def forward(self, state):
        out = F.relu(self.l1(state))
        out = F.relu(self.l2(out))
        return self.max_val * torch.tanh(self.l3(out))
    
class Critic(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Critic, self).__init__()
        
        # critic 1
        self.l1 = nn.Linear(in_dim + out_dim, temp)
        self.l2 = nn.Linear(temp, temp)
        self.l3 = nn.Linear(temp, out_dim)

        # critic 2
        self.l4 = nn.Linear(in_dim + out_dim, temp)
        self.l5 = nn.Linear(temp, temp)
        self.l6 = nn.Linear(temp, out_dim)

    def forward(self, state, action):
        return self.critic_1(state, action), self.critic_2(state, action)
    
    def critic_1(self, state, action):
        state_action = torch.cat([state, action], 1)
        out = F.relu(self.l1(state_action))
        out = F.relu(self.l2(out))
        out = self.l3(out)
        return out
    
    def critic_2(self, state, action):
        state_action = torch.cat([state, action], 1)
        out = F.relu(self.l4(state_action))
        out = F.relu(self.l5(out))
        out = self.l6(out)
        return out

class TD3(Agent):
    def __init__(self, in_dim, out_dim, max_val, noise = 0.25, copy_step = 2, gamma = 0.99, target_learn=0.005, batch_size=128, device="cpu"):
        super().__init__()
        
        self.memory = ReplayMemory(in_dim, out_dim, max_mem=2**17, device=device)

        self.actor = Actor(in_dim, out_dim, max_val).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target = self.actor_target.to(device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.001)

        self.critic = Critic(in_dim, out_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target = self.critic_target.to(device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.0001)

        self.gamma = gamma
        self.target_learn = target_learn # tau
        self.batch_size = batch_size
        self.max_val = max_val

        self.current_step = 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.noise = noise
        self.copy_step = copy_step
        self.device = device

    def choose_action(self, state, action_space):
        self.prev_state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if (self.current_step < 10000):
            self.action_taken = torch.tensor(action_space.sample())
        else:
            self.action_taken = self.actor(self.prev_state).detach()
        return self.action_taken
    
    def update(self, new_state: np.ndarray, reward: float, is_terminal: bool, is_trunc: bool):
        self.current_step += 1

        new_state = torch.tensor(new_state, dtype=torch.float32, device=self.device)
        
        self.memory.push(
            self.prev_state, self.action_taken, new_state, reward, is_terminal or is_trunc
        )
        if self.memory.size < 10000:
            return

        with torch.no_grad():
            state, action, next_state, reward, terminal = self.memory.sample(self.batch_size)
            
            noise = (torch.rand_like(action, device=self.device) * self.noise).clamp(-0.5, 0.5) # * (0.99**(self.memory.size // 10000))

            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_val, self.max_val)
            
            target_critic_1, target_critic_2 = self.critic_target(next_state, next_action)
            target_critic = torch.min(target_critic_1, target_critic_2)
            target_critic = reward.reshape(-1, 1) + (1-terminal.reshape(-1, 1)) * self.gamma * target_critic

        current_critic_1, current_critic_2 = self.critic(state, action)
        loss_critic = F.mse_loss(current_critic_1, target_critic) + F.mse_loss(current_critic_2, target_critic)

        self.critic_optim.zero_grad()
        loss_critic.backward()
        self.critic_optim.step()
       
        if self.current_step % self.copy_step == 0:
            loss_actor = -self.critic.critic_1(state, self.actor(state)).mean()
            self.actor_optim.zero_grad()
            loss_actor.backward()
            self.actor_optim.step()

            for target_weights, weights in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_weights.data.copy_(self.target_learn * weights.data + (1.0 - self.target_learn) * target_weights.data)
        
            for target_weights, weights in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_weights.data.copy_(self.target_learn * weights.data + (1.0 - self.target_learn) * target_weights.data)

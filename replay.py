import torch
from torch import nn, Tensor
from torch.nn import functional as F

from collections import deque, namedtuple


Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "terminal")
)

MAX_MEMORY = 2**17

class ReplayMemory(object):
    def __init__(self, in_dim, out_dim, max_mem, device="cpu"):
        self.curr = 0
        self.size = 0
        
        self.max_mem = max_mem
        self.state = torch.zeros((self.max_mem, in_dim), device=device)
        self.action = torch.zeros((self.max_mem, out_dim), device=device)
        self.next_state = torch.zeros((self.max_mem, in_dim), device=device)
        self.reward = torch.zeros((self.max_mem,), device=device)
        self.terminal = torch.zeros((self.max_mem,), device=device)

        self.device = device

    def push(self, state, action, next_state, reward, terminal):
        self.state[self.curr] = state
        self.action[self.curr] = action
        self.next_state[self.curr] = next_state
        self.reward[self.curr] = reward
        self.terminal[self.curr] = terminal

        self.curr = (self.curr + 1) % self.max_mem
        self.size = min(self.size + 1, self.max_mem)

    def sample(self, batch_size):
        rand_elems = torch.randint(0, self.size, size=(batch_size,))

        return (
            self.state[rand_elems].clone(),
            self.action[rand_elems].clone(),
            self.next_state[rand_elems].clone(),
            self.reward[rand_elems].clone(),
            self.terminal[rand_elems].clone(),
        )

    def __len__(self):
        return self.size


class PrioritiseReplay(object):
    """
    Doesnt work :(
    """
    def __init__(self, in_dim, out_dim, device="cpu"):
        self.curr = 0
        self.size = 0

        self.state = torch.zeros((MAX_MEMORY, in_dim), device=device)
        self.action = torch.zeros((MAX_MEMORY, out_dim), device=device)
        self.next_state = torch.zeros((MAX_MEMORY, in_dim), device=device)
        self.reward = torch.zeros((MAX_MEMORY,), device=device)
        self.terminal = torch.zeros((MAX_MEMORY,), device=device)
        self.td = torch.ones((MAX_MEMORY,), device=device)

        self.device = device

    def push(self, state, action, next_state, reward, terminal):
        self.state[self.curr] = state
        self.action[self.curr] = action
        self.next_state[self.curr] = next_state
        self.reward[self.curr] = reward
        self.terminal[self.curr] = terminal
        if self.size > 0:
            self.td[self.curr] = self.td[: self.size].mean()

        self.curr = (self.curr + 1) % MAX_MEMORY
        self.size = min(self.size + 1, MAX_MEMORY)

    def sample(self, batch_size):
        td = self.td[: self.size]
        weights = batch_size * td / td.sum()

        choose = weights > torch.rand_like(weights)
        selected = torch.linspace(
            0, self.size - 1, self.size, dtype=torch.long, device=self.device
        )[choose]

        curr_size = selected.shape[0]
        if curr_size < batch_size:
            selected = torch.cat(
                (
                    selected,
                    torch.randint(
                        0,
                        self.size,
                        (batch_size - curr_size,),
                        dtype=torch.long,
                        device=self.device,
                    ),
                ),
                dim=0,
            )
        elif curr_size > batch_size:
            selected = selected[torch.randperm(curr_size)[:batch_size]]

        return (
            self.state[selected].clone(),
            self.action[selected].clone(),
            self.next_state[selected].clone(),
            self.reward[selected].clone(),
            self.terminal[selected].clone(),
            selected,
        )

    def update_td(self, indexs, td):
        self.td[indexs] = td

    def __len__(self):
        return self.size

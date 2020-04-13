import numpy as np
import random

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

from ddpg_agent import Agent
from ounoise import OUNoise
from replay_buffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, seed, batch_size,
                 update_every=5, n_update_networks=5):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents
            seed (int): random seed
            batch_size (int): minibatch size
            update_every (int): after how many timesteps to update the network
            n_update_networks (int): how often to update the network in a row
        """
        self.agents = []
        for i in range(num_agents):
            self.agents.append(Agent(state_size, action_size, num_agents, seed, batch_size, i))

        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = ReplayBuffer(action_size, buffer_size=int(1e6), batch_size=batch_size, seed=seed)

        self.update_every = update_every
        self.n_update_networks = n_update_networks
        self.step_count = 0

    def step(self, states, actions, rewards, next_states, dones):
        # add new experience to every agent
        self.memory.add(states, actions, rewards, next_states, dones)

        # check if enough samples available
        if len(self.memory) < self.batch_size:
            return

        self.step_count += 1
        if self.step_count % self.update_every != 0:
            return

        for i in range(self.n_update_networks):
            for agent in self.agents:
                # get experiences
                experiences = self.memory.sample()

                # add actions_next to experiences
                states_all, _, _, _, _ = experiences

                # add actions_next_target
                actions_next_target_all = []
                for i in range(len(self.agents)):
                    states_self = states_all[:,i,:].view(-1, self.state_size)
                    actions_next_self = self.agents[i].actor_target(states_self)
                    actions_next_target_all.append(actions_next_self)
                actions_next_target_all = torch.stack(actions_next_target_all, dim=1)
                experiences += (actions_next_target_all, )

                # add actions_next_local
                actions_next_local_all = []
                for i in range(len(self.agents)):
                    states_self = states_all[:,i,:].view(-1, self.state_size)
                    actions_next_self = self.agents[i].actor_local(states_self)
                    actions_next_local_all.append(actions_next_self)
                actions_next_local_all = torch.stack(actions_next_local_all, dim=1)
                experiences += (actions_next_local_all, )

                # learn
                agent.learn(experiences, agent.gamma, agent.tau)

    def act(self, states, add_noise=True):
        actions = []
        for i in range(len(self.agents)):
            actions.append(self.agents[i].act(states[i], add_noise))
        return np.asarray(actions)

    def reset(self):
        for agent in self.agents:
            agent.noise.reset()

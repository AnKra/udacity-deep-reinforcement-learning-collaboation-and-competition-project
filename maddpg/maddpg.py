import numpy as np
import random

import torch

from maddpg.ddpg_agent import Agent
from replay_buffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG():
    """Interacts with and learns from the environment."""

    def __init__(self, agents, seed, batch_size, update_every=5, n_update_networks=5):
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
        self.num_agents = len(agents)
        self.agents = agents

        self.batch_size = batch_size
        self.memory = ReplayBuffer(buffer_size=int(1e6), batch_size=batch_size, seed=seed)

        self.update_every = update_every
        self.n_update_networks = n_update_networks
        self.step_count = 0

    def step(self, states, actions, rewards, next_states, dones):
        # add new experience to every agent
        states = states.reshape(1, -1)
        actions = actions.reshape(1, -1)
        next_states = next_states.reshape(1, -1)
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
                actions_next_target_all = torch.Tensor().to(device)
                for i in range(self.num_agents):
                    states_self = states_all.view(self.batch_size, self.num_agents, -1)[:,i,:]
                    actions_next_self = self.agents[i].actor_target(states_self)
                    actions_next_target_all = torch.cat((actions_next_target_all, actions_next_self), dim=1)
                experiences += (actions_next_target_all, )

                # add actions_next_local
                actions_next_local_all = torch.Tensor().to(device)
                for i in range(self.num_agents):
                    states_self = states_all.view(self.batch_size, self.num_agents, -1)[:,i,:]
                    actions_next_self = self.agents[i].actor_local(states_self)
                    actions_next_local_all = torch.cat((actions_next_local_all, actions_next_self), dim=1)
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

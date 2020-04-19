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
        self.agents = agents
        self.num_all_agents = 0
        for agent in agents:
            self.num_all_agents += agent.num_agents

        self.batch_size = batch_size
        self.memory = ReplayBuffer(buffer_size=int(1e6), batch_size=batch_size, seed=seed)

        self.update_every = update_every
        self.n_update_networks = n_update_networks
        self.step_count = 0

    def step(self, states, actions, rewards, next_states, done, discretize_actions=False):
        # add new experience to every agent
        states = states.reshape(1, -1)
        actions = actions.reshape(1, -1)
        rewards = rewards.reshape(1, -1)
        next_states = next_states.reshape(1, -1)
        self.memory.add(states, actions, rewards, next_states, done)

        # check if enough samples available
        if len(self.memory) < self.batch_size:
            return

        self.step_count += 1
        if self.step_count % self.update_every != 0:
            return

        for i in range(self.n_update_networks):
            agent_index = 0
            for agent in self.agents:
                # get experiences
                experiences = self.memory.sample()

                # add actions_next to experiences
                states_all, _, _, _, _ = experiences

                # add actions_next_target
                actions_next_target_all = self.get_actions_next_target_all(states_all, discretize_actions)
                experiences += (actions_next_target_all, )

                # add actions_next_local
                actions_next_local_all = self.get_actions_next_local_all(states_all, discretize_actions)
                experiences += (actions_next_local_all, )

                # learn
                agent.learn(experiences, agent.gamma, agent.tau, agent_index)
                agent_index += 1

    def get_actions_next_target_all(self, states_all, discretize_actions):
        actions_next_target_all = torch.Tensor().to(device)
        agent_index = 0
        for agent in self.agents:
            for _ in range(agent.num_agents):
                states_self = states_all.view(self.batch_size, self.num_all_agents, -1)[:,agent_index,:]
                actions_next_self = agent.actor_target(states_self)
                if discretize_actions:
                    actions_next_self = torch.argmax(actions_next_self, dim=1, keepdim=True).float()
                actions_next_target_all = torch.cat((actions_next_target_all, actions_next_self), dim=1)
                agent_index += 1
        return actions_next_target_all

    def get_actions_next_local_all(self, states_all, discretize_actions):
        actions_next_local_all = torch.Tensor().to(device)
        agent_index = 0
        for agent in self.agents:
            for _ in range(agent.num_agents):
                states_self = states_all.view(self.batch_size, self.num_all_agents, -1)[:,agent_index,:]
                actions_next_self = agent.actor_local(states_self)
                if discretize_actions:
                    actions_next_self = torch.argmax(actions_next_self, dim=1, keepdim=True).float()
                actions_next_local_all = torch.cat((actions_next_local_all, actions_next_self), dim=1)
                agent_index += 1
        return actions_next_local_all


    def act(self, states, add_noise=True, discretize_actions=False):
        actions_all = []
        agent_index = 0
        for agent in self.agents:
            actions_of_agent = []
            for _ in range(agent.num_agents):
                action = agent.act(states[agent_index], add_noise)
                if discretize_actions:
                    action = np.argmax(action, axis=0)
                actions_of_agent.append(action)
                agent_index += 1
            actions_all.append(np.stack(actions_of_agent))
        return np.vstack(actions_all)

    def reset(self):
        for agent in self.agents:
            agent.noise.reset()

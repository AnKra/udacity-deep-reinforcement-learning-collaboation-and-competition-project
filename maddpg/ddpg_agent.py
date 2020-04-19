import numpy as np
import random

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

from ounoise import OUNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, num_all_agents, seed,
                 batch_size, buffer_size=int(1e6), gamma=0.99, tau=1e-3, lr_actor=4e-4,
                 lr_critic=4e-4, weight_decay=0, discrete_actions=False):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_all_agents (int): number of agents
            seed (int): random seed
            batch_size (int): minibatch size
            buffer_size (int): replay buffer size
            gamma (float): discount factor
            tau (float): for soft update of target parameters
            lr_actor (float): learning rate of the actor
            lr_critic (float): learning rate of the critic
            weight_decay (float): L2 weight decay
        """
        random.seed(seed)

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.num_all_agents = num_all_agents
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.noise = OUNoise(action_size, seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed, use_batch_norm_layers=False).to(device)
        self.actor_target = Actor(state_size, action_size, seed, use_batch_norm_layers=False).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        if discrete_actions:
            action_size = 1
        self.critic_local = Critic(state_size * num_all_agents, action_size * num_all_agents, seed).to(device)
        self.critic_target = Critic(state_size * num_all_agents, action_size * num_all_agents, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, tau, agent_index):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states_all, actions_all, rewards_all, next_states_all, dones, actions_next_target_all, actions_next_local_all = experiences

        rewards_self = rewards_all[:, agent_index]
        states_self = states_all.view(-1, self.num_all_agents, self.state_size)[: ,agent_index, :]
        del rewards_all

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        Q_targets_next = self.critic_target(next_states_all, actions_next_target_all)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards_self + (gamma * Q_targets_next) * (1 - dones)
        # Compute critic loss
        Q_expected = self.critic_local(states_all, actions_all)
        critic_loss = F.mse_loss(Q_expected.view(-1, self.batch_size), Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actor_loss = -self.critic_local(states_all, actions_next_local_all).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, tau)
        self.soft_update(self.actor_local, self.actor_target, tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

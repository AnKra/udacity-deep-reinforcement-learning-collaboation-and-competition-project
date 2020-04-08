import numpy as np
import random

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

from ounoise import OUNoise
from replay_buffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, seed, batch_size,
                 buffer_size=int(1e6), gamma=0.99, tau=1e-3, lr_actor=4e-4,
                 lr_critic=4e-4, weight_decay=0):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents
            seed (int): random seed
            batch_size (int): minibatch size
            buffer_size (int): replay buffer size
            gamma (float): discount factor
            tau (float): for soft update of target parameters
            lr_actor (float): learning rate of the actor
            lr_critic (float): learning rate of the critic
            weight_decay (float): L2 weight decay
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed, use_batch_norm_layers=False).to(device)
        self.actor_target = Actor(state_size, action_size, seed, use_batch_norm_layers=False).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        print(self.actor_local)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)

        print(self.critic_local)

        # Noise process
        self.noise = OUNoise(action_size, seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)

    def add_experience(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory."""
        # Save experience / reward
        self.memory.add(states, actions, rewards, next_states, dones)

    def get_sample(self):
        """Use random sample from buffer to learn."""
        # Learn every time step.
        # Learn, if enough samples are available in memory
        if len(self.memory) >= self.batch_size:
            return self.memory.sample()
        else:
            return None

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

    def learn(self, self_experiences, others_experiences, gamma, tau):
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
        states, actions, rewards, next_states, dones = self_experiences
        states_n, actions_n, _, next_states_n, n_dones = others_experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next_n = self.actor_target(next_states_n)
        Q_targets_next = self.critic_target(next_states_n, actions_next_n)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next.view(-1, self.num_agents)) * (1 - n_dones.view(-1, self.num_agents))
        # Compute critic loss
        Q_expected = self.critic_local(states_n, actions_n)  # TODO incompatible shapes???
        critic_loss = F.mse_loss(Q_expected.view(-1, self.num_agents), Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
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

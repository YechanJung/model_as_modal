import random

from tqdm import tqdm
from collections import deque

import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader

class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=(64, 64 ),
        activation_fn=torch.tanh,
        n_steps=2048,
        n_epochs=10,
        batch_size=64,
        policy_lr=0.0003,
        value_lr=0.0003,
        gamma=0.99,
        lmda=0.95,
        clip_ratio=0.2,
        vf_coef=1.0,
        ent_coef=0.01,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = MLPGaussianPolicy(state_dim, action_dim, hidden_dims, activation_fn).to(self.device)
        self.value = MLPStateValue(state_dim, hidden_dims, activation_fn).to(self.device)        
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lmda = lmda
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=value_lr)
        
        self.buffer = RolloutBuffer()

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action, log_prob = self.policy.sample(state)
        return action.cpu().detach().numpy(), log_prob.cpu().detach().numpy()

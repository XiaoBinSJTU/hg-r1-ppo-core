"""Compact structured reward surrogate; not a domain language model."""

import torch
from torch import nn

class RewardModel(nn.Module):
    def __init__(self, dimension=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(10, dimension), nn.SiLU(), nn.Linear(dimension, dimension), nn.SiLU())
        self.reconstruction = nn.Linear(dimension, 10)
        self.reward_head = nn.Linear(dimension, 4)
        self.threshold = nn.Linear(dimension, 1)

    def forward(self, context):
        h = self.encoder(context)
        return self.reward_head(h).softmax(-1), .6 + .4 * self.threshold(h).sigmoid().squeeze(-1)

    def log_density(self, context, candidate):
        weights, _ = self(context)
        concentration = 1 + 24 * weights
        return torch.distributions.Dirichlet(concentration).log_prob(candidate.clamp_min(1e-5))


def physical_validate(weights, threshold, context):
    return (torch.isfinite(weights).all(-1) & (weights >= 0).all(-1)
            & ((weights.sum(-1) - 1).abs() < 1e-4) & (threshold <= .80)
            & ((context[:, 0] < .7) | (weights[:, 1] >= weights[:, 0])))


def reward_weights(model, context, mechanism):
    if mechanism == 'fixed_PPO':
        return torch.tensor([.35, .35, .15, .15], device=context.device).expand(len(context), -1)
    supplied = context.clone()
    if mechanism == 'LMGT_proxy_PPO':
        supplied[:, 2:6] = 0
    weights, threshold = model(supplied)
    if mechanism in ('HG_reward_proxy_PPO', 'no_DPO'):
        valid = physical_validate(weights, threshold, supplied)
        # Project rejected candidates into the physical feasible set, without target scores.
        repaired = weights.clone()
        critical = supplied[:, 0] >= .7
        risk, operation = repaired[:, 1].clone(), repaired[:, 0].clone()
        repaired[:, 1] = torch.where(critical, torch.maximum(risk, operation), risk)
        repaired = repaired / repaired.sum(-1, keepdim=True)
        weights = torch.where(valid[:, None], weights, repaired)
    return weights

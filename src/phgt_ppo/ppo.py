"""Actor-critic decisions and the detached generalized advantage operator."""

import torch
from torch import nn

def network(widths, output):
    layers, previous = [], 16
    for width in widths:
        layers.extend([nn.Linear(previous, width), nn.SiLU()])
        previous = width
    layers.append(nn.Linear(previous, output))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.actor = network(cfg['ppo']['hidden'], 3)
        self.critic = network(cfg['ppo']['hidden'], 1)

    def forward(self, state):
        logits = self.actor(state)
        blocked = state[:, 10] > 1
        logits = logits.clone()
        logits[:, :2] = logits[:, :2].masked_fill(blocked[:, None], -1e4)
        return torch.distributions.Categorical(logits=logits), self.critic(state).squeeze(-1)


def generalized_advantage_estimate(rewards, values, dones, next_value, gamma=.99, gae_lambda=.95):
    """Evaluate the source GAE recurrence on caller-supplied tensors.

    Inputs have shape [time, batch], with a [batch] bootstrap value.
    This operator never creates trajectories or accesses an environment.
    """
    if rewards.shape != values.shape or dones.shape != rewards.shape:
        raise ValueError('Rewards, values and done masks must share a shape')
    if rewards.ndim != 2 or next_value.shape != rewards.shape[1:]:
        raise ValueError('Expected time-by-batch tensors and a batch bootstrap')
    with torch.no_grad():
        advantage = torch.zeros_like(next_value)
        advantages = []
        for step in reversed(range(len(rewards))):
            alive = (~dones[step]).float()
            delta = rewards[step] + gamma * next_value * alive - values[step]
            advantage = delta + gamma * gae_lambda * alive * advantage
            advantages.append(advantage)
            next_value = values[step]
        advantages = torch.stack(list(reversed(advantages)))
        returns = advantages + values
    return advantages, returns

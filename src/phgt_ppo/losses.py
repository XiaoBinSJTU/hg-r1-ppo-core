"""Observed forecasting, Jacobian consistency, recovery and clipped policy losses."""

import torch
import torch.nn.functional as F


def forecast_loss(output, batch, cfg, local, physical, recovery_enabled=None):
    apply_recovery = physical if recovery_enabled is None else recovery_enabled
    error = (output['prediction'] - batch['y']).square() * batch['ymask']
    loss = error.sum() / batch['ymask'].sum().clamp_min(1)
    if local:
        loss = loss + .1 * F.mse_loss(output['tau'] / 100, batch['tau'] / 100)
        loss = loss + .05 * F.mse_loss(output['mass'] / 25, batch['mass'] / 25)
        if physical:
            consistency = F.mse_loss(output['raw_tau'] / 100, output['physical_tau'] / 100)
            loss = loss + cfg['forecast']['torque_loss_weight'] * consistency
        if apply_recovery:
            previous = torch.cat([batch['last_mass'].unsqueeze(-1), output['mass'][:, :-1]], dim=1)
            residual = F.relu(output['mass'] - .3 * previous).square() * batch['actions']
            recovery = residual.sum() / batch['actions'].sum().clamp_min(1) / 625
            loss = loss + cfg['forecast']['recovery_loss_weight'] * recovery
    return loss


def dpo_loss(chosen_logp, rejected_logp, reference_chosen, reference_rejected, beta):
    margin = (chosen_logp - rejected_logp) - (reference_chosen - reference_rejected)
    return -F.logsigmoid(beta * margin).mean()


def ppo_loss(new_logp, old_logp, advantage, value, returns, entropy, cfg):
    ratio = (new_logp - old_logp).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1-cfg['clip'], 1+cfg['clip']) * advantage)
    return -surrogate.mean() + cfg['value_weight'] * F.mse_loss(value, returns) - cfg['entropy_weight'] * entropy.mean()

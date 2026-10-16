"""Sixteen-dimensional policy states and structured reward contexts."""

import torch


def policy_state(mass, ratio, age, thermal, uncertainty, recovery, max_mass):
    module = torch.cat([mass[:, None] / max_mass, ratio[:, 1:]], dim=1)
    anomaly = torch.stack([ratio.max(1).values, mass / max_mass, uncertainty], dim=1)
    force_error = (mass / max_mass).square()
    extra = torch.stack([mass / max_mass, ratio.max(1).values, force_error,
                         age / 50, torch.full_like(mass, .6), thermal, uncertainty], dim=1)
    return torch.cat([module, anomaly, extra], dim=1)


def reward_context(state, diagnostics, predicted_recovery=None):
    return torch.stack([state[:, 9], state[:, 10],
                        torch.full_like(state[:, 0], diagnostics[0]), state[:, 11],
                        torch.full_like(state[:, 0], diagnostics[1]), (state[:, 12] < .06).float(),
                        state[:, 14], state[:, 15] if predicted_recovery is None else predicted_recovery,
                        state[:, 6], state[:, 7]], dim=1).clamp(0, 2)

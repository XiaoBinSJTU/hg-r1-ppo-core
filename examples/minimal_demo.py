"""Synthetic inputs for demonstration only; no data or model files are read."""

import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import torch
from phgt_ppo import ForecastModel, RewardModel, ActorCritic
from phgt_ppo import forecast_loss, dpo_loss, ppo_loss, generalized_advantage_estimate
from phgt_ppo import physical_validate, reward_weights
from phgt_ppo.config import load_demo_config
from phgt_ppo.features import policy_state, reward_context
from phgt_ppo.physics import unit_load_map
from phgt_ppo.random_utils import seed_all, select_device


def synthetic_inputs(device):
    """Return tiny interface tensors unrelated to any recorded equipment data."""
    batch, history, channels, horizon = 2, 16, 18, 4
    x = torch.randn(batch, history, channels, device=device) * .1
    mass = torch.rand(batch, horizon, device=device) + 1
    jac = torch.randn(batch, horizon, 6, device=device) * .04
    return {'x': x, 'mask': torch.ones_like(x),
            'hist_jac': torch.randn(batch, history, 6, device=device) * .04,
            'actions': torch.tensor([[0., 1., 0., 0.], [1., 0., 0., 0.]], device=device),
            'jac': jac, 'y': torch.randn(batch, horizon, channels, device=device) * .1,
            'ymask': torch.ones(batch, horizon, channels, device=device),
            'mass': mass, 'tau': mass.unsqueeze(-1) * jac,
            'last_mass': torch.ones(batch, device=device)}


def run_demo(device_mode='auto'):
    seed_all(17)
    device, selected = select_device(device_mode)
    cfg = load_demo_config()
    batch = synthetic_inputs(device)
    mean, scale = torch.zeros(18, device=device), torch.ones(18, device=device)
    model = ForecastModel(18, cfg, method='PHGT', local=True).to(device)
    before = model.decoder[-1].weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    output = model(batch, mean, scale)
    assert output['prediction'].shape == (2, 4, 18)
    assert output['tau'].shape == (2, 4, 6)
    assert all(torch.isfinite(value).all() for value in output.values())
    loss = forecast_loss(output, batch, cfg, local=True, physical=True)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.decoder[-1].weight.grad).all()
    optimizer.step()
    assert not torch.equal(before, model.decoder[-1].weight)

    # These are interface values, not equipment limits or a simulated case.
    mass = torch.ones(2, device=device)
    ratio = torch.full((2, 6), .5, device=device)
    state = policy_state(mass, ratio, mass, mass*.5, mass*.02, mass*.1, 1.)
    context = reward_context(state, (0., 0.), mass*.1)
    reward = RewardModel().to(device)
    weights = reward_weights(reward, context, 'HG_reward_proxy_PPO')
    assert torch.allclose(weights.sum(-1), torch.ones(2, device=device))
    checked_context = context.clone()
    checked_context[:, 0] = .9
    checked_weights = weights.clone()
    checked_weights[:, 1] = torch.maximum(checked_weights[:, 0], checked_weights[:, 1])
    checked_weights = checked_weights / checked_weights.sum(-1, keepdim=True)
    assert physical_validate(checked_weights, torch.full((2,), .75, device=device), checked_context).all()
    chosen = torch.full((2, 4), .25, device=device)
    rejected = torch.tensor([[.4, .2, .2, .2], [.2, .4, .2, .2]], device=device)
    cp, rp = reward.log_density(context, chosen), reward.log_density(context, rejected)
    preference = dpo_loss(cp, rp, cp.detach(), rp.detach(), .1)
    preference.backward()
    assert reward.reward_head.weight.grad.abs().sum() > 0

    policy = ActorCritic(cfg).to(device)
    distribution, value = policy(state)
    action = distribution.sample()
    advantage, returns = generalized_advantage_estimate(
        torch.full((2, 2), .1, device=device), torch.zeros(2, 2, device=device),
        torch.tensor([[False, False], [True, True]], device=device), torch.zeros(2, device=device))
    policy_objective = ppo_loss(distribution.log_prob(action), distribution.log_prob(action).detach(),
                               advantage[0], value, returns[0], distribution.entropy(), cfg['ppo'])
    policy_objective.backward()
    assert policy.actor[-1].weight.grad.abs().sum() > 0
    assert not advantage.requires_grad
    # Caller-selected, dimensionless geometry checks only the generic wrench map.
    mapping, wrench = unit_load_map([[0., 0., 0., 0., 0., 0.]], [1., 1., 1., 1., 1., 1.], .1)
    assert mapping.shape == (1, 6) and wrench.shape == (1, 6)
    return {'status': 'PASS', 'device': selected, 'checks': 'forward/loss/backward/constraints/decision/GAE'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    result = run_demo(args.device)
    print('PASS: core forward / loss / backward / constraints / decision / GAE; device=' + result['device'])
    print('Synthetic inputs for demonstration only; no paper metrics or files were produced.')


if __name__ == '__main__':
    main()

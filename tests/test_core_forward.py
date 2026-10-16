"""Synthetic inputs for demonstration only: model shape and gradients."""

import pytest
import torch
from examples.minimal_demo import synthetic_inputs, run_demo
from phgt_ppo import ForecastModel, ActorCritic
from phgt_ppo.config import load_demo_config
from phgt_ppo.random_utils import seed_all


def test_forward_actions_and_physical_mapping():
    seed_all(17)
    cfg = load_demo_config()
    batch = synthetic_inputs('cpu')
    model = ForecastModel(18,cfg,local=True).eval()
    mean, scale = torch.zeros(18), torch.ones(18)
    out = model(batch,mean,scale)
    assert out['prediction'].shape == (2,4,18)
    assert torch.allclose(out['physical_tau'],out['mass'].unsqueeze(-1)*batch['jac'])
    assert torch.allclose(out['tau'],.5*(out['raw_tau']+out['physical_tau']))
    assert all(torch.isfinite(value).all() for value in out.values())
    batch['actions'] = 1-batch['actions']
    changed = model(batch,mean,scale)
    assert not torch.allclose(out['prediction'],changed['prediction'])


def test_source_switches_are_real_computations():
    cfg = load_demo_config()
    batch = synthetic_inputs('cpu')
    outputs = []
    for variant in ('full','no_roles','no_relational','no_jacobian'):
        seed_all(17)
        model = ForecastModel(18,cfg,local=True,variant=variant).eval()
        outputs.append(model(batch,torch.zeros(18),torch.ones(18))['prediction'])
    assert all(not torch.allclose(outputs[0],item) for item in outputs[1:])


def test_policy_source_mask():
    model = ActorCritic(load_demo_config())
    state = torch.zeros(2,16)
    state[0,10] = 1.1
    distribution, value = model(state)
    assert distribution.probs.shape == (2,3) and value.shape == (2,)
    assert distribution.probs[0,2] == 1


def test_cpu_minimal_demo():
    assert run_demo('cpu')['status'] == 'PASS'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU unavailable')
def test_gpu_minimal_demo():
    assert run_demo('cuda')['status'] == 'PASS'

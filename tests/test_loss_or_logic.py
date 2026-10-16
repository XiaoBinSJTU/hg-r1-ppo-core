"""Loss derivatives, temporal recurrence and public math utilities."""

import numpy as np
import torch
from examples.minimal_demo import synthetic_inputs
from phgt_ppo import ForecastModel, forecast_loss, dpo_loss, ppo_loss
from phgt_ppo import RewardModel, reward_weights, physical_validate, generalized_advantage_estimate
from phgt_ppo.config import load_demo_config
from phgt_ppo.physics import incidence, unit_load_map
from phgt_ppo.preprocessing import causal_fill, fit_scaler


def test_forecast_backward_and_isolated_unloading():
    cfg = load_demo_config()
    batch = synthetic_inputs('cpu')
    model = ForecastModel(18,cfg,local=True,variant='no_jacobian')
    output = model(batch,torch.zeros(18),torch.ones(18))
    output['mass'] = torch.full_like(output['mass'],2.,requires_grad=True)
    retained = forecast_loss(output,batch,cfg,True,False,True)
    omitted = forecast_loss(output,batch,cfg,True,False,False)
    assert retained > omitted
    retained.backward()
    assert model.decoder[-1].weight.grad.abs().sum() > 0
    assert torch.isfinite(output['mass'].grad).all()


def test_preference_direction_and_backward():
    value = torch.ones(2,requires_grad=True)
    zero = torch.zeros(2)
    favorable = dpo_loss(value,zero,zero,zero,.1)
    unfavorable = dpo_loss(zero,value,zero,zero,.1)
    assert favorable < unfavorable
    favorable.backward()
    assert torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0


def test_reward_constraints_and_density():
    model = RewardModel()
    context = torch.rand(3,10)
    context[:,0] = .9
    weights = reward_weights(model,context,'HG_reward_proxy_PPO')
    assert torch.allclose(weights.sum(1),torch.ones(3))
    assert (weights[:,1] >= weights[:,0]).all()
    assert physical_validate(weights,torch.full((3,),.75),context).all()
    assert not physical_validate(weights,torch.ones(3),context).any()
    assert torch.isfinite(model.log_density(context,torch.full((3,4),.25))).all()


def test_gae_termination_and_bootstrap():
    rewards = torch.tensor([[1.,1.],[2.,2.]])
    values = torch.zeros_like(rewards)
    done = torch.tensor([[False,False],[True,False]])
    advantages, returns = generalized_advantage_estimate(rewards,values,done,torch.tensor([4.,4.]),.5,1.)
    expected = torch.tensor([[2.,3.],[2.,4.]])
    assert torch.equal(advantages,expected) and torch.equal(returns,expected)
    assert not advantages.requires_grad


def test_ppo_clipping_gradient():
    cfg = load_demo_config()['ppo']
    logp = torch.zeros(3,requires_grad=True)
    loss = ppo_loss(logp,torch.zeros(3),torch.tensor([1.,-.5,.2]),torch.zeros(3),torch.ones(3),torch.ones(3),cfg)
    assert torch.isfinite(loss)
    loss.backward()
    assert logp.grad.abs().sum() > 0


def test_incidence_and_generic_geometry():
    h, roles = incidence()
    assert h.shape == roles.shape == (6,7)
    assert set(roles[h.bool()].tolist()) == {0,1,2}
    mapping, wrench = unit_load_map(np.zeros((1,6)),np.ones(6),.1)
    assert mapping.shape == wrench.shape == (1,6)
    assert np.isfinite(mapping).all()
    expected = np.array([[0., 50.031, 40.221, 0., 20.601, 0.]])
    assert np.allclose(mapping,expected,atol=1e-5)
    assert np.allclose(wrench,np.array([[0.,0.,-9.81,0.,.981,0.]]))


def test_causal_preprocessing():
    x = np.array([[1.],[np.nan],[np.nan],[np.nan],[9.]])
    filled = causal_fill(x)
    assert np.allclose(filled[:3,0],[1,1,1]) and np.isnan(filled[3,0])
    mean, scale = fit_scaler(np.array([[0.],[2.],[1000.]]),np.array([0,0,2]))
    assert np.allclose(mean,[1]) and np.allclose(scale,[1])

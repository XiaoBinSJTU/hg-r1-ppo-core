# Method-to-code map

All paths below exist in this package. Section/equation labels describe the related manuscript in the source project's scope notes; 

| Method step | Public implementation | Validation |
|---|---|---|
| Role-constrained hyperedge aggregation, Eq. 2 | `src/phgt_ppo/models.py`: `RoleHypergraph.forward`; `src/phgt_ppo/physics.py`: `incidence` | Finite forward pass, role-switch behavior and gradients |
| Causal relational temporal encoding, Eq. 3 | `src/phgt_ppo/models.py`: `ForecastModel.forward`, `static_layers`, `dynamic_layers`, `temporal` | Full model shape and alternative mechanism tests |
| Action-conditioned load prediction | `src/phgt_ppo/models.py`: action embedding, decoder and physical torque projection | Changing action changes prediction; load identity checked |
| Gravitational wrench/Jacobian mapping, Eq. 1 | `src/phgt_ppo/physics.py`: `unit_load_map` | Caller-chosen toy geometry; mass-linearity identity |
| Observed, Jacobian and unloading objectives, Eq. 4 | `src/phgt_ppo/losses.py`: `forecast_loss` | Finite loss, genuine gradients and unloading-loss isolation |
| Compact structured reward surrogate | `src/phgt_ppo/reward.py`: `RewardModel`, `physical_validate`, `reward_weights` | Simplex, warning check, critical-stage weight repair |
| Preference objective | `src/phgt_ppo/losses.py`: `dpo_loss`; `RewardModel.log_density` | Preference direction and gradient propagation |
| PPO decisions and objective, Eq. 6 | `src/phgt_ppo/ppo.py`: `ActorCritic`; `src/phgt_ppo/losses.py`: `ppo_loss` | Categorical action/value shapes, source safety mask, backward |
| Generalized advantage recurrence | `src/phgt_ppo/ppo.py`: `generalized_advantage_estimate` | Terminal masking, bootstrap and detached recurrence |
| State/context interfaces | `src/phgt_ppo/features.py`: `policy_state`, `reward_context` | 16-value state and 10-value context |
| Generic missingness and scaling | `src/phgt_ppo/preprocessing.py`: `causal_fill`, `fit_scaler` | Causal two-step fill and training-only moments |

## Interface contract

`ForecastModel(channels=18, cfg, method='PHGT', local=True)` consumes a mapping of tensors. Forecast inputs are `x` and `mask` with shape `[batch,16,18]`, `hist_jac` with `[batch,16,6]`, `actions` with `[batch,4]`, and `jac` with `[batch,4,6]`. Normalization mean/scale have 18 entries. `jac` represents unit-mass joint-load mappings for caller-supplied planned poses, not hidden future observations. Outputs are `prediction [batch,4,18]`, positive `mass [batch,4]`, and three torque arrays `[batch,4,6]`.

`forecast_loss` additionally needs caller-provided supervision tensors `y`, `ymask`, `mass`, `tau` and `last_mass`. No labels or data loader are supplied in the repository. The example creates arbitrary tensors only to exercise the differentiable interface.

`RewardModel` accepts `[batch,10]` contexts and returns four normalized weights plus a scalar threshold. `ActorCritic` accepts `[batch,16]` states and returns a three-action distribution and scalar value. Its source mask uses state column 10 as the maximum normalized torque ratio.

`generalized_advantage_estimate` accepts `[time,batch]` rewards, values and boolean terminal flags plus a `[batch]` bootstrap. Outputs are detached raw advantages and returns. Advantage normalization and an optimizer update are responsibilities of caller code.

## Manuscript figures

The README displays these four JMS profile-entry [8] figures in order. Their filenames follow the author's figure index; URLs have not been checked live.

| Figure | Subject | Related implementation |
|---|---|---|
| Fig. 1 | Motivation of slag-removal timing | Problem context; no industrial records or experiment results are included |
| Fig. 2 | Overall framework | `ForecastModel`, `RewardModel`, `ActorCritic` and the loss operators |
| Fig. 3 | Physics-constrained hypergraph state transition modeling | `RoleHypergraph`, `ForecastModel`, `unit_load_map`, `forecast_loss` |
| Fig. 4 | DeepSeek-R1-guided PPO timing optimization | Structured reward surrogate, `dpo_loss`, `ActorCritic`, `ppo_loss` and GAE; full-size language-model training is excluded |

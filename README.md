# PHGT and Reward-Guided PPO: Core Algorithm Reference

[Bin Xiao's GitHub profile](https://github.com/XiaoBinSJTU)

## Paper information

**Related manuscript:** *Proximal policy optimization for slag-removal timing in steelmaking slag-handling robots guided by DeepSeek-R1 fine-tuned under temporal hypergraph constraints*.

**Author:** Bin Xiao. **Journal:** JMS. **Status:** Submitted (unpublished). **GitHub profile entry:** [8]. **Year:** TO_VERIFY; no published DOI is claimed.

This is a core reference extracted from the available research prototype. The PHGT operators, physical losses, structured reward surrogate and PPO mathematics are executable. The package does **not** implement the full-size DeepSeek-R1 domain adaptation pipeline, contain trained weights, or reproduce the paper's experiments.

## Graphical abstract

The author's graphical-abstract index associates these four figures with profile entry **[8] (JMS, submitted)**. They are displayed in manuscript order. **LINK_NOT_LIVE_VERIFIED:** filenames and paths match that index.

### Fig. 1 - Motivation of slag-removal timing

![Fig. 1 - Motivation of slag-removal timing](https://raw.githubusercontent.com/XiaoBinSJTU/XiaoBinSJTU/main/assets/graphical-abstracts/08-jms-slag-removal-motivation.svg)

[View SVG](https://raw.githubusercontent.com/XiaoBinSJTU/XiaoBinSJTU/main/assets/graphical-abstracts/08-jms-slag-removal-motivation.svg)

### Fig. 2 - Overall framework

![Fig. 2 - Overall framework](https://raw.githubusercontent.com/XiaoBinSJTU/XiaoBinSJTU/main/assets/graphical-abstracts/08-jms-slag-removal-framework.svg)

[View SVG](https://raw.githubusercontent.com/XiaoBinSJTU/XiaoBinSJTU/main/assets/graphical-abstracts/08-jms-slag-removal-framework.svg)

### Fig. 3 - Physics-constrained hypergraph state transition modeling

![Fig. 3 - Physics-constrained hypergraph state transition modeling](https://raw.githubusercontent.com/XiaoBinSJTU/XiaoBinSJTU/main/assets/graphical-abstracts/08-jms-slag-removal-state-transition.svg)

[View SVG](https://raw.githubusercontent.com/XiaoBinSJTU/XiaoBinSJTU/main/assets/graphical-abstracts/08-jms-slag-removal-state-transition.svg)

### Fig. 4 - DeepSeek-R1-guided PPO timing optimization

![Fig. 4 - DeepSeek-R1-guided PPO timing optimization](https://raw.githubusercontent.com/XiaoBinSJTU/XiaoBinSJTU/main/assets/graphical-abstracts/08-jms-slag-removal-ppo.svg)

[View SVG](https://raw.githubusercontent.com/XiaoBinSJTU/XiaoBinSJTU/main/assets/graphical-abstracts/08-jms-slag-removal-ppo.svg)

## Research highlights

Slag-cleaning timing connects load propagation, post-intervention recovery and maintenance decisions. This reference exposes role-aware hypergraph aggregation, causal relational temporal encoding, action-conditioned prediction and physical consistency losses. A compact structured reward surrogate illustrates constraint-aware reward interfaces, while an actor-critic model and clipped PPO objective expose decision and optimization logic. The executable example checks these computations using synthetic tensors without making an industrial-performance claim.

## Method overview

### 1. Role-aware physical hypergraph

`RoleHypergraph` applies incidence-role transforms and masked attention over a six-node illustrative topology. The fixed incidence structure is a source implementation assumption, rather than a disclosed industrial system model.

### 2. Relational temporal encoding

`ForecastModel` combines static previous-cycle conditioning and dynamic causal relations before a temporal Transformer. Its decoder conditions four-step predictions on candidate actions and planned unit-load mappings.

### 3. Physical load and recovery objectives

A generic caller-parameterized serial-chain helper computes gravitational wrench mapping. Forecast losses combine masked observation error, auxiliary supervision, Jacobian consistency and unloading constraints. The source's normalization conventions and optional mechanism switches are retained.

### 4. Structured reward and preferences

`RewardModel` produces four simplex weights and a scalar threshold from ten context values. Feasibility checks and weight repair retain the source logic. `dpo_loss` provides the preference objective with a caller-supplied reference. This compact neural surrogate is **not** the manuscript's language model, DAPT/SFT/DPO data pipeline or LoRA training.

### 5. PPO decisions and advantages

`ActorCritic` maps a 16-value state to three discrete actions and a value estimate. The source torque-ratio action mask, clipped PPO loss and detached GAE recurrence are included. Environment rollouts and full policy-training orchestration are excluded.

## Method-to-code map

| Method | Code |
|---|---|
| Hypergraph roles and attention | [models.py](src/phgt_ppo/models.py): `RoleHypergraph` |
| Relational encoding and action-conditioned forecasting | [models.py](src/phgt_ppo/models.py): `ForecastModel` |
| Illustrative incidence and generic load mapping | [physics.py](src/phgt_ppo/physics.py): `incidence`, `unit_load_map` |
| Forecast, DPO and PPO objectives | [losses.py](src/phgt_ppo/losses.py) |
| Structured reward and feasibility | [reward.py](src/phgt_ppo/reward.py) |
| Actor-critic and GAE | [ppo.py](src/phgt_ppo/ppo.py) |
| State/context interfaces | [features.py](src/phgt_ppo/features.py) |
| Causal preprocessing | [preprocessing.py](src/phgt_ppo/preprocessing.py) |

See [METHOD_TO_CODE.md](docs/METHOD_TO_CODE.md) for tensor contracts and validation coverage.

## Quick start - Windows PowerShell

Use an existing compatible Python/PyTorch environment, or prepare a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -B examples/minimal_demo.py
python -B examples/minimal_demo.py --device cpu
python -B -m pytest -q
```

An editable install is optional: `python -m pip install -e .`. The example and tests also work directly from this source tree. No network access, dataset download, pretrained model or account is required at runtime. Dependency installation is the reader's separate setup step; no installer is invoked by the code.

The demo prefers GPU when available and supports CPU. It instantiates PHGT, evaluates the physical loss, executes a genuine gradient update, checks reward constraints and preference gradients, and exercises the actor-critic/PPO/GAE computations. **Synthetic inputs for demonstration only**: arbitrary small tensors test interfaces rather than emulate a case study. No data, weights, numeric losses, performance scores or experiment files are saved. Console output contains pass/fail checks and a neutral device state only.

Verified locally: Python 3.11.5, PyTorch 2.5.1, NumPy 2.3.5, PyYAML 6.0 and pytest 7.4.0. The minimal demo passed on CPU and GPU; 19 operator/release-scope tests passed. No hardware identity is recorded.

The compatibility record is limited to the actually tested environment recorded below. Other combinations are not claimed as tested. Dependencies are intentionally minimal: PyTorch, NumPy, PyYAML and pytest; SciPy, plotting, model-serving and distributed-training packages are unnecessary.


## Author

[Bin Xiao - XiaoBinSJTU](https://github.com/XiaoBinSJTU)

# FML-bench-Lite OfficeHome Baseline Protocol (frozen)

Protocol ID: `fml-lite-officehome-baseline-v1`
Frozen at: 2026-09-11
Status: frozen BEFORE any baseline measurement was observed.

## 1. Upstream provenance

| Item | Value |
|---|---|
| Upstream repository | https://github.com/facebookresearch/DomainBed.git |
| Upstream snapshot commit | `b93c22a1cfc3b2428398272c1a116c8de1f4139e` |
| FML-bench source | https://github.com/qrzou/FML-bench.git |
| FML-bench commit | `d336651ebea50c622c256f02ded82b68b4451fdc` |
| FML task | `ml_tasks/Generalization_domainbed_officehome` |
| FML evaluator script | `train_eval.py`, sha256 `0684a668f9344d5ad1ad477c983a6c367ac4bf1ffbaedb41baf0ef19f5265f5d` |

## 2. Data

| Item | Value |
|---|---|
| Dataset | OfficeHome (65 classes, 4 domains: Art, Clipart, Product, Real World) |
| Target test env | env0 (Art) |
| Source training envs | env1 (Clipart), env2 (Product), env3 (Real World) |
| Split | 30% val (env0 out-split), 70% test (env0 in-split) via holdout_fraction 0.3 |

## 3. Baseline algorithm & model

- Algorithm: ERM (Empirical Risk Minimization)
- Model: ResNet-50
- Steps: 5001

## 4. Metric

- Primary metric: `avg_acc_mean` (generalization accuracy on unseen Art domain).
- Target direction: higher.
- Reference FML baseline: ~0.8339 - 0.8832 on val.

## 5. Worker edit surface

- `domainbed/algorithms.py`
- `domainbed/hparams_registry.py`

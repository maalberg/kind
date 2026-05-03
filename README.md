# Kalman-Inspired Neural Decomposition (KIND)

A hybrid dynamics modeling framework for stable long-horizon prediction under recursive rollout.

## Motivation

Learned dynamics models often achieve low one-step prediction error but fail under recursive application, leading to rollout drift and instability.

KIND addresses this by enforcing structural stability through an uncertainty-triggered decomposition into:
- a contractive nominal model
- a bounded excursion model

This enables consistent long-horizon predictions even in contact-rich systems.

## Repository Contents

- PyTorch implementation of KIND
- Example notebooks for:
  - Kalman filter experiments used in IFAC2026 paper (accepted)
  - Duffing oscillator experiments used in CDC2026 paper (under review)
  - MuJoCo experiments

## Quick Start

Jupyter notebooks are provided in an 'executed' state, so one could start browsing them immediately.

## Results

On MuJoCo tasks Hopper and Walker2d KIND demonstrates:
- Stable rollouts up to 800 steps
- Near-constant MSE over long horizons
- Robustness across random seeds

Baselines (global models, ensembles, MoE) exhibit:
- rapid error growth
- high seed sensitivity
- occasional divergence

## Citation

If you use this code, please cite:

@inproceedings{Maalberg_2026_kind,
  title        = {{KIND}: A {K}alman-inspired Adaptive Estimator for {SRF} Cavity Detuning},
  author       = {Maalberg, Andrei and Neumann, Axel and Echevarria, Pablo and Ushakov, Andriy and Knobloch, Jens},
  booktitle    = {Proc. 23rd IFAC World Congr.},
  note         = {accepted},
  year         = {2026},
}

## Status

This repository is under active development. Additional environments (e.g., Ant, Humanoid) and experiments will be added.

# Double Heston Option Pricing and Hedging

**Author:** Martin Molina-Fructuoso  
**Institution:** Erdos Institute – Quant Finance Bootcamp  
**Date:** November 2025  

---

## Overview

This project implements the **Double Heston stochastic volatility model** for option pricing and delta hedging. The library supports:

- Monte Carlo simulation of single and double Heston models  
- Fourier-based pricing using characteristic functions  
- Calculation of option Greeks  
- Hedging portfolio simulations under both correctly specified and mis-specified models  

The implementation is modular, allowing reproducible experiments and extension to alternative stochastic volatility models.

---

## Features

- **Monte Carlo Pricing:** Euler discretization for single and double Heston models. Convergence behavior verified against analytical results.  
- **Analytical Pricing:** Fourier inversion of characteristic functions for European options.  
- **Delta Hedging:** Simulation of hedging portfolios under various drift and volatility scenarios.  
- **Model Comparison:** Evaluation of hedging performance under simpler Heston specifications.  

---

## Results Summary

- **Monte Carlo Convergence:** Standard deviation of the estimator decreases approximately as \(1/\sqrt{N}\). Small Euler discretization bias observed for low step counts.  
- **Delta Hedging:** Hedging portfolios closely track option prices; terminal hedging errors converge to zero with finer rebalancing grids. Deviations under extreme drift values are consistent with theoretical expectations.  
- **Hedging Mis-specification:** Using simpler Heston models for hedging significantly increases terminal error, highlighting the importance of correctly capturing stochastic volatility factors.

---

## Quick Usage Example

```python
from double_heston import DoubleHeston
import numpy as np

params = {
    'v0_1': 0.04, 'kappa_1': 2.0, 'theta_1': 0.04, 'sigma_1': 0.3, 'rho_1': -0.7,
    'v0_2': 0.02, 'kappa_2': 1.0, 'theta_2': 0.02, 'sigma_2': 0.2, 'rho_2': 0.5,
    'r': 0.01, 'q': 0.0
}

model = DoubleHeston(params)

S0 = 100
K = 100
T = 1.0
n_paths = 100_000
n_steps = 252

price = model.price_mc(S0=S0, K=K, T=T, n_paths=n_paths, n_steps=n_steps)
delta = model.delta_mc(S0=S0, K=K, T=T, n_paths=n_paths, n_steps=n_steps)

print(f"Monte Carlo price: {price:.4f}")
print(f"Monte Carlo delta: {delta:.4f}")


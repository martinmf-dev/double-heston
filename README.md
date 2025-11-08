# Executive Summary  
## The Double Heston Method  
**Author:** Martin Molina-Fructuoso  
**Institution:** Erdos Institute - Quant Finance Bootcamp  

---

This project explored option pricing and delta hedging under the **Double Heston stochastic volatility model**, combining Monte Carlo simulations, analytical methods, and hedging experiments to analyze model accuracy and robustness.

---

## Monte Carlo Price and Convergence

**Objective:**  
Assess the convergence of Monte Carlo estimators for Double Heston option prices against closed-form and high precision MC estimate.

**Outcome:**  
- Standard deviation decreases roughly as $1/\sqrt{N}$, confirming expected convergence.  
- Comparison to large-path MC confirms stable convergence; small Euler discretization bias noted.

**Key Tasks:**  
- Monte Carlo implementation  
- Numerical convergence analysis  
- Variance behavior assessment

---

## Delta Hedging under Neutral and Biased Drift

**Objective:**  
Evaluate hedging performance using the correct Double Heston model for different drift scenarios.

**Outcome:**  
- Spread decreases as number of rebalancing steps increases; portfolio tracks option price closely.  
- Extreme drift values exhibit small deviations; t-tests mostly confirm mean-zero hedging error.

**Key Tasks:**  
- Delta hedging implementation  
- Statistical evaluation of hedging errors  
- Stochastic volatility modeling

---

## Hedging Mis-specification

**Objective:**  
Investigate the effect of using incorrect hedging models (simpler Heston variants) on hedging performance.

**Outcome:**  
- Correct Double Heston produces terminal error close to zero, symmetric distribution.  
- Demonstrates that ignoring stochastic volatility factors can significantly degrade hedging performance.

**Key Tasks:**  
- Model misspecification analysis  
- Risk assessment  
- Comparative hedging evaluation

---

## Future Extensions

- **Alternative discretization schemes:** Use Milstein, Quadratic Exponential (QE), or other higher-order methods to reduce bias in Monte Carlo simulations.  
- **Additional model features:** Incorporate stochastic interest rates, jumps, or correlated factors; analyze sensitivity to parameters.  
- **Market calibration:** Fit the Double Heston model to real option prices.  
- **Implied volatility surfaces:** Produce full surfaces across strikes and maturities for risk management, pricing, and scenario analysis.  
- **Extended hedging strategies:** Test delta, vega, and other Greeks under alternative hedging approaches and more realistic market assumptions.

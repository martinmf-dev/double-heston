import numpy as np
from scipy.stats import norm

class BS:
    """
    Black-Scholes model for European option pricing and path simulation.

    Parameters
    ----------
    r : float, default=0.03
        Risk-free interest rate.
    q : float, default=0.02
        Dividend yield.
    sigma : float, default=0.2
        Constant volatility of the asset.
    """

    def __init__(self, r=0.03, q=0.02, sigma=0.2):
        self.model_type = 'bs'
        self.r = r
        self.q = q
        self.sigma = sigma

    def price_greeks_vect(self, K, Tau, S):
        """
        Vectorized Black-Scholes call price and delta for multiple paths and steps.

        Parameters
        ----------
        K : float
            Option strike.
        Tau : array, shape (N_paths, N_steps)
            Time to maturity for each path/step.
        S : array, shape (N_paths, N_steps)
            Spot prices.

        Returns
        -------
        dict with keys:
        - 'Price_call': array, shape (N_paths, N_steps)
        - 'Delta': array, shape (N_paths, N_steps)
        """
        sigma = self.sigma
        r, q = self.r, self.q

        sqrt_Tau = np.sqrt(np.maximum(Tau, 1e-12))  # avoid division by zero
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * Tau) / (sigma * sqrt_Tau)
        d2 = d1 - sigma * sqrt_Tau

        Price_call = S * np.exp(-q * Tau) * norm.cdf(d1) - K * np.exp(-r * Tau) * norm.cdf(d2)
        Delta = np.exp(-q * Tau) * norm.cdf(d1)

        return {"Price_call": Price_call, "Delta": Delta}

    def simulate_paths(self, mu, T, S0, N_paths, N_steps, seed):
        """
        Simulate asset paths under Black-Scholes using geometric Brownian motion with general drift.

        Parameters
        ----------
        mu : float
            Drift of the asset (set mu=r for risk-neutral simulation)
        T : float
            Time horizon.
        S0 : float
            Initial asset price.
        N_paths : int
            Number of Monte Carlo paths.
        N_steps : int
            Number of time steps.
        seed : int
            Random seed.

        Returns
        -------
        S : ndarray
            Simulated asset price paths, shape (N_paths, N_steps)
        """
        q, sigma = self.q, self.sigma
        rng = np.random.default_rng(seed)
        dt = T / (N_steps - 1)

        Z = rng.standard_normal((N_paths, N_steps))
        S = np.zeros((N_paths, N_steps))
        S[:, 0] = S0

        for t in range(1, N_steps):
            S[:, t] = S[:, t-1] * np.exp((mu - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t])

        return S

    def mc_price(self, K, T, S0, N_paths, N_steps, seed):
        """
        Compute the European call price using Monte Carlo simulation.

        Parameters
        ----------
        K : float
            Strike price.
        T : float
            Time to maturity.
        S0 : float
            Initial asset price.
        N_paths : int
            Number of Monte Carlo paths.
        N_steps : int
            Number of time steps.
        seed : int
            Random seed.

        Returns
        -------
        call_price : float
            Monte Carlo estimate of the European call option price.
        """
        S_paths = self.simulate_paths(mu=self.r, T=T, S0=S0, N_paths=N_paths, N_steps=N_steps, seed=seed)
        S_T = S_paths[:, -1]
        payoffs = np.maximum(S_T - K, 0)
        return np.exp(-self.r * T) * np.mean(payoffs)
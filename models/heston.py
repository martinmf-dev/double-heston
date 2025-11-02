import numpy as np

class Heston:
    """
    Heston stochastic volatility model for option pricing and path simulation.

    Parameters
    ----------
    r : float, default=0.03
        Risk-free interest rate.
    q : float, default=0.02
        Dividend yield.
    kappa : float, default=5.0
        Mean-reversion speed of the variance.
    theta : float, default=0.05
        Long-run average variance (variance level).
    sigma : float, default=0.5
        Volatility of variance (vol of vol).
    rho : float, default=-0.8
        Correlation between the asset price and variance Brownian motions.
    """

    # Reference values. T=0.5, S=100, K=100, q=0.02, r=0.03, kappa=5, sigma=0.5, rho=-0.8, theta=v0=0.05, 
    # trapezoid rule with phi in [0.00001,50], increments of 0.001, 
    # Put: 5.7590, Call: 6.2528
    # If q=0 (no dividend)
    # Put: 5.3790, Call: 6.8678
    
    def __init__(self, r=0.03, q=0.02, kappa=5.0, theta=0.05, sigma=0.5, rho=-0.8):
        self.r = r
        self.q = q
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho

    def cf(self, phis, Pnum, K, tau, S, v):
        """
        Compute the integrand of the characteristic function for the risk-neutral probabilities.
    
        This integrand is used in the semi-analytical Heston pricing formula.
    
        Parameters
        ----------
        phis : array_like
            Array of integration variable phi values.
        Pnum : int
            1 or 2, selects which risk-neutral probability (P1 or P2).
        K : float
            Strike price of the option.
        tau : float
            Time to maturity (T - t).
        S : float
            Spot price at the current time.
        v : float
            Spot variance at the current time.
    
        Returns
        -------
        ndarray
            Real part of the integrand evaluated at each phi.
        """
        
        r, q = self.r, self.q
        kappa, theta, sigma, rho = self.kappa, self.theta, self.sigma, self.rho

        
        x = np.log(S)
        
        if Pnum==1:
            b = kappa - rho*sigma
            u = 1/2
        elif Pnum==2:
            b= kappa
            u = -1/2
        else:
            raise ValueError(f"Invalid Pnum: {Pnum}. Must be 1 or 2.")
    
        d = np.sqrt((rho*sigma*1j*phis-b)**2-sigma**2*(2*u*1j*phis-phis**2))
        c = (b-rho*sigma*1j*phis-d)/(b-rho*sigma*1j*phis+d)
    
        D = (b-rho*sigma*1j*phis-d)/sigma**2*((1-np.exp(-d*tau))/(1-c*np.exp(-d*tau)))
        G = (1-c*np.exp(-d*tau))/(1-c)
        C = (r-q)*1j*phis*tau + (kappa*theta)/sigma**2*((b-rho*sigma*1j*phis-d)*tau-2*np.log(G))
    
        return np.exp(C+D*v+1j*phis*x)

    def cf_vect(self, phis, Pnum, K, tau, S, v):
        """
        Compute the integrand of the characteristic function for the risk-neutral probabilities (vectorized).
    
        Parameters
        ----------
        phis : array, shape (N_phi,)
            Integration variable.
        Pnum : int
            1 or 2, selects which risk-neutral probability (P1 or P2).
        K : float
            Strike price (unused in cf itself but kept for signature compatibility).
        tau : array, shape (N_paths, N_steps)
            Time to maturity for each path/step.
        S : array, shape (N_paths, N_steps)
            Spot prices.
        v : array, shape (N_paths, N_steps)
            Spot variances.
    
        Returns
        -------
        array, shape (N_paths, N_steps, N_phi)
            Characteristic function evaluated at each phi for all paths and steps.
        """
        r, q = self.r, self.q
        kappa, theta, sigma, rho = self.kappa, self.theta, self.sigma, self.rho
    
        # reshape for broadcasting
        S = S[:, :, np.newaxis]  # (N_paths, N_steps, 1)
        v = v[:, :, np.newaxis]  # (N_paths, N_steps, 1)
        tau = tau[:, :, np.newaxis]  # (N_paths, N_steps, 1)
        phis = phis[np.newaxis, np.newaxis, :]  # (1, 1, N_phi)
    
        x = np.log(S)
    
        if Pnum == 1:
            b = kappa - rho * sigma
            u = 0.5
        elif Pnum == 2:
            b = kappa
            u = -0.5
        else:
            raise ValueError("Pnum must be 1 or 2.")
    
        d = np.sqrt((rho * sigma * 1j * phis - b) ** 2 - sigma ** 2 * (2 * u * 1j * phis - phis ** 2))
        c = (b - rho * sigma * 1j * phis - d) / (b - rho * sigma * 1j * phis + d)
    
        D = (b - rho * sigma * 1j * phis - d) / sigma ** 2 * ((1 - np.exp(-d * tau)) / (1 - c * np.exp(-d * tau)))
        G = (1 - c * np.exp(-d * tau)) / (1 - c)
        C = (r - q) * 1j * phis * tau + (kappa * theta) / sigma ** 2 * ((b - rho * sigma * 1j * phis - d) * tau - 2 * np.log(G))
    
        return np.exp(C + D * v + 1j * phis * x)
    
    def cf_price(self, Lphi,Uphi,dphi, K, tau, S, v):
        """
        Compute the European call option price using the Heston characteristic function.
    
        Parameters
        ----------
        Lphi : float
            Lower bound of integration for phi.
        Uphi : float
            Upper bound of integration for phi.
        dphi : float
            Step size for numerical integration.
        K : float
            Strike price.
        tau : float
            Time to maturity.
        S : float
            Spot price.
        v : float
            Spot variance.
    
        Returns
        -------
        call_price : float
            The Heston call option price.
        """
        
        r, q = self.r, self.q
        
        # Integration grid
        phis = np.arange(Lphi, Uphi, dphi)


        f1 = self.cf(phis=phis, Pnum=1, K=K, tau=tau, S=S, v=v)
        f2 = self.cf(phis=phis, Pnum=2, K=K, tau=tau, S=S, v=v)

        int1 = np.real(np.exp(-1j*phis*np.log(K))*f1/(1j*phis))
        int2 = np.real(np.exp(-1j*phis*np.log(K))*f2/(1j*phis))

        
        # Integrals
        I1 = np.trapezoid(int1, dx=dphi)
        I2 = np.trapezoid(int2, dx=dphi)
    
        # Probabilities
        P1 = 0.5 + I1 / np.pi
        P2 = 0.5 + I2 / np.pi
    
        # Call price
        call_price = S * np.exp(-q * tau) * P1 - K * np.exp(-r * tau) * P2

        return call_price

    def price_greeks(self, Lphi, Uphi, dphi, K, tau, S, v):
        """
        Compute call price and delta (vectorized over phis, single path S/v).

        Returns a dictionary with:
        - "call_price": call price
        - "delta": Δ = exp(-q*tau)*P1
        """
        r, q = self.r, self.q

        phis = np.arange(Lphi, Uphi, dphi)

        f1 = self.cf(phis=phis, Pnum=1, K=K, tau=tau, S=S, v=v)
        f2 = self.cf(phis=phis, Pnum=2, K=K, tau=tau, S=S, v=v)

        int1 = np.real(np.exp(-1j * phis * np.log(K)) * f1 / (1j * phis))
        int2 = np.real(np.exp(-1j * phis * np.log(K)) * f2 / (1j * phis))

        I1 = np.trapezoid(int1, dx=dphi)
        I2 = np.trapezoid(int2, dx=dphi)

        P1 = 0.5 + I1 / np.pi
        P2 = 0.5 + I2 / np.pi

        call_price = S * np.exp(-q * tau) * P1 - K * np.exp(-r * tau) * P2
        delta = np.exp(-q * tau) * P1

        return {"call_price": call_price, "delta": delta}

    def price_greeks_vect(self, Lphi, Uphi, dphi, K, tau, S, v):
        """
        Vectorized Heston call price and delta for multiple paths and steps.
    
        Parameters
        ----------
        Lphi, Uphi, dphi : floats
            Integration limits and step size.
        K : float
            Option strike.
        tau : array, shape (N_paths, N_steps)
            Time to maturity for each path/step.
        S : array, shape (N_paths, N_steps)
            Spot prices.
        v : array, shape (N_paths, N_steps)
            Spot variances.
        model : Heston instance
    
        Returns
        -------
        dict with keys:
        - 'call_price': array, shape (N_paths, N_steps)
        - 'delta': array, shape (N_paths, N_steps)
        """
        r, q = self.r, self.q
        phis = np.arange(Lphi, Uphi, dphi)
    
        f1 = self.cf_vect(phis, Pnum=1, K=K, tau=tau, S=S, v=v)
        f2 = self.cf_vect(phis, Pnum=2, K=K, tau=tau, S=S, v=v)
    
        # Compute integrand
        exp_term = np.exp(-1j * phis * np.log(K)) / (1j * phis)
        int1 = np.real(exp_term[np.newaxis,np.newaxis,:] * f1)
        int2 = np.real(exp_term[np.newaxis,np.newaxis,:] * f2)
    
        # Integrate along phi axis
        I1 = np.trapezoid(int1, dx=dphi, axis=2)
        I2 = np.trapezoid(int2, dx=dphi, axis=2)
    
        P1 = 0.5 + I1 / np.pi
        P2 = 0.5 + I2 / np.pi
    
        call_price = S * np.exp(-q * tau) * P1 - K * np.exp(-r * tau) * P2
        delta = np.exp(-q * tau) * P1
    
        return {"call_price": call_price, "delta": delta}
        
    def simulate_paths(self, N_paths, N_steps, T, S0, v0, seed):
        """
        Simulate asset and variance paths using the Euler-Maruyama scheme for the Heston model.
    
        Parameters
        ----------
        N_paths : int
            Number of Monte Carlo paths.
        N_steps : int
            Number of time steps.
        T : float
            Time horizon.
        S0 : float
            Initial asset price.
        v0 : float
            Initial variance.
    
        Returns
        -------
        S : ndarray
            Simulated asset price paths, shape (N_paths, N_steps).
        V : ndarray
            Simulated variance paths, shape (N_paths, N_steps).
        """
        
        r, q = self.r, self.q
        kappa, theta, sigma, rho = self.kappa, self.theta, self.sigma, self.rho
        
        # Define Brownian noises for all paths and steps
        rng = np.random.default_rng(seed)
        Z_1 = rng.standard_normal((N_paths, N_steps))
        Z_2 = rng.standard_normal((N_paths, N_steps))
        Z_V = Z_1
        Z_S = rho* Z_V + np.sqrt(1-rho**2)*Z_2
    
        # Euler scheme
        dt= T/(N_steps-1)
        
        S = np.zeros((N_paths, N_steps))
        V = np.zeros((N_paths, N_steps))
    
        S[:,0]= S0
        V[:,0]= v0
    
        for step in range(1, N_steps):
            V[:,step] = V[:,step-1] + kappa*(theta-V[:,step-1])*dt+sigma*np.sqrt(V[:,step-1])*np.sqrt(dt)*Z_V[:, step]
            V[:,step] = np.maximum(0,V[:,step])
            S[:,step] = S[:, step-1]*np.exp((r-q-1/2*V[:,step-1])*dt + np.sqrt(V[:,step-1])*np.sqrt(dt)*Z_S[:,step])
        return S, V
    
    def mc_price(self, N_paths, N_steps, K, T, S0, v0,seed):
        """
        Compute the European call price using Monte Carlo simulation.
    
        Parameters
        ----------
        N_paths : int
            Number of Monte Carlo paths.
        N_steps : int
            Number of time steps.
        K : float
            Strike price.
        T : float
            Time to maturity.
        S0 : float
            Initial asset price.
        v0 : float
            Initial variance.
    
        Returns
        -------
        call_price : float
            Monte Carlo estimate of the European call option price.
        """
        
        r = self.r
        S_paths = self.simulate_paths(N_paths=N_paths, N_steps=N_steps, T=T, S0=S0, v0=v0, seed=seed)[0]
        S_T = S_paths[:,-1]
        payoffs = np.maximum (S_T-K,0)
        return np.exp(-r*T)*np.mean(payoffs)


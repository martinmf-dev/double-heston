import numpy as np

class DoubleHeston:
    """
    Double Heston stochastic volatility model for option pricing and path simulation.

    Parameters
    ----------
    r : float, default=0.03
        Risk-free interest rate.
    kappa1 : float, default=5.0
        Mean-reversion speed of variance 1.
    kappa2 : float, default=5.0
        Mean-reversion speed of variance 2.
    theta1 : float, default=0.05
        Long-run average variance 1 (variance level).
    theta2 : float, default=0.05
        Long-run average variance 2 (variance level).
    sigma1 : float, default=0.5
        Volatility of variance 1 (vol of vol).
    sigma2 : float, default=0.5
        Volatility of variance 2 (vol of vol).
    rho1 : float, default=-0.8
        Correlation between Brownian motions
    rho2 : float, default=-0.8
        Correlation between Brownian motions
    """


    # Reference values. T=0.5, S=100, K=100, q=0.02, r=0.03, kappa=5, sigma=0.5, rho=-0.8, theta=v0=0.05, 
    # trapezoid rule with phi in [0.00001,50], increments of 0.001, 
    # Put: 5.7590, Call: 6.2528
    # If q=0 (no dividend)
    # Put: 5.3790, Call: 6.8678
    
    def __init__(self, r=0.03, kappa1=0.9, kappa2=1.2, theta1=0.1, theta2=0.15, sigma1=0.1, sigma2=0.2, rho1=-0.5, rho2=-0.5):
        self.r = r
        self.kappa1 = kappa1
        self.kappa2 = kappa2
        self.theta1 = theta1
        self.theta2 = theta2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho1 = rho1
        self.rho2 = rho2

    # def cf_integrand(self, phis, Pnum, K, tau, S, v):
    #     """
    #     Compute the integrand of the characteristic function for the risk-neutral probabilities.
    
    #     This integrand is used in the semi-analytical Heston pricing formula.
    
    #     Parameters
    #     ----------
    #     phis : array_like
    #         Array of integration variable phi values.
    #     Pnum : int
    #         1 or 2, selects which risk-neutral probability (P1 or P2).
    #     K : float
    #         Strike price of the option.
    #     tau : float
    #         Time to maturity (T - t).
    #     S : float
    #         Spot price at the current time.
    #     v : float
    #         Spot variance at the current time.
    
    #     Returns
    #     -------
    #     integrand : ndarray
    #         Real part of the integrand evaluated at each phi.
    #     """
        
    #     r, q = self.r, self.q
    #     kappa, theta, sigma, rho = self.kappa, self.theta, self.sigma, self.rho

        
    #     x = np.log(S)
        
    #     if Pnum==1:
    #         b = kappa - rho*sigma
    #         u = 1/2
    #     elif Pnum==2:
    #         b= kappa
    #         u = -1/2
    #     else:
    #         raise ValueError(f"Invalid Pnum: {Pnum}. Must be 1 or 2.")
    
    #     d = np.sqrt((rho*sigma*1j*phis-b)**2-sigma**2*(2*u*1j*phis-phis**2))
    #     c = (b-rho*sigma*1j*phis-d)/(b-rho*sigma*1j*phis+d)
    
    #     D = (b-rho*sigma*1j*phis-d)/sigma**2*((1-np.exp(-d*tau))/(1-c*np.exp(-d*tau)))
    #     G = (1-c*np.exp(-d*tau))/(1-c)
    #     C = (r-q)*1j*phis*tau + (kappa*theta)/sigma**2*((b-rho*sigma*1j*phis-d)*tau-2*np.log(G))
    
    #     f = np.exp(C+D*v+1j*phis*x)
    
    #     integrand = np.exp(-1j*phis*np.log(K))*f/(1j*phis)
        
    #     return integrand.real
    
    # def cf_price(self, Lphi,Uphi,dphi, K, tau, S, v):
    #     """
    #     Compute the European call option price using the Heston characteristic function.
    
    #     Parameters
    #     ----------
    #     Lphi : float
    #         Lower bound of integration for phi.
    #     Uphi : float
    #         Upper bound of integration for phi.
    #     dphi : float
    #         Step size for numerical integration.
    #     K : float
    #         Strike price.
    #     tau : float
    #         Time to maturity.
    #     S : float
    #         Spot price.
    #     v : float
    #         Spot variance.
    
    #     Returns
    #     -------
    #     call_price : float
    #         The Heston call option price.
    #     """
        
    #     r, q = self.r, self.q
        
    #     # Integration grid
    #     phis = np.arange(Lphi, Uphi, dphi)
    
    #     int1 = self.cf_integrand(phis=phis, Pnum=1, K=K, tau=tau, S=S, v=v)
    #     int2 = self.cf_integrand(phis=phis, Pnum=2, K=K, tau=tau, S=S, v=v)
        
    #     # Integrals
    #     I1 = np.trapezoid(int1, dx=dphi)
    #     I2 = np.trapezoid(int2, dx=dphi)
    
    #     # Probabilities
    #     P1 = 0.5 + I1 / np.pi
    #     P2 = 0.5 + I2 / np.pi
    
    #     # Call price
    #     call_price = S * np.exp(-q * tau) * P1 - K * np.exp(-r * tau) * P2

    #     return call_price

    def simulate_paths(self, N_paths, N_steps, T, S0, v01, v02, seed):
        """
        Simulate asset and variance paths using the scheme by Gauthier and Possamai for the Double Heston model.
    
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
        v01 : float
            Initial value of variance 1.
        v01 : float
            Initial value of variance 2.
    
        Returns
        -------
        S : ndarray
            Simulated asset price paths, shape (N_paths, N_steps).
        V1 : ndarray
            Simulated variance 1 paths, shape (N_paths, N_steps).
        V2 : ndarray
            Simulated variance 2 paths, shape (N_paths, N_steps).
        """
        
        r = self.r
        kappa1, theta1, sigma1, rho1 = self.kappa1, self.theta1, self.sigma1, self.rho1
        kappa2, theta2, sigma2, rho2 = self.kappa2, self.theta2, self.sigma2, self.rho2

        dt= T/(N_steps-1)
        
        # Constants required for motion correlations
        K01 = -(rho1*kappa1*theta1)/sigma1*dt
        K11 = dt/2*(kappa1*rho1/sigma1 -1/2)-rho1/sigma1
        K21 = dt/2*(kappa1*rho1/sigma1 - 1/2)+rho1/sigma1
        K31 = dt/2*(1-rho1**2)

        K02 = -(rho2*kappa2*theta2)/sigma2*dt
        K12 = dt/2*(kappa2*rho2/sigma2-1/2)-rho2/sigma2
        K22 = dt/2*(kappa2*rho2/sigma2-1/2)+rho2/sigma2
        K32 = dt/2*(1-rho2**2)

        
        # Define Brownian motions
        rng = np.random.default_rng(seed)
        # Brownian motions for the price
        B1 = rng.standard_normal((N_paths, N_steps))
        B2 = rng.standard_normal((N_paths, N_steps))
        # Brownian motions for the variances
        G1 = rng.standard_normal((N_paths, N_steps))
        G2 = rng.standard_normal((N_paths, N_steps))
    
        S = np.zeros((N_paths, N_steps))
        V1 = np.zeros((N_paths, N_steps))
        V2 = np.zeros((N_paths, N_steps))
    
        S[:,0]= S0
        V1[:,0]= v01
        V2[:,0]= v02
    
        for step in range(1, N_steps):
            V1[:,step] = V1[:,step-1] + kappa1*(theta1-V1[:,step-1])*dt + sigma1*np.sqrt(V1[:,step-1])*np.sqrt(dt)*G1[:,step]
            V1[:,step] = np.maximum(0,V1[:,step])
            V2[:,step] = V2[:,step-1] + kappa2*(theta2-V2[:,step-1])*dt + sigma2*np.sqrt(V2[:,step-1])*np.sqrt(dt)*G2[:,step]
            V2[:,step] = np.maximum(0,V2[:,step])
            S[:,step] = np.exp(r*(step)*dt)*np.exp(np.log(np.exp(-r*(step-1)*dt)*S[:,step-1])
                                                 +K01+K11*V1[:,step-1]+K21*V1[:,step]
                                                 +np.sqrt(K31*(V1[:,step-1]+V1[:,step]))*B1[:,step]
                                                 +K02+K12*V2[:,step-1]+K22*V2[:,step]
                                                 +np.sqrt(K32*(V2[:,step-1]+V2[:,step]))*B2[:,step]
                                                )
        return S, V1, V2
    
    def mc_price(self, N_paths, N_steps, K, T, S0, v01, v02,seed):
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
        v01 : float
            Initial variance 1.
        v02 : float
            Initial variance 2.
    
        Returns
        -------
        call_price : float
            Monte Carlo estimate of the European call option price.
        """
        
        r = self.r
        S_paths = self.simulate_paths(N_paths=N_paths, N_steps=N_steps, T=T, S0=S0, v01=v01, v02=v02, seed=seed)[0]
        S_T = S_paths[:,-1]
        payoffs = np.maximum (S_T-K,0)
        return np.exp(-r*T)*np.mean(payoffs)


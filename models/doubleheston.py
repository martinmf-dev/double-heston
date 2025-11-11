import numpy as np

class DoubleHeston:
    """
    Double Heston stochastic volatility model for option pricing and path simulation.

    Parameters
    ----------
    r : float, default=0.03
        Risk-free interest rate.
    q : float, default=0.0
        Dividend yield.
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
    
    def __init__(self, r=0.03, q=0.0, kappa1=0.9, kappa2=1.2, theta1=0.1, theta2=0.15, sigma1=0.1, sigma2=0.2, rho1=-0.5, rho2=-0.5):
        self.model_type = 'doubleheston'
        self.r = r
        self.q = q
        self.kappa1 = kappa1
        self.kappa2 = kappa2
        self.theta1 = theta1
        self.theta2 = theta2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho1 = rho1
        self.rho2 = rho2

    
    def cf_vect(self, Tau, S, V1, V2, Phi):
        """
        Compute the integrand of the characteristic function for the risk-neutral probabilities (vectorized).
    
        This integrand is used in the semi-analytical double Heston pricing formula.
    
        Parameters
        ----------
        Phi : array, shape (N_phi,)
            Array of integration variable phi values.
        Tau : array, shape (N_paths, N_steps)
            Time to maturity (T - t) for each path/ step.
        S : array, shape (N_paths, N_steps)
            Spot prices.
        V1 : array, shape (N_paths, N_steps)
            Spot variances 1.
        V2 : array, shape (N_paths, N_steps)
            Spot variances 2.
    
        Returns
        -------
        array, shape (N_pahts, N_steps, N_phi)
            Characteristic function evaluated at each phi for all paths and steps.
        """
        
        r, q = self.r, self.q
        kappa1, theta1, sigma1, rho1 = self.kappa1, self.theta1, self.sigma1, self.rho1
        kappa2, theta2, sigma2, rho2 = self.kappa2, self.theta2, self.sigma2, self.rho2

        # reshape for broadcasting
        S = S[:, :, np.newaxis]  # (N_paths, N_steps, 1)
        V1 = V1[:, :, np.newaxis]  # (N_paths, N_steps, 1)
        V2 = V2[:, :, np.newaxis]  # (N_paths, N_steps, 1)
        Tau = Tau[:, np.newaxis]  
        Phi = Phi[ np.newaxis, :]  
        
        X = np.log(S)

        d1 = np.sqrt((kappa1-rho1*sigma1*Phi*1j)**2+sigma1**2*Phi*(Phi+1j))
        d2 = np.sqrt((kappa2-rho2*sigma2*Phi*1j)**2+sigma2**2*Phi*(Phi+1j))
        c1 = (kappa1-rho1*sigma1*Phi*1j-d1)/(kappa1-rho1*sigma1*Phi*1j+d1)
        c2 = (kappa2-rho2*sigma2*Phi*1j-d2)/(kappa2-rho2*sigma2*Phi*1j+d2)

        B1 = (kappa1-rho1*sigma1*Phi*1j-d1)/(sigma1**2)*((1-np.exp(-d1*Tau))/(1-c1*np.exp(-d1*Tau)))
        B2 = (kappa2-rho2*sigma2*Phi*1j-d2)/(sigma2**2)*((1-np.exp(-d2*Tau))/(1-c2*np.exp(-d2*Tau)))
        G1 = (1-c1*np.exp(-d1*Tau))/(1-c1)
        G2 = (1-c2*np.exp(-d2*Tau))/(1-c2)

        A =( (r-q)*Phi*1j*Tau + (kappa1*theta1)/sigma1**2*((kappa1-rho1*sigma1*Phi*1j-d1)*Tau-2*np.log(G1))
                               + (kappa2*theta2)/sigma2**2*((kappa2-rho2*sigma2*Phi*1j-d2)*Tau-2*np.log(G2)))
    
        return np.exp(A[np.newaxis,:,:]+1j*Phi[np.newaxis,:,:]*X+B1[np.newaxis,:,:]*V1+B2[np.newaxis,:,:]*V2)

    
    def price_greeks_vect(self, K, Tau, S, V1, V2, quad_rule, quad_params):
        """
        Vectorized Double Heston call price and delta for multiple paths and steps.

        Returns
        ---------------
        dict with keys:
        - 'Price_call': array, shape (N_paths, N_steps)
        - 'Delta': array, shape (N_paths, N_steps)
        """

        r, q = self.r, self.q

        if quad_rule == 'trapezoidal':
            required = {'Lphi', 'Uphi', 'dphi'}
            missing = required - quad_params.keys()
            if missing:
                raise ValueError(f"missing parameters for trapezoidal quadrature rule: {missing}")

            Lphi = quad_params['Lphi']
            Uphi = quad_params['Uphi']
            dphi = quad_params['dphi']
            
            # Integration grid
            Phi = np.arange(Lphi, Uphi, dphi)
    
            f1 = self.cf_vect(Phi=Phi-1j, Tau=Tau, S=S, V1=V1, V2=V2)
            f2 = self.cf_vect(Phi=Phi, Tau=Tau, S=S, V1=V1, V2=V2)
            exp_term = np.exp(-1j*Phi*np.log(K))/(1j*Phi)
            int1 = np.real(exp_term[np.newaxis,np.newaxis,:]*f1/(S[:,:,np.newaxis]*np.exp((r-q)*Tau[np.newaxis,:,np.newaxis])))
            int2 = np.real(exp_term[np.newaxis,np.newaxis,:]*f2)
         
            # Integrals
            I1 = np.trapezoid(int1, dx=dphi, axis=2)
            I2 = np.trapezoid(int2, dx=dphi, axis=2)

        elif quad_rule == 'laguerre':
            required = {'nodes'}
            missing = required - quad_params.keys()
            if missing:
                raise ValueError(f"missing parameters for laguerre quadrature rule: {missing}")
            
            nodes = quad_params['nodes']
            Phi, W = np.polynomial.laguerre.laggauss(nodes)

            f1 = self.cf_vect(Phi=Phi-1j, Tau=Tau, S=S, V1=V1, V2=V2)
            f2 = self.cf_vect(Phi=Phi, Tau=Tau, S=S, V1=V1, V2=V2)
            exp_term = np.exp(-1j*Phi*np.log(K))/(1j*Phi)
            
            exp_term = exp_term[np.newaxis,np.newaxis,:]
            Phi = Phi[np.newaxis,np.newaxis,:]
            W = W[np.newaxis,np.newaxis,:]
            
            int1 = np.real(W*np.exp(Phi)*exp_term*f1/(S[:,:,np.newaxis]*np.exp((r-q)*Tau[np.newaxis,:,np.newaxis])))
            int2 = np.real(W*np.exp(Phi)*exp_term*f2)

            I1 = np.sum(int1, axis=2)
            I2 = np.sum(int2, axis=2)
            
        else:
            raise ValueError("quad_rule must be 'trapezoidal' or 'laguerre'")
        
        # Probabilities
        P1 = 0.5 + I1 / np.pi
        P2 = 0.5 + I2 / np.pi
    
        # Call price
        Price_call = S * np.exp(-q * Tau) * P1 - K * np.exp(-r * Tau) * P2
        Delta = np.exp(-q*Tau)*P1

        return {"Price_call": Price_call, "Delta": Delta}

    
    def simulate_paths(self,mu,  T, S0, v01, v02, N_paths, N_steps, seed):
        """
        Simulate asset and variance paths using the scheme by Gauthier and Possamai for the Double Heston model.
    
        Parameters
        ----------
       mu: float
            Drift, set mu=r for risk neutral
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
        
        q = self.q
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
            S[:,step] = np.exp((mu-q)*(step)*dt)*np.exp(np.log(np.exp(-(mu-q)*(step-1)*dt)*S[:,step-1])
                                                 +K01+K11*V1[:,step-1]+K21*V1[:,step]
                                                 +np.sqrt(K31*(V1[:,step-1]+V1[:,step]))*B1[:,step]
                                                 +K02+K12*V2[:,step-1]+K22*V2[:,step]
                                                 +np.sqrt(K32*(V2[:,step-1]+V2[:,step]))*B2[:,step])
                                                
        return S, V1, V2

    
    def mc_price(self, K, T, S0, v01, v02, N_paths, N_steps, seed):
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
        S_paths = self.simulate_paths(mu=r,N_paths=N_paths, N_steps=N_steps, T=T, S0=S0, v01=v01, v02=v02, seed=seed)[0]
        S_T = S_paths[:,-1]
        payoffs = np.maximum (S_T-K,0)
        return np.exp(-r*T)*np.mean(payoffs)

    
##########################################################################
## Deprecated functions kept just for reference, to be removed eventually
##########################################################################

    
    def cf(self, phis, Pnum, K, tau, S, v1, v2):
        """
        Compute the integrand of the characteristic function for the risk-neutral probabilities.
    
        This integrand is used in the semi-analytical double Heston pricing formula.
    
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
        v1 : float
            Spot variance 1 at the current time.
        v2 : floate
            Spot variance 2 at the current time.
    
        Returns
        -------
        integrand : ndarray
            Real part of the integrand evaluated at each phi.
        """
        
        r, q = self.r, self.q
        kappa1, theta1, sigma1, rho1 = self.kappa1, self.theta1, self.sigma1, self.rho1
        kappa2, theta2, sigma2, rho2 = self.kappa2, self.theta2, self.sigma2, self.rho2
    
        x = np.log(S)

        d1 = np.sqrt((kappa1-rho1*sigma1*phis*1j)**2+sigma1**2*phis*(phis+1j))
        d2 = np.sqrt((kappa2-rho2*sigma2*phis*1j)**2+sigma2**2*phis*(phis+1j))
        c1 = (kappa1-rho1*sigma1*phis*1j-d1)/(kappa1-rho1*sigma1*phis*1j+d1)
        c2 = (kappa2-rho2*sigma2*phis*1j-d2)/(kappa2-rho2*sigma2*phis*1j+d2)

        B1 = (kappa1-rho1*sigma1*phis*1j-d1)/(sigma1**2)*((1-np.exp(-d1*tau))/(1-c1*np.exp(-d1*tau)))
        B2 = (kappa2-rho2*sigma2*phis*1j-d2)/(sigma2**2)*((1-np.exp(-d2*tau))/(1-c2*np.exp(-d2*tau)))
        G1 = (1-c1*np.exp(-d1*tau))/(1-c1)
        G2 = (1-c2*np.exp(-d2*tau))/(1-c2)

        A =( (r-q)*phis*1j*tau + (kappa1*theta1)/sigma1**2*((kappa1-rho1*sigma1*phis*1j-d1)*tau-2*np.log(G1))
                               + (kappa2*theta2)/sigma2**2*((kappa2-rho2*sigma2*phis*1j-d2)*tau-2*np.log(G2)))
    
        return np.exp(A+1j*phis*x+B1*v1+B2*v2)

    
    def cf_price(self, Lphi,Uphi,dphi, K, tau, S, v1, v2):
        """
        Compute the European call option price using the Double Heston characteristic function.
    
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
        v1: float
            Spot variance 1.
        v2: float
            Spot variance 2.
    
        Returns
        -------
        call_price : float
            The Double Heston call option price.
        """
        
        r, q = self.r, self.q
        
        # Integration grid
        phis = np.arange(Lphi, Uphi, dphi)

        f1 = self.cf(phis=phis-1j, Pnum=1, K=K, tau=tau, S=S, v1=v1, v2=v2)
        f2 = self.cf(phis=phis, Pnum=1, K=K, tau=tau, S=S, v1=v1, v2=v2)
        int1 = np.real(np.exp(-1j*phis*np.log(K))*f1/(1j*phis*S*np.exp((r-q)*tau)))
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


    def price_greeks(self, Lphi, Uphi, dphi, K, tau, S, v1, v2):
        """
        Compute call price and delta (vectorized over phis, single path S/v).

        Returns a dictionary with:
        - "call price": call price
        - "delta": Δ = exp(-q*tau)*P1
        """

        r, q = self.r, self.q
        
        # Integration grid
        phis = np.arange(Lphi, Uphi, dphi)

        f1 = self.cf(phis=phis-1j, Pnum=1, K=K, tau=tau, S=S, v1=v1, v2=v2)
        f2 = self.cf(phis=phis, Pnum=1, K=K, tau=tau, S=S, v1=v1, v2=v2)
        int1 = np.real(np.exp(-1j*phis*np.log(K))*f1/(1j*phis*S*np.exp((r-q)*tau)))
        int2 = np.real(np.exp(-1j*phis*np.log(K))*f2/(1j*phis))
     
        # Integrals
        I1 = np.trapezoid(int1, dx=dphi)
        I2 = np.trapezoid(int2, dx=dphi)
    
        # Probabilities
        P1 = 0.5 + I1 / np.pi
        P2 = 0.5 + I2 / np.pi
    
        # Call price
        call_price = S * np.exp(-q * tau) * P1 - K * np.exp(-r * tau) * P2
        delta = np.exp(-q*tau)*P1

        return {"call_price": call_price, "delta": delta}

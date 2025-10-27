# Older implementations

# Hestonfunctions written directly in the notebook

def heston_cf_integrand(phis, Pnum, K, tau, S, v, r=0.03, q=0.02, kappa=5.0, theta=0.05, sigma=0.5, rho=-0.8):
    """
    Returns the integrand for the risk neutral probabilities P1 and P2 needed in the formula for the price at the points specified by phis
    
    Parameters
    ----------
    phis : An array of values of phi (the integration variable)
    Pnum : 1 or 2 (to choose P1 or P2)
    
    K: Strike price
    S: Spot price (Price at time t)
    tau: Time to maturity (T-t)
    r: Risk free rate
    q: Dividend yield

    v: Spot variance
    kappa: volatility mean reversion speed
    theta: volatility mean reversion level
    sigma: volatility of the variance
    rho: correlation between the Brownian motions
    """

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

    f = np.exp(C+D*v+1j*phis*x)

    integrand = np.exp(-1j*phis*np.log(K))*f/(1j*phis)
    
    return integrand.real

def heston_cf_price(Lphi,Uphi,dphi, K, tau, S, v, r=0.03, q=0.02, kappa=5.0, theta=0.05, sigma=0.5, rho=-0.8):
 
    # Integration grid
    phis = np.arange(Lphi, Uphi + dphi, dphi)
    N = len(phis)

    # Integrands for P1 and P2 (real-valued)
    int1 = np.zeros(N)
    int2 = np.zeros(N)

    int1 = heston_cf_integrand(phis, 1, K, tau, S, v, r, q, kappa, theta, sigma, rho)
    int2 = heston_cf_integrand(phis, 2, K, tau, S, v, r, q, kappa, theta, sigma, rho)
    
    # Integrals
    I1 = np.trapezoid(int1, dx=dphi)
    I2 = np.trapezoid(int2, dx=dphi)

    # Probabilities
    P1 = 0.5 + I1 / np.pi
    P2 = 0.5 + I2 / np.pi

    # Call price
    call_price = S * np.exp(-q * tau) * P1 - K * np.exp(-r * tau) * P2

    return call_price

def heston_simulate_paths(N_paths, N_steps, T, S0, v0, r=0.03, q=0.02, kappa=5.0, theta=0.05, sigma=0.5, rho=-0.8):
    
    # Define Brownian noises for all paths and steps
    rng = np.random.default_rng(20)
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
        V[:,step] = np.maximum(V[:,step],0)
        S[:,step] = S[:, step-1]*np.exp((r-q-1/2*V[:,step-1])*dt + np.sqrt(V[:,step-1])*np.sqrt(dt)*Z_S[:,step])
    return S, V

def heston_mc_price(S_paths, K, T, r=0.03):
    S_T = S_paths[:,-1]
    payoffs = np.maximum (S_T-K,0)
    return np.exp(-r*T)*np.mean(payoffs)

# Reference values. T=0.5, S=100, K=100, q=0.02, r=0.03, kappa=5, sigma=0.5, rho=-0.8, theta=v0=0.05, 
# trapezoid rule with phi in [0.00001,50], increments of 0.001, 
# Put: 5.7590, Call: 6.2528
# If q=0 (no dividend)
# Put: 5.3790, Call: 6.8678


# Plots paths
import matplotlib.pyplot as plt
import numpy as np

def plot_paths(S=None, V=None):
    """
    Plots simulated Heston paths using the 'tab20c' colormap.
    
    Parameters
    ----------
    S : np.array or None
        Stock price paths, shape (N_paths, N_steps)
    V : np.array or None
        Variance paths, shape (N_paths, N_steps)
    """
    if S is None and V is None:
        raise ValueError("At least one of S or V must be provided.")
    
    cmap = plt.get_cmap('tab20')
    
    if S is not None:
        plt.figure(figsize=(10, 5))
        n_paths = S.shape[0]
        for i in range(n_paths):
            plt.plot(S[i,:], color=cmap(i / n_paths), alpha=0.8)
        plt.title('Stock Paths')
        plt.xlabel('Time step')
        plt.ylabel('Stock price')
        plt.show()
    
    if V is not None:
        plt.figure(figsize=(10, 5))
        n_paths = V.shape[0]
        for i in range(n_paths):
            plt.plot(V[i,:], color=cmap(i / n_paths), alpha=0.8)
        plt.title('Variance Paths')
        plt.xlabel('Time step')
        plt.ylabel('Variance')
        plt.show()

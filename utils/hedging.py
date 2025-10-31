import numpy as np

def delta_hedge_onepath(model, model_type, K, N_steps, T, S0, seed, **kwargs):
    """
    Delta hedging for a single path (seller's point of view) with stepwise cash discounting.
    
    Parameters
    ----------
    model : instance
        A model implementing .simulate_paths() and .price_greeks().
    model_type : str
        Either 'heston' or 'doubleheston'.
    K : float
        Option strike.
    N_steps : int
        Number of time steps.
    T : float
        Time to maturity.
    S0 : float
        Initial stock price.
    seed : int
        Random seed.
    **kwargs : dict
        Model-specific parameters: 
        - v0 (Heston) 
        - v01, v02 (DoubleHeston)
    """
    # integration parameters for CF pricing
    Lphi = 1e-5
    Uphi = 50
    dphi = 0.001
    
    # 1. simulate one path
    sim_paths = model.simulate_paths(N_paths=1, N_steps=N_steps, T=T, S0=S0, seed=seed, **kwargs)

    # select paths according to model type
    if model_type == 'heston':
        S_path, V_path = sim_paths
        S_path = S_path[0]
        V_path = V_path[0]
    elif model_type == 'doubleheston':
        S_path, V1_path, V2_path = sim_paths
        S_path = S_path[0]
        V1_path = V1_path[0]
        V2_path = V2_path[0]
    else:
        raise ValueError("model_type must be 'heston' or 'doubleheston'")

    dt = T / (N_steps - 1)
    r = model.r

    # time to maturity
    tau = (N_steps - 1 - np.arange(N_steps)) * dt

    # initialize arrays for recording
    opt_price = np.zeros(N_steps)
    delta = np.zeros(N_steps)
    cash = np.zeros(N_steps)
    portfolio = np.zeros(N_steps)

    # initial greeks, delta, and cash for chosen model
    if model_type == 'heston':
        greeks = model.price_greeks(Lphi=Lphi, Uphi=Uphi, dphi=dphi, K=K, tau=tau[0], S=S_path[0], v=V_path[0])
    else:
        greeks = model.price_greeks(Lphi=Lphi, Uphi=Uphi, dphi=dphi, K=K, tau=tau[0], S=S_path[0], v1=V1_path[0], v2=V2_path[0])
        

    opt_price[0] = greeks["call_price"]
    delta[0] = greeks["delta"]
    cash[0] = greeks["call_price"] - delta[0] * S_path[0]  # initial hedge

    # initial portfolio value
    portfolio[0] = delta[0] * S_path[0] + cash[0]

    # hedging loop
    for t in range(1, N_steps - 1):  # stop before maturity
        # accrue interest on cash
        cash[t] = cash[t-1] * np.exp(r * dt)

        # compute new delta
        if model_type == 'heston':
            greeks = model.price_greeks(Lphi=Lphi, Uphi=Uphi, dphi=dphi, K=K, tau=tau[t], S=S_path[t], v=V_path[t])
        else:
            greeks = model.price_greeks(Lphi=Lphi, Uphi=Uphi, dphi=dphi, K=K, tau=tau[t], S=S_path[t], v1=V1_path[t], v2=V2_path[t])
        
        delta[t] = greeks["delta"]
        opt_price[t] = greeks["call_price"]
        
        # rebalance stock
        delta_change = delta[t] - delta[t-1]
        cash[t] -= delta_change * S_path[t]

        # portfolio value after rebalancing
        portfolio[t] = delta[t] * S_path[t] + cash[t]

    # at maturity: accrue final interest
    cash[-1] = cash[-2] * np.exp(r * dt)
    delta[-1] = delta[-2]  # no need to rebalance
    portfolio[-1] = delta[-1] * S_path[-1] + cash[-1]

    # compute liability and hedging error
    liability_T = max(S_path[-1] - K, 0)
    opt_price[-1] = liability_T
    hedging_error = portfolio[-1] - liability_T

    # choose which V_path(s) to return
    if model_type == 'heston':
        V_paths_return = V_path
    else:
        V_paths_return = (V1_path, V2_path)
    
    return {
        "S_path": S_path,
        "V_path": V_paths_return,
        "opt_price": opt_price,
        "delta": delta,
        "cash": cash,
        "portfolio": portfolio,
        "liability_T": liability_T,
        "hedging_error": hedging_error,
    }

# def delta_hedge_onepath(model, K, N_steps, T, S0, v0, seed):
#     """
#     Delta hedging for a single path (seller's point of view) with stepwise cash discounting.

#     Parameters
#     ----------
#     model : instance
#         A model implementing .simulate_paths() and .price_greeks().
#     K : float
#         Option strike.
#     N_steps : int
#         Number of time steps.
#     T : float
#         Time to maturity.
#     S0, v0 : float
#         Initial stock price and variance.
#     seed : int
#         Random seed.

#     Returns
#     -------
#     portfolio_T : float
#         Hedged portfolio value at maturity.
#     liability_T : float
#         Option payoff at maturity.
#     hedging_error : float
#         Difference between portfolio and liability.
#     """
#     # integration parameters for CF pricing
#     Lphi = 1e-5
#     Uphi = 50
#     dphi = 0.001
    
#     # 1. simulate one path
#     S_path, V_path = model.simulate_paths(N_paths=1, N_steps=N_steps, T=T, S0=S0, v0=v0, seed=seed)
#     S_path = S_path[0]  # shape (N_steps,)
#     V_path = V_path[0]

#     dt = T / (N_steps - 1)
#     r = model.r

#     # time to maturity
#     tau = (N_steps - 1 - np.arange(N_steps)) * dt

#     # initialize arrays for recording
#     opt_price = np.zeros(N_steps)
#     delta = np.zeros(N_steps)
#     cash = np.zeros(N_steps)
#     portfolio = np.zeros(N_steps)

#     # initial greeks, delta, and cash
#     greeks = model.price_greeks(Lphi=Lphi, Uphi=Uphi, dphi=dphi, K=K, tau=tau[0], S=S_path[0], v=V_path[0])
#     opt_price[0] = greeks["call_price"]
#     delta[0] = greeks["delta"]
#     cash[0] = greeks["call_price"] - delta[0] * S_path[0]  # initial hedge

#     # initial portfolio value
#     portfolio[0] = delta[0] * S_path[0] + cash[0]

#     # hedging loop
#     for t in range(1, N_steps - 1):  # stop before maturity
#         # accrue interest on cash
#         cash[t] = cash[t-1] * np.exp(r * dt)

#         # compute new delta
#         greeks = model.price_greeks(Lphi=Lphi, Uphi=Uphi, dphi=dphi, K=K, tau=tau[t], S=S_path[t], v=V_path[t])
#         delta[t] = greeks["delta"]
#         opt_price[t] = greeks["call_price"]
        
#         # rebalance stock
#         delta_change = delta[t] - delta[t-1]
#         cash[t] -= delta_change * S_path[t]

#         # portfolio value after rebalancing
#         portfolio[t] = delta[t] * S_path[t] + cash[t]

#     # at maturity: accrue final interest
#     cash[-1] = cash[-2] * np.exp(r * dt)
#     delta[-1] = delta[-2]  # no need to rebalance
#     portfolio[-1] = delta[-1] * S_path[-1] + cash[-1]

#     # compute liability and hedging error
#     liability_T = max(S_path[-1] - K, 0)
#     opt_price[-1] = liability_T
#     hedging_error = portfolio[-1] - liability_T

#     return {
#         "S_path": S_path,
#         "V_path": V_path,
#         "opt_price": opt_price,
#         "delta": delta,
#         "cash": cash,
#         "portfolio": portfolio,
#         "liability_T": liability_T,
#         "hedging_error": hedging_error,
#     }

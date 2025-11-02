import numpy as np
import matplotlib.pyplot as plt

def hedge_plot(opt_price, portfolio):
    plt.figure(figsize=(10,5))
    plt.plot(opt_price, label="Model option price $C_t$")
    plt.plot(portfolio, '--', label="Hedging portfolio")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.title("Portfolio vs model option price")
    plt.legend()
    plt.grid(True)
    plt.show()

def delta_hedge(model, model_type, N_paths, N_steps, K, T, S0, seed, **kwargs):
    """
    Delta hedging for a single path (seller's point of view) with stepwise cash discounting. Price and delta are vectorized.
    
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
    sim_paths = model.simulate_paths(N_paths=N_paths, N_steps=N_steps, T=T, S0=S0, seed=seed, **kwargs)

    # select paths according to model type
    if model_type == 'heston':
        S_path, V_path = sim_paths
    elif model_type == 'doubleheston':
        S_path, V1_path, V2_path = sim_paths
    else:
        raise ValueError("model_type must be 'heston' or 'doubleheston'")

    dt = T / (N_steps - 1)
    r = model.r

    # time to maturity
    tau = (N_steps - 1 - np.arange(N_steps)) * dt
    tau = np.tile(tau, (N_paths, 1))

    # initialize arrays for recording
    cash = np.zeros((N_paths, N_steps))
    portfolio = np.zeros((N_paths, N_steps))
    liability_T = np.zeros(N_paths)
    hedging_error = np.zeros(N_paths)

    if model_type == 'heston':
        greeks = model.price_greeks_vect(Lphi=Lphi, Uphi=Uphi, dphi=dphi, K=K, tau=tau, S=S_path, v=V_path)
    elif model_type == 'doubleheston':
        greeks = model.price_greeks_vect(Lphi=Lphi, Uphi=Uphi, dphi=dphi, K=K, tau=tau, S=S_path, v1=V1_path, v2=V2_path)
    else:
        raise ValueError("model_type must be 'heston' or 'doubleheston'")

    opt_price = greeks["call_price"]
    delta = greeks["delta"]

    cash[:,0] = opt_price[:,0] - delta[:,0]*S_path[:,0]
    portfolio[:,0]=delta[:,0]*S_path[:,0]+ cash[:,0]
    
    for t in range(1, N_steps-1): # stop before maturity
        cash[:,t] = cash[:,t-1] * np.exp(r*dt)
        delta_change = delta[:,t]-delta[:,t-1]
        cash[:,t] -= delta_change*S_path[:,t] 

        # Portfolio value after rebalancing
        portfolio[:,t] = delta[:,t]*S_path[:,t]+cash[:,t]

    # At maturity accrue final interest
    cash[:,-1] = cash[:,-2]*np.exp(r*dt)
    portfolio[:,-1] = delta[:,-2]*S_path[:,-1]+cash[:,-1]

    # Compute liability and hedging error
    liability_T = np.maximum(S_path[:,-1]-K,0)
    hedging_error = portfolio[:,-1]- liability_T

    # choose which V_path(s) to return
    if model_type == 'heston':
        V_paths_return = V_path
    else:
        V_paths_return = (V1_path, V2_path)

    print(np.mean(hedging_error), np.std(hedging_error))
    
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

# def delta_hedge_loops(model, model_type, N_paths, N_steps, K, T, S0, seed, **kwargs):
#     """
#     Delta hedging for a single path (seller's point of view) with stepwise cash discounting.
    
#     Parameters
#     ----------
#     model : instance
#         A model implementing .simulate_paths() and .price_greeks().
#     model_type : str
#         Either 'heston' or 'doubleheston'.
#     K : float
#         Option strike.
#     N_steps : int
#         Number of time steps.
#     T : float
#         Time to maturity.
#     S0 : float
#         Initial stock price.
#     seed : int
#         Random seed.
#     **kwargs : dict
#         Model-specific parameters: 
#         - v0 (Heston) 
#         - v01, v02 (DoubleHeston)
#     """
#     # integration parameters for CF pricing
#     Lphi = 1e-5
#     Uphi = 50
#     dphi = 0.001
    
#     # 1. simulate one path
#     sim_paths = model.simulate_paths(N_paths=N_paths, N_steps=N_steps, T=T, S0=S0, seed=seed, **kwargs)

#     # select paths according to model type
#     if model_type == 'heston':
#         S_path, V_path = sim_paths
#     elif model_type == 'doubleheston':
#         S_path, V1_path, V2_path = sim_paths
#     else:
#         raise ValueError("model_type must be 'heston' or 'doubleheston'")

#     dt = T / (N_steps - 1)
#     r = model.r

#     # time to maturity
#     tau = (N_steps - 1 - np.arange(N_steps)) * dt

#     # initialize arrays for recording
#     opt_price = np.zeros((N_paths, N_steps))
#     delta = np.zeros((N_paths, N_steps))
#     cash = np.zeros((N_paths, N_steps))
#     portfolio = np.zeros((N_paths, N_steps))

#     liability_T = np.zeros(N_paths)
#     hedging_error = np.zeros(N_paths)
    
#     for i in range(N_paths):
#         # initial greeks, delta, and cash for chosen model
#         if model_type == 'heston':
#             greeks = model.price_greeks(Lphi=Lphi, Uphi=Uphi, dphi=dphi, K=K, tau=tau[0], S=S_path[i,0], v=V_path[i,0])
#         else:
#             greeks = model.price_greeks(Lphi=Lphi, Uphi=Uphi, dphi=dphi, K=K, tau=tau[0], S=S_path[i,0], v1=V1_path[i,0], v2=V2_path[i,0])
            
#         opt_price[i,0] = greeks["call_price"]
#         delta[i,0] = greeks["delta"]
#         cash[i,0] = greeks["call_price"] - delta[i,0] * S_path[i,0]  # initial hedge
    
#         # initial portfolio value
#         portfolio[i,0] = delta[i,0] * S_path[i,0] + cash[i,0]
    
#         # hedging loop
#         for t in range(1, N_steps - 1):  # stop before maturity
#             # accrue interest on cash
#             cash[i,t] = cash[i,t-1] * np.exp(r * dt)
    
#             # compute new delta
#             if model_type == 'heston':
#                 greeks = model.price_greeks(Lphi=Lphi, Uphi=Uphi, dphi=dphi, K=K, tau=tau[t], S=S_path[i,t], v=V_path[i,t])
#             else:
#                 greeks = model.price_greeks(Lphi=Lphi, Uphi=Uphi, dphi=dphi, K=K, tau=tau[t], S=S_path[i,t], v1=V1_path[i,t], v2=V2_path[i,t])
            
#             delta[i,t] = greeks["delta"]
#             opt_price[i,t] = greeks["call_price"]
            
#             # rebalance stock
#             delta_change = delta[i,t] - delta[i,t-1] # Note that delta_change depends on i
#             cash[i,t] -= delta_change * S_path[i,t]
    
#             # portfolio value after rebalancing
#             portfolio[i,t] = delta[i,t] * S_path[i,t] + cash[i,t]


#         if model_type == 'heston':
#             greeks = model.price_greeks(Lphi=Lphi, Uphi=Uphi, dphi=dphi, K=K, tau=tau[-1], S=S_path[i,-1], v=V_path[i,-1])
#         else:
#             greeks = model.price_greeks(Lphi=Lphi, Uphi=Uphi, dphi=dphi, K=K, tau=tau[-1], S=S_path[i,-1], v1=V1_path[i,-1], v2=V2_path[i,-1])
            
#         opt_price[i,-1] = greeks["call_price"]
#         delta[i,-1] = greeks["delta"] 
        
#         # at maturity: accrue final interest
#         cash[i,-1] = cash[i,-2] * np.exp(r * dt)     
#         portfolio[i,-1] = delta[i,-2] * S_path[i,-1] + cash[i,-1]
    
#         # compute liability and hedging error
#         liability_T[i] = max(S_path[i,-1] - K, 0)
#         hedging_error[i] = portfolio[i,-1] - liability_T[i]

#         print(f"path {i} finished")

#     # choose which V_path(s) to return
#     if model_type == 'heston':
#         V_paths_return = V_path
#     else:
#         V_paths_return = (V1_path, V2_path)

#     print(np.mean(hedging_error), np.std(hedging_error))
    
#     return {
#         "S_path": S_path,
#         "V_path": V_paths_return,
#         "opt_price": opt_price,
#         "delta": delta,
#         "cash": cash,
#         "portfolio": portfolio,
#         "liability_T": liability_T,
#         "hedging_error": hedging_error,
#     }

# def price_greeks_vect_test(model, model_type, N_paths, N_steps, K, T, S0, seed, **kwargs):
    
#     if model_type=='heston':
#         S,V=model.simulate_paths(N_paths=N_paths, N_steps=N_steps, T=T, S0=S0, seed=seed, **kwargs)
#     elif model_type=='doubleheston':
#         S,V1,V2 = model.simulate_paths(N_paths=N_paths, N_steps=N_steps, T=T, S0=S0, seed=seed, **kwargs)
#     else:
#         raise ValueError("model_type must be 'heston' or 'doubleheston'")
        
#     dt = T / (N_steps - 1)
#     tau = (N_steps - 1 - np.arange(N_steps)) * dt # shape (N_steps,) 
#     tau = np.tile(tau, (N_paths, 1))

#     if model_type == 'heston':
#         greeks_vect=model.price_greeks_vect(Lphi=1e-5, Uphi=50, dphi=0.001, K=K, tau=tau, S=S, v=V)
#     elif model_type=='doubleheston':
#         greeks_vect=model.price_greeks_vect(Lphi=1e-5, Uphi=50, dphi=0.001, K=K, tau=tau, S=S, v1=V1, v2=V2)
#     else:
#         raise ValueError("model_type must be 'heston' or 'doubleheston'")

#     return greeks_vect

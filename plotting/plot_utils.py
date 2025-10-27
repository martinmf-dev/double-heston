import matplotlib.pyplot as plt
import numpy as np

def plot_paths(S=None, V=None, title=None):
    """
    Plots simulated Heston paths using the 'tab20' colormap.

    Parameters
    ----------
    S : np.array or None
        Stock price paths, shape (N_paths, N_steps)
    V : np.array or None
        Variance paths, shape (N_paths, N_steps)
    title : str or None, optional
        Overall figure title (displayed at the top)
    """
    if S is None and V is None:
        raise ValueError("At least one of S or V must be provided.")
    
    cmap = plt.get_cmap('tab20')

    # Case 1: both S and V are provided → horizontal layout (1x2)
    if S is not None and V is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
        n_paths_S = S.shape[0]
        n_paths_V = V.shape[0]

        for i in range(n_paths_S):
            axes[0].plot(S[i, :], color=cmap(i / n_paths_S), alpha=0.8)
        axes[0].set_title('Stock Paths')
        axes[0].set_xlabel('Time step')
        axes[0].set_ylabel('Stock price')

        for i in range(n_paths_V):
            axes[1].plot(V[i, :], color=cmap(i / n_paths_V), alpha=0.8)
        axes[1].set_title('Variance Paths')
        axes[1].set_xlabel('Time step')
        axes[1].set_ylabel('Variance')

        if title:
            fig.suptitle(title, fontsize=14)

        fig.tight_layout(rect=[0, 0, 1, 1])
        plt.show()

    # Case 2: only S
    elif S is not None:
        plt.figure(figsize=(10, 5))
        n_paths = S.shape[0]
        for i in range(n_paths):
            plt.plot(S[i, :], color=cmap(i / n_paths), alpha=0.8)
        plt.title(title or 'Stock Paths')
        plt.xlabel('Time step')
        plt.ylabel('Stock price')
        plt.show()

    # Case 3: only V
    elif V is not None:
        plt.figure(figsize=(10, 5))
        n_paths = V.shape[0]
        for i in range(n_paths):
            plt.plot(V[i, :], color=cmap(i / n_paths), alpha=0.8)
        plt.title(title or 'Variance Paths')
        plt.xlabel('Time step')
        plt.ylabel('Variance')
        plt.show()
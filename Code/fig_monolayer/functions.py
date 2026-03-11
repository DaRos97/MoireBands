import numpy as np
import scipy
import CORE_functions as cfs


def weight_spreading(weight,K_temp,E_temp,K_list,e_grid,pars_spread):
    """Compute the weight spreading in k and e.

    Parameters
    ----------
    weight : float
        Weight to spread.
    K : float
        Momentum position of weight.
    E : float
        Energy position of weight.
    k_grid : np.ndarray
        Grid of values over which evaluate the spreading in momentum.
    e_grid : np.ndarray
        Grid of values over which evaluate the spreading in energy.
    pars_spread : tuple
        Parameters of spreading: gamma_k, gamma_e, type_of_spread (Gauss or Lorentz).

    Returns
    -------
    np.ndarray
        Grid of energy and momentum values over which the weight located at K,E has been spread using the type_of_spread function by values spread_K and spread_E.
    """
    spread_k,spread_E,type_of_spread = pars_spread
    k_grid = np.linalg.norm(K_list-K_temp,axis=1)[:,None]
    if type_of_spread == 'Lorentz':
        E2 = spread_E**2
        K2 = spread_k**2
        return weight/(k_grid**2+K2)/((e_grid-E_temp)**2+E2)
    elif type_of_spread == 'Gauss':
        return weight*np.exp(-(k_grid/spread_k)**2)*np.exp(-((e_grid-E_temp)/spread_E)**2)

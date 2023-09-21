"""TAKE Fitting Programme"""

# Imports
import copy
import math
import numpy as np
import pandas as pd
import io
import itertools
from scipy import optimize
from skopt import gp_minimize  # install as scikit-optimize module
import logging
import warnings
from take_2 import take_prep_2 as take_prep
from take_2 import take_plotting_2 as take_plot
import timeit

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


# rate equation - manipulate equation using k[x], conc[x] and ord[x] as required
#def rate_eq(k, conc, ord, temp):
#    return k[0] * np.prod([conc[j] ** ord[j] for j in range(len(ord))])

def rate_eq_custom(k, conc, rxns_r, ord, temp):
    print("Manipulate equation using k[x], conc[x], rxns_r[x], ord[x] and temp as required")
    return


def rate_eq_standard(k, conc, rxns_r, ord, temp):
    return k[0] * np.prod([conc[rxns_r[i]] ** ord[i] for i in range(len(ord))])


def rate_eq_Arrhenius(k, conc, rxns_r, ord, temp):
    return k[0] * math.exp(-k[1] / (temp * 8.314462)) * np.prod([conc[rxns_r[i]] ** ord[i] for i in range(len(ord))])


def rate_eq_Eyring(k, conc, rxns_r, ord, temp):
    return ((1 * 1.380649E-23 * temp) / 6.626070E-34) * math.exp(k[0] / 8.314462) * math.exp(-k[1] / (temp * 8.314462)) \
           * np.prod([conc[rxns_r[i]] ** ord[i] for i in range(len(ord))])


def rate_eq_MM(k, conc, ord, temp):
    return k[0] * (conc[0] ** ord[0]) * conc[3] * (conc[1] / (k[1] + conc[1]))


func_map = {
    "standard": rate_eq_standard,
    "Arrhenius": rate_eq_Arrhenius,
    "Eyring": rate_eq_Eyring,
    "MM": rate_eq_MM,
    "custom": rate_eq_custom
}


# general kinetic simulator using Euler method
def eq_sim_gen(stoich, mol0, mol_end, add_pops, vol, vol_loss_rat, temp, rate_eq, inc, rate_loc, rxns_r, k, ord,
               var_locs, t_span, fit_param, fit_param_locs):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    k_adj = copy.deepcopy(k)  # added to prevent editing of variables
    ord_adj = copy.deepcopy(ord)  # added to prevent editing of variables
    var_k = [fit_param[i] for i in fit_param_locs[0]]
    for i, (j, m) in enumerate(var_locs[0]):
        k_adj[j][m] = var_k[i]
    var_ord = [fit_param[i] for i in fit_param_locs[1]]
    for i, (j, m) in enumerate(var_locs[1]):
        ord_adj[j][m] = var_ord[i]
    pois = [fit_param[i] for i in fit_param_locs[2]]
    pops = np.zeros((len(t_span) + 1, len(mol0)))
    rates = np.zeros((len(t_span) + 1, len(ord_adj)))
    pops[0] = mol0
    for i in range(len(var_locs[2])):
        pops[:, var_locs[2][i]] -= pois[i]
    # rates[i - 1] = rate_calc(i - 1, k_adj, pops, vol, rxns_r, ord_adj, temp, rate_eq)
    # pops[i, :] = mol_calc(i, pops, vol_loss_rat, t_span[i - 1], rates, stoich, vol, add_pops, rate_loc)
    # mol_calc(i, pops, vol_loss_rat, t_span[i - 1], rates[i - 1, :], stoich, vol, add_pops, rate_loc)
    # rates[i - 1] = rk_rate_calc(i - 1, k_adj, pops, vol, rxns_r, ord_adj, temp, rate_eq, vol_loss_rat, t_span[i - 1], stoich, add_pops, rate_loc)
    for i in range(1, len(t_span) + 1):
        rates[i - 1] = rate_calc(i - 1, k_adj, pops, vol, rxns_r, ord_adj, temp, rate_eq)
        pops[i, :] = mol_calc(i, pops, vol_loss_rat, t_span[i - 1], rates, stoich, vol, add_pops, rate_loc)
    rates[i] = rate_calc(i, k_adj, pops, vol, rxns_r, ord_adj, temp, rate_eq)
    pops[pops < 0] = 0
    pops[:] = [pops[i, :] / vol[i] for i in range(0, len(t_span) + 1)]
    exp_t_rows = list(range(0, len(t_span) + 1, inc - 1))
    pops, rates = pops[exp_t_rows], rates[exp_t_rows]
    return [pops, rates]


def rate_calc(i, k, pops, vol, rxns_r, ord, temp, rate_eq):
    conc = np.divide(pops[i, :], vol[i])
    # conc = np.divide(pops, vol[i])
    conc[conc < 0] = 0
    return [rate_eq(k[j], conc, rxns_r[j], ord[j], temp[i]) for j in range(len(k))]


def mol_calc(i, conc, vol_loss_rat, t_span, rates, stoich, vol, add_pops, rate_loc):
    # return [conc[i - 1, j] * vol_loss_rat[i] + t_span * sum([rates[rate_loc[j][m]] * stoich[j][m]
    #        for m in range(len(rate_loc[j]))]) * vol[i - 1] + add_pops[i, j] for j in range(len(rate_loc))]
    return [conc[i - 1, j] * vol_loss_rat[i] + t_span * sum([rates[i - 1, rate_loc[j][m]] * stoich[j][m]
            for m in range(len(rate_loc[j]))]) * vol[i - 1] + add_pops[i, j] for j in range(len(rate_loc))]


def rk_rate_calc(i, k, pops, vol, rxns_r, ord, temp, rate_eq, vol_loss_rat, t_span, stoich, add_pops, rate_loc):
    k1 = rate_calc(i, k, pops[i, :], vol, rxns_r, ord, temp, rate_eq)
    k2 = rate_calc(i, k, mol_calc(i, pops, vol_loss_rat, t_span, [m / 2 for m in k1], stoich, vol, add_pops, rate_loc), vol,
                   rxns_r, ord, temp, rate_eq)
    k3 = rate_calc(i, k, mol_calc(i, pops, vol_loss_rat, t_span, [m / 2 for m in k2], stoich, vol, add_pops, rate_loc), vol,
                   rxns_r, ord, temp, rate_eq)
    k4 = rate_calc(i, k, mol_calc(i, pops, vol_loss_rat, t_span, k3, stoich, vol, add_pops, rate_loc), vol,
                   rxns_r, ord, temp, rate_eq)
    #k2 = rate_calc(i, k, np.add(pops, 0.5 * k1 * t_span), vol, rxns_r, ord, temp, rate_eq)
    #k3 = rate_calc(i, k, np.add(pops, 0.5 * k2 * t_span), vol, rxns_r, ord, temp, rate_eq)
    #k4 = rate_calc(i, k, np.add(pops, k3 * t_span), vol, rxns_r, ord, temp, rate_eq)
    return [(1.0 / 6.0) * k1[m] + 2 * k2[m] + 2 * k3[m] + k4[m] for m in range(len(k1))]


# general kinetic simulator using runge-kutta method
def eq_sim_gen_rk(stoich, mol0, mol_end, add_pops, vol, vol_loss_rat, inc, k, ord, r_locs, p_locs, c_locs,
               var_locs, t_fit, fit_param, fit_param_locs):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    k = fit_param[fit_param_locs[0]]
    var_ord = [fit_param[i] for i in fit_param_locs[1]]
    for i, j in enumerate(var_locs[1]):
        ord[j] = var_ord[i]
    pois = [fit_param[i] for i in fit_param_locs[2]]
    pops = np.zeros((len(t_fit), len(mol0)))
    rate = np.zeros(len(t_fit))
    pops[0] = mol0
    for i in range(len(var_locs[2])):
        pops[:, var_locs[2][i]] -= pois[i]
    i = 0
    for i in range(1, len(t_fit)):
        t_span = t_fit[i] - t_fit[i - 1]
        rate[i - 1] = rk_rate_calc(i - 1, k, pops, vol, ord, t_span)
        pops[i, r_locs] = mol_calc(i, pops, vol_loss_rat, t_span, rate, stoich, vol, add_pops, -1)
        pops[i, p_locs] = mol_calc(i, pops, vol_loss_rat, t_span, rate, stoich, vol, add_pops, 1)
        pops[i, c_locs] = mol_calc(i, pops, vol_loss_rat, t_span, rate, stoich, vol, add_pops, 0)
    rate[i] = rk_rate_calc(i, k, pops, vol, ord, t_span)
    pops[pops < 0] = 0
    pops[:] = [pops[i, :] / vol[i] for i in range(0, len(t_fit))]
    exp_t_rows = list(range(0, len(t_fit), inc - 1))
    pops, rate = pops[exp_t_rows], rate[exp_t_rows]
    return [pops, rate]


# simulate TAKE experiments
def sim_take(t, spec_type=None, spec_name=None, rxns=None, react_vol_init=None, stoich=1, mol0=None, mol_end=None, add_sol_conc=None,
             add_cont_rate=None, t_cont=None, add_one_shot=None, t_one_shot=None, add_col=None, sub_cont_rate=None,
             sub_aliq=None, t_aliq=None, sub_col=None, t_col=None, col=None, temp0=293, temp_cont=None, t_temp=None,
             temp_col=None, rate_eq_type="standard", k_lim=None, ord_lim=None, pois_lim=None, fit_asp=None,
             win=1, inc=1, rand_fac=0.1):
    """
    Params
    ------
    t : numpy.array, list or tuple
        Time values to perform simulation with. Type numpy.array or list will use the exact values.
        Type tuple of the form (start, end, step size) will make time values using these parameters
    spec_type : str or list of str
        Type of each species: "r" for reactant, "p" for product, "c" for catalyst
    react_vol_init : float
        Initial reaction solution volume in volume_unit
    spec_name : str or list of str or None
        Name of each species. Species are given default names if none are provided
    stoich : list of int or None
        Stoichiometry of species, use "None" for catalyst. Default 1
    mol0 : list of float or None
        Initial moles of species in moles_unit or None if data do not need scaling
    mol_end : list of float or None
        Final moles of species in moles_unit or None if data do not need scaling
    add_sol_conc : list of float or None, optional
        Concentration of solution being added for each species in moles_unit volume_unit^-1.
        Default None (no addition solution for all species)
    add_cont_rate : list of float or list of tuple of float or None, optional
        Continuous addition rates of species in moles_unit volume_unit^-1 time_unit^-1.
        Default None (no continuous addition for all species)
    t_cont : list of tuple of float or None, optional
        Times at which continuous addition began for each species in time_unit^-1.
        Default None (no continuous addition for all species)
    add_one_shot : list of tuple of float or None, optional
        One shot additions in volume_unit for each species. Default None (no one shot additions for all species)
    t_one_shot : list of tuple of float or None, optional
        Times at which one shot additions occurred in time_unit^-1 for each species.
        Default None (no additions for all species)
    add_col : list of int or None, optional
        Index of addition column for each species, where addition column is in volume_unit.
        If not None, overwrites add_cont_rate, t_cont, add_one_shot and t_one_shot for each species.
        Default None (no add_col for all species)
    sub_cont_rate : float or None, optional
        Continuous addition rates of species in moles_unit volume_unit^-1 time_unit^-1.
        Default None (no continuous addition for all species)
    sub_aliq : float or list of float or None, optional
        Aliquot subtractions in volume_unit for each species. Default None (no aliquot subtractions)
    t_aliq : float or list of float or None, optional
        Times at which aliquot subtractions occurred in time_unit^-1.
        Default None (no aliquot subtractions)
    sub_col : list of int or None, optional
        Index of subtraction column, where subtraction column is in volume_unit.
        If not None, overwrites sub_cont_rate, sub_aliq and t_aliq.
        Default None (no sub_col)
    t_col : int
        Index of time column. Default 0
    col : list of int
        Index of species column. Default 1
    k_lim : float or tuple of float
        Estimated rate constant in (moles_unit volume_unit)^(sum of orders^-1 + 1) time_unit^-1.
        Can be specified as exact value for fixed variable or variable with bounds (estimate, factor difference) or
        (estimate, lower, upper). Default bounds set as (automated estimate, estimate * 1E-3, estimate * 1E3)
    ord_lim : float or list of tuple of float
        Species reaction order. Can be specified as exact value for fixed variable or
        variable with bounds (estimate, lower, upper) for each species. Default bounds set as (1, 0, 2) for "r" and "c" species and 0 for "p" species
    pois_lim : float, str or tuple of float or str, optional
        Moles of species poisoned in moles_unit. Can be specified as exact value for fixed variable,
        variable with bounds (estimate, lower, upper), or "max" with bounds (0, 0, max species concentration).
        Default assumes no poisoning occurs for all species
    fit_asp : list of str, optional
        Species to fit to: "y" to fit to species, "n" not to fit to species. Default "y"
    TIC_col : int, optional
        Index of TIC column or None if no TIC. Default None
    scale_avg_num : int, optional
        Number of data points from which to calculate mol0 and mol_end. Default 0 (no scaling)
    win : int, optional
        Smoothing window, default 1 if smoothing not required
    inc : int, optional
        Increments between adjacent points for improved simulation, default 1 for using raw time points
    """

    # Prepare parameters
    if type(t) is tuple:
        t = np.linspace(t[0], t[1], int((t[1] - t[0]) / t[2]) + 1)
    spec_name, num_spec, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, \
    add_one_shot, t_one_shot, add_col, sub_aliq, t_aliq, t_col, col, temp_cont, t_temp, rate_loc, rxns_r, k_lim, ord_lim, pois_lim, \
    fit_asp, fix_pois_locs, var_locs, fit_asp_locs, fit_param_locs, \
    inc = take_prep.param_prep(spec_name, spec_type, rxns, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont,
                               add_one_shot, t_one_shot, add_col, sub_aliq, t_aliq, t_col, col, temp_cont, t_temp,
                               rate_eq_type, k_lim, ord_lim, pois_lim, fit_asp, inc)
    rand_fac = take_prep.type_to_list(rand_fac)
    rand_fac = [i if i is not None else 0 for i in rand_fac]
    if len(rand_fac) == 1: rand_fac * num_spec

    # Calculate iterative species additions, volumes, temperature and define rate equation
    add_pops, vol_data, vol_loss_rat = take_prep.get_add_pops_vol(t, np.reshape(t, (len(t), 1)), num_spec, react_vol_init,
                            add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col,
                            sub_cont_rate, sub_aliq, t_aliq, sub_col, win=win)
    temp, _ = take_prep.get_temp(t, np.reshape(t, (len(t), 1)), temp0, temp_cont, t_temp, temp_col, win=win, inc=inc)

    rate_eq = func_map.get(rate_eq_type)

    # Define initial values and lower and upper limits for parameters to fit: ord and pois
    for i in range(num_spec):
        if mol0[i] is None:
            mol0[i] = 0  # May cause issues
        if mol_end[i] is None:
            mol_end[i] = 0  # May cause issues
    for i in fix_pois_locs:
        mol0[i] -= pois_lim[i]
        mol_end[i] -= pois_lim[i]

    # Run simulation
    warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
    fit_pops_all, fit_rate_all = eq_sim_gen(stoich, mol0, mol_end, add_pops, vol_data, vol_loss_rat, temp, rate_eq, inc,
                                            rate_loc, rxns_r, k_lim, ord_lim, [[], [], []], np.diff(t, axis=0), k_lim, [[], [], []])

    # Add noise component
    fit_pops_all += (mol0[0] / react_vol_init) * ((np.random.rand(*fit_pops_all.shape) - 0.5) * 2) * rand_fac

    # Make numpy arrays into DataFrames for improved presentation
    x_data_df = pd.DataFrame(t, columns=["Time / time_unit"])
    y_fit_conc_headers = [i + " fit conc. / moles_unit volume_unit$^{-1}$" for i in spec_name]
    y_fit_rate_headers = ["Reaction " + str(i + 1) + " fit rate / moles_unit volume_unit$^{-1}$ time_unit$^{-1}$" for i in range(len(k_lim))]
    y_fit_conc_df = pd.DataFrame(fit_pops_all, columns=y_fit_conc_headers)
    y_fit_rate_df = pd.DataFrame(fit_rate_all, columns=y_fit_rate_headers)

    temp_df = pd.DataFrame(temp, columns=["Temperature / temp_unit"])

    return x_data_df, y_fit_conc_df, y_fit_rate_df, temp_df, ord_lim


# fit TAKE expeirments
def fit_take(df, spec_name=None, spec_type=None, rxns=None, react_vol_init=None, stoich=1, mol0=None, mol_end=None, add_sol_conc=None,
             add_cont_rate=None, t_cont=None, add_one_shot=None, t_one_shot=None, add_col=None, sub_cont_rate=None,
             sub_aliq=None, t_aliq=None, sub_col=None, t_col=0, col=1, temp0=293, temp_cont=None, t_temp=None,
             temp_col=None, rate_eq_type="standard", k_lim=None, ord_lim=None, pois_lim=None, fit_asp="y", TIC_col=None,
             scale_avg_num=0, win=1, inc=1):

    num_spec, spec_name, stoich, mol0, mol_end, col, rate_loc, rxns_r, k_lim, ord_lim, pois_lim, inc,\
    add_pops, vol, vol_loss_rat, rate_eq,\
    fix_pois_locs, var_locs, fit_asp_locs, fit_param_locs,\
    x_data, x_data_add, x_data_add_to_fit, y_data_to_fit, data_mod, temp, temp_to_fit, col_ext, exp_rates = \
    pre_fit_take(df, spec_name, spec_type, rxns, react_vol_init, stoich, mol0, mol_end,
                 add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col, sub_cont_rate, sub_aliq,
                 t_aliq, sub_col, t_col, col, temp0, temp_cont, t_temp, temp_col, rate_eq_type,
                 k_lim, ord_lim, pois_lim, fit_asp, TIC_col, scale_avg_num, win, inc)

    x_data_df, y_exp_conc_df, y_exp_rate_df, y_fit_conc_df, y_fit_rate_df, \
    k_val, k_fit, k_fit_err, ord_fit, ord_fit_err, \
    pois_fit, pois_fit_err, fit_param_rss, fit_param_r2, temp_df = \
    fitting_func(num_spec, spec_name, stoich, mol0, mol_end, col, rate_eq_type, rate_loc, rxns_r, k_lim, ord_lim, pois_lim, inc,
                 add_pops, vol, vol_loss_rat, rate_eq,
                 fix_pois_locs, var_locs, fit_asp_locs, fit_param_locs,
                 x_data, x_data_add, x_data_add_to_fit, y_data_to_fit, data_mod, temp, temp_to_fit, col_ext, exp_rates)

    fit_summary = take_prep.FitSummary(x_data_df, y_exp_conc_df, y_exp_rate_df, y_fit_conc_df, y_fit_rate_df,
                                       k_val, k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err,
                                       fit_param_rss, fit_param_r2, temp_df, col, ord_lim)

    return x_data_df, y_exp_conc_df, y_exp_rate_df, y_fit_conc_df, y_fit_rate_df, k_val, k_fit, k_fit_err, \
           ord_fit, ord_fit_err, pois_fit, pois_fit_err, fit_param_rss, fit_param_r2, temp_df, col, ord_lim


def pre_fit_take(df, spec_name=None, spec_type=None, rxns=None, react_vol_init=None, stoich=1, mol0=None, mol_end=None, add_sol_conc=None,
             add_cont_rate=None, t_cont=None, add_one_shot=None, t_one_shot=None, add_col=None, sub_cont_rate=None,
             sub_aliq=None, t_aliq=None, sub_col=None, t_col=0, col=1, temp0=293, temp_cont=None, t_temp=None,
             temp_col=None, rate_eq_type="standard", k_lim=None, ord_lim=None, pois_lim=None, fit_asp="y", TIC_col=None,
             scale_avg_num=0, win=1, inc=1):
    """
    Params
    ------
    df : pandas.DataFrame
        The reaction data
    spec_type : str or list of str
        Type of each species: "r" for reactant, "p" for product, "c" for catalyst
    react_vol_init : float
        Initial reaction solution volume in volume_unit
    spec_name : str or list of str or None
        Name of each species. Species are given default names if none are provided
    stoich : list of int or None
        Stoichiometry of species, use "None" for catalyst. Default 1
    mol0 : list of float or None
        Initial moles of species in moles_unit or None if data do not need scaling
    mol_end : list of float or None
        Final moles of species in moles_unit or None if data do not need scaling
    add_sol_conc : list of float or None, optional
        Concentration of solution being added for each species in moles_unit volume_unit^-1.
        Default None (no addition solution for all species)
    add_cont_rate : list of float or list of tuple of float or None, optional
        Continuous addition rates of species in moles_unit volume_unit^-1 time_unit^-1.
        Default None (no continuous addition for all species)
    t_cont : list of tuple of float or None, optional
        Times at which continuous addition began for each species in time_unit^-1.
        Default None (no continuous addition for all species)
    add_one_shot : list of tuple of float or None, optional
        One shot additions in volume_unit for each species. Default None (no one shot additions for all species)
    t_one_shot : list of tuple of float or None, optional
        Times at which one shot additions occurred in time_unit^-1 for each species.
        Default None (no additions for all species)
    add_col : list of int or None, optional
        Index of addition column for each species, where addition column is in volume_unit.
        If not None, overwrites add_cont_rate, t_cont, add_one_shot and t_one_shot for each species.
        Default None (no add_col for all species)
    sub_cont_rate : float or None, optional
        Continuous addition rates of species in moles_unit volume_unit^-1 time_unit^-1.
        Default None (no continuous addition for all species)
    sub_aliq : float or list of float or None, optional
        Aliquot subtractions in volume_unit for each species. Default None (no aliquot subtractions)
    t_aliq : float or list of float or None, optional
        Times at which aliquot subtractions occurred in time_unit^-1.
        Default None (no aliquot subtractions)
    sub_col : list of int or None, optional
        Index of subtraction column, where subtraction column is in volume_unit.
        If not None, overwrites sub_cont_rate, sub_aliq and t_aliq.
        Default None (no sub_col)
    t_col : int
        Index of time column. Default 0
    col : list of int
        Index of species column. Default 1
    k_lim : float or tuple of float
        Estimated rate constant in (moles_unit volume_unit)^(sum of orders^-1 + 1) time_unit^-1.
        Can be specified as exact value for fixed variable or variable with bounds (estimate, factor difference) or
        (estimate, lower, upper). Default bounds set as (automated estimate, estimate * 1E-3, estimate * 1E3)
    ord_lim : float or list of tuple of float
        Species reaction order. Can be specified as exact value for fixed variable or
        variable with bounds (estimate, lower, upper) for each species.
        Default bounds set as (1, 0, 2) for "r" and "c" species and 0 for "p" species
    pois_lim : float, str or tuple of float or str, optional
        Moles of species poisoned in moles_unit. Can be specified as exact value for fixed variable,
        variable with bounds (estimate, lower, upper), or "max" with bounds (0, 0, max species concentration).
        Default assumes no poisoning occurs for all species
    fit_asp : list of str, optional
        Species to fit to: "y" to fit to species, "n" not to fit to species. Default "y"
    TIC_col : int, optional
        Index of tic column or None if no tic. Default None
    scale_avg_num : int, optional
        Number of data points from which to calculate mol0 and mol_end. Default 0 (no scaling)
    win : int, optional
        Smoothing window, default 1 if smoothing not required
    inc : int, optional
        Increments between adjacent points for improved simulation, default 1 for using raw time points
    """

    spec_name, num_spec, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, \
    add_one_shot, t_one_shot, add_col, sub_aliq, t_aliq, t_col, col, temp_cont, t_temp, rate_loc, rxns_r, k_lim, ord_lim, pois_lim, \
    fit_asp, fix_pois_locs, var_locs, fit_asp_locs, fit_param_locs, \
    inc = take_prep.param_prep(spec_name, spec_type, rxns, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont,
                    add_one_shot, t_one_shot, add_col, sub_aliq, t_aliq, t_col, col, temp_cont, t_temp,
                    rate_eq_type, k_lim, ord_lim, pois_lim, fit_asp, inc)

    # Get x_data
    data_org = df.to_numpy()
    x_data = take_prep.data_smooth(data_org, t_col, win)
    x_data_add = take_prep.add_sim(np.reshape(x_data, (len(x_data))), inc)

    # Get tic
    tic = take_prep.data_smooth(data_org, TIC_col, win) if TIC_col is not None else None

    # Calculate iterative species additions, volumes, temperatures and the appropriate rate function
    add_pops, vol, vol_loss_rat = take_prep.get_add_pops_vol(data_org, data_org[:, t_col], num_spec,
                                            react_vol_init, add_sol_conc, add_cont_rate, t_cont, add_one_shot, 
                                            t_one_shot, add_col, sub_cont_rate, sub_aliq, t_aliq, sub_col,
                                            win=win, inc=inc)
    temp, temp_to_fit = take_prep.get_temp(data_org, data_org[:, t_col], temp0, temp_cont, t_temp, temp_col,
                              win=win, inc=inc)
    rate_eq = func_map.get(rate_eq_type)

    # Determine mol0, mol_end and scale data as required
    data_mod = np.empty((len(x_data), num_spec))
    col_ext = []
    for i in range(num_spec):
        if col[i] is not None:
            col_ext = [*col_ext, i]
            data_i = take_prep.data_smooth(data_org, col[i], win)
            data_i = take_prep.tic_norm(data_i, tic)
            if mol0[i] is None and scale_avg_num == 0:
                mol0[i] = data_i[0] * vol[0]
            elif mol0[i] is None and scale_avg_num > 0:
                mol0[i] = np.mean([data_i[j] * vol[j] for j in range(scale_avg_num)])
            elif mol0[i] is not None and mol0[i] != 0 and scale_avg_num > 0 and (
                        mol_end[i] is None or mol0[i] >= mol_end[i]):
                data_scale = np.mean([data_i[j] / (mol0[i] / vol[j]) for j in range(scale_avg_num)])
                data_i = data_i / data_scale
            if mol_end[i] is None and scale_avg_num == 0:
                mol_end[i] = data_i[-1] * vol[-1]
            elif mol_end[i] is None and scale_avg_num > 0:
                mol_end[i] = np.mean([data_i[j] * vol[j] for j in range(-scale_avg_num, 0)])
            elif mol_end[i] is not None and mol_end[i] != 0 and scale_avg_num > 0 and (
                        mol0[i] is None or mol_end[i] >= mol0[i]):
                data_scale = np.mean([data_i[j] / (mol_end[i] / vol[j]) for j in range(-scale_avg_num, 0)])
                data_i = data_i / data_scale
            data_mod[:, i] = data_i
        if col[i] is None and mol0[i] is None:
            mol0[i] = 0  # May cause issues
        if col[i] is None and mol_end[i] is None:
            mol_end[i] = 0  # May cause issues

    warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
    exp_t_rows = list(range(0, len(x_data_add), inc - 1))
    exp_rates = np.zeros((len(exp_t_rows), num_spec))
    for i in range(num_spec):
        if col[i] is not None:
            exp_rates[:, i] = np.gradient(data_mod[:, i], x_data_add[exp_t_rows])

    # Manipulate data for fitting
    x_data_add_to_fit = np.empty(0)
    y_data_to_fit = np.empty(0)
    for i in range(len(fit_asp_locs)):
        x_data_add_to_fit = np.append(x_data_add_to_fit, x_data_add, axis=0)
        y_data_to_fit = np.append(y_data_to_fit, data_mod[:, fit_asp_locs[i]], axis=0)
    return num_spec, spec_name, stoich, mol0, mol_end, col, rate_loc, rxns_r, k_lim, ord_lim, pois_lim, inc,\
           add_pops, vol, vol_loss_rat, rate_eq,\
           fix_pois_locs, var_locs, fit_asp_locs, fit_param_locs,\
           x_data, x_data_add, x_data_add_to_fit, y_data_to_fit, data_mod, temp, temp_to_fit, col_ext, exp_rates


def fitting_func(num_spec, spec_name, stoich, mol0, mol_end, col, rate_eq_type, rate_loc, rxns_r, k_lim, ord_lim, pois_lim, inc,
                 add_pops, vol, vol_loss_rat, rate_eq,
                 fix_pois_locs, var_locs, fit_asp_locs, fit_param_locs,
                 x_data, x_data_add, x_data_add_to_fit, y_data_to_fit, data_mod, temp, temp_to_fit, col_ext, exp_rates):

    x_data_add_to_fit_span = np.diff(x_data_add_to_fit[:int(len(x_data_add_to_fit) / len(fit_asp_locs))], axis=0)
    # Function for sorting data for fitting
    def eq_sim_fit(_, *fit_param):
        pops, rate = eq_sim_gen(stoich, mol0, mol_end, add_pops, vol, vol_loss_rat, temp_to_fit, rate_eq, inc, rate_loc,
                                rxns_r, k_lim, ord_lim, var_locs, x_data_add_to_fit_span, fit_param, fit_param_locs)
        pops_reshape = np.empty(0)
        for i in fit_asp_locs:
            pops_reshape = np.append(pops_reshape, pops[:, i], axis=0)
        return pops_reshape

    # Function for guessing initial constant values, based on given allowed parameter combinations and spacing type
    def guess_sim(x_data, y_data, ord_val, pois_val, param_comb):
        guess_list = []
        for i in range(len(param_comb)):
            if len(param_comb[i]) < 4:
                guess_list.append([param_comb[i][0]])
            elif 'a' in param_comb[i][3]:
                guess_list.append(np.linspace(*param_comb[i][:-1], param_comb[i][2] + 1).tolist())
            elif 'g' in param_comb[i][3]:
                guess_list.append([param_comb[i][2] ** j for j in range(param_comb[i][0], param_comb[i][1])])
        pre_k_guess_arr = np.array(list(itertools.product(*guess_list)))
        k_guess_arr = np.column_stack((pre_k_guess_arr, np.zeros(len(pre_k_guess_arr))))
        for i in range(len(k_guess_arr)):
            fit_guess = eq_sim_fit(x_data, *k_guess_arr[i, :-1], *ord_val, *pois_val)
            k_guess_arr[i, -1], _ = take_prep.residuals(y_data, fit_guess)
        sort_indices = np.argsort(k_guess_arr[:, -1])
        k_guess_sort = k_guess_arr[sort_indices]
        # take_plot.plot_other_fits_3D(k_guess_arr)
        return k_guess_sort[0, :-1].tolist()

    # Define initial values and lower and upper limits for parameters to fit: ord and pois
    ord_val, ord_min, ord_max, pois_val, pois_min, pois_max = [], [], [], [], [], []
    for i, j in var_locs[1]:
        ord_val.append(ord_lim[i][j][0])
        ord_min.append(ord_lim[i][j][1])
        ord_max.append(ord_lim[i][j][2])
    for i in range(len(var_locs[2])):
        unpack_pois_lim = pois_lim[var_locs[2][i]]
        if "max" in unpack_pois_lim:
            pois_val.append(0)
            pois_min.append(0)
            pois_max.append(max(mol0[i], mol_end[i]))
        else:
            pois_val.append(unpack_pois_lim[0])
            pois_min.append(unpack_pois_lim[1])
            pois_max.append(unpack_pois_lim[2])
    for i in fix_pois_locs:
        mol0[i] -= pois_lim[i]
        mol_end[i] -= pois_lim[i]

    # Define initial constant (k) values to fit
    # starttime = timeit.default_timer()
    k_val, k_min, k_max, param_comb = [], [], [], []
    for i, j in var_locs[0]:
        if isinstance(k_lim[i][j], (tuple, list)) and k_lim[i][j][0] is None:
            param_comb.append(take_prep.param_comb_dict[rate_eq_type][j])  # swapped from i to j - may need repair
        else:
            param_comb.append(k_lim[i][j])
    k_val = guess_sim(x_data_add_to_fit, y_data_to_fit, ord_val, pois_val, param_comb)

    # Define lower and upper limits for constant(s) (k) values to fit
    bound_adj = 1E-6
    for i, (j, m) in enumerate(var_locs[0]):
        if isinstance(k_lim[j][m], (tuple, list)) and (len(k_lim[j][m]) > 1 and k_lim[j][m][1] is None) or \
                (len(k_lim[j][m]) > 2 and k_lim[j][m][2] is None):
            if ("standard" or "MM") in rate_eq_type or ("Arrhenius" in rate_eq_type and m == 0):
                k_min.append(k_val[i] * bound_adj)
                k_max.append(k_val[i] / bound_adj)
            if "Eyring" in rate_eq_type or ("Arrhenius" in rate_eq_type and m == 1):
                k_min.append(take_prep.param_comb_dict[rate_eq_type][m][0])
                k_max.append(take_prep.param_comb_dict[rate_eq_type][m][1])
        elif len(k_lim[j][m]) == 2:
            k_min.append(k_val[i] * k_lim[j][m][1])
            k_max.append(k_val[i] / k_lim[j][m][1])
        elif len(k_lim[j][m]) == 3:
            k_min.append(k_lim[j][m][1])
            k_max.append(k_lim[j][m][2])
        # print(timeit.default_timer() - starttime)

    init_param = [*k_val, *ord_val, *pois_val]
    low_bounds = [*k_min, *ord_min, *pois_min]
    up_bounds = [*k_max, *ord_max, *pois_max]

    # Path if no parameters were set to fit that avoids applying fitting
    if not init_param or not low_bounds or not up_bounds:
        print("No parameters set to fit - no fitting applied.")
        x_data = pd.DataFrame(x_data, columns=["Time / time_unit"])
        y_exp_conc_headers = [spec_name[i] + " exp. conc. / moles_unit volume_unit$^{-1}$"
                              for i in range(num_spec) if col[i] is not None]
        y_exp_rate_headers = [spec_name[i] + " exp. rate / moles_unit volume_unit$^{-1}$ time_unit$^{-1}$"
                              for i in range(num_spec) if col[i] is not None]
        y_exp_conc = pd.DataFrame(data_mod[:, col_ext], columns=y_exp_conc_headers)
        y_exp_rate = pd.DataFrame(exp_rates[:, col_ext], columns=y_exp_rate_headers)
        temp_df = pd.DataFrame(temp, columns=["Temperature / temp_unit"])
        return x_data, y_exp_conc, y_exp_rate, None, None, [], [], [], [], [], [], [], [], [], temp_df

    # Apply fittings, determine optimal parameters and determine resulting fits
    fit_popt, fit_param_pcov = optimize.curve_fit(eq_sim_fit, x_data_add_to_fit, y_data_to_fit, init_param, maxfev=10000,
                                       bounds=(low_bounds, up_bounds), method='trf')

    k_fit = fit_popt[fit_param_locs[0]]
    ord_fit = fit_popt[fit_param_locs[1]]
    pois_fit = fit_popt[fit_param_locs[2]]

    fit_pops_set = eq_sim_fit(x_data_add_to_fit, *k_fit, *ord_fit, *pois_fit)
    fit_pops_all, fit_rate_all = eq_sim_gen(stoich, mol0, mol_end, add_pops, vol, vol_loss_rat, temp_to_fit, rate_eq,
                                            inc, rate_loc, rxns_r, k_lim, ord_lim, var_locs,
                                            np.diff(x_data_add, axis=0), fit_popt, fit_param_locs)

    # Calculate residuals and errors
    fit_param_err = np.sqrt(np.diag(fit_param_pcov))
    k_fit_err = fit_param_err[fit_param_locs[0]]
    ord_fit_err = fit_param_err[fit_param_locs[1]]
    pois_fit_err = fit_param_err[fit_param_locs[2]]
    fit_param_rss, fit_param_r2 = take_prep.residuals(y_data_to_fit, fit_pops_set)
    fit_param_aic = len(y_data_to_fit) * math.log(fit_param_rss / len(y_data_to_fit)) + 2 * len(init_param)

    # Prepare data for output
    x_data_df = pd.DataFrame(x_data, columns=["Time / time_unit"])
    y_exp_conc_headers = [spec_name[i] + " exp. conc. / moles_unit volume_unit$^{-1}$"
                          for i in range(num_spec) if col[i] is not None]
    y_exp_rate_headers = [spec_name[i] + " exp. rate / moles_unit volume_unit$^{-1}$ time_unit$^{-1}$"
                          for i in range(num_spec) if col[i] is not None]
    y_fit_conc_headers = [i + " fit conc. / moles_unit volume_unit$^{-1}$" for i in spec_name]
    y_fit_rate_headers = ["Reaction " + str(i + 1) + " fit rate / moles_unit volume_unit$^{-1}$ time_unit$^{-1}$" for i in range(len(k_lim))]
    y_exp_conc_df = pd.DataFrame(data_mod[:, col_ext], columns=y_exp_conc_headers)
    y_exp_rate_df = pd.DataFrame(exp_rates[:, col_ext], columns=y_exp_rate_headers)
    y_fit_conc_df = pd.DataFrame(fit_pops_all, columns=y_fit_conc_headers)
    y_fit_rate_df = pd.DataFrame(fit_rate_all, columns=y_fit_rate_headers)
    temp_df = pd.DataFrame(temp, columns=["Temperature / temp_unit"])

    if not var_locs[0]:
        k_val, k_fit, k_fit_err = "N/A", "N/A", "N/A"
    if not var_locs[1]:
        ord_fit, ord_fit_err = "N/A", "N/A"
    if not var_locs[2]:
        pois_fit, pois_fit_err = "N/A", "N/A"
        t_del_fit, t_del_fit_err = "N/A", "N/A"
    else:
        pois_fit, pois_fit_err = pois_fit / vol[0], pois_fit_err / vol[0]
        t_del_fit, t_del_fit_err = pois_fit * 1, pois_fit_err * 1  # need to make t_del work somehow

    return x_data_df, y_exp_conc_df, y_exp_rate_df, y_fit_conc_df, y_fit_rate_df, k_val, k_fit, k_fit_err, \
           ord_fit, ord_fit_err, pois_fit, pois_fit_err, fit_param_rss, fit_param_r2, temp_df


def fit_err_real(df, spec_name=None, spec_type=None, rxns=None, react_vol_init=None, stoich=1, mol0=None, mol_end=None, add_sol_conc=None,
             add_cont_rate=None, t_cont=None, add_one_shot=None, t_one_shot=None, add_col=None, sub_cont_rate=None,
             sub_aliq=None, t_aliq=None, sub_col=None, t_col=0, col=1, temp0=293, temp_cont=None, t_temp=None,
             temp_col=None, rate_eq_type="standard", k_lim=None, ord_lim=None, pois_lim=None, fit_asp="y", TIC_col=None,
             scale_avg_num=0, win=1, inc=1):

    num_spec, spec_name, stoich, mol0, mol_end, col, rate_loc, rxns_r, k_lim, ord_lim, pois_lim, inc,\
    add_pops, vol, vol_loss_rat, rate_eq,\
    fix_pois_locs, var_locs, fit_asp_locs, fit_param_locs,\
    x_data, x_data_add, x_data_add_to_fit, y_data_to_fit, data_mod, temp, temp_to_fit, col_ext, exp_rates = \
    pre_fit_take(df, spec_name, spec_type, rxns, react_vol_init, stoich, mol0, mol_end,
                 add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col, sub_cont_rate, sub_aliq,
                 t_aliq, sub_col, t_col, col, temp0, temp_cont, t_temp, temp_col, rate_eq_type,
                 k_lim, ord_lim, pois_lim, fit_asp, TIC_col, scale_avg_num, win, inc)

    x_data_df, y_exp_conc_df, y_exp_rate_df, y_fit_conc_df, y_fit_rate_df, \
    k_val, k_fit, k_fit_err, ord_fit, ord_fit_err, \
    pois_fit, pois_fit_err, fit_param_rss, fit_param_r2, temp_df = \
    fitting_func(num_spec, spec_name, stoich, mol0, mol_end, col, rate_eq_type, rate_loc, rxns_r, k_lim, ord_lim, pois_lim, inc,
                 add_pops, vol, vol_loss_rat, rate_eq,
                 fix_pois_locs, var_locs, fit_asp_locs, fit_param_locs,
                 x_data, x_data_add, x_data_add_to_fit, y_data_to_fit, data_mod, temp, temp_to_fit, col_ext, exp_rates)

    bound_adj = 1E-3
    k_lim = [[(*k_val, bound_adj)]]

    real_err_inc = 10  # enter number of increments between min and max orders
    real_err_inc += 1
    test_ord_var_pre = [np.round(np.linspace(ord_lim[0][i][1], ord_lim[0][i][2], real_err_inc), 1).tolist() for i in range(len(ord_lim[0]))
                        if isinstance(ord_lim[0][i], tuple)]
    test_ord_var = list(itertools.product(*test_ord_var_pre))
    test_ord_all_pre = [np.linspace(ord_lim[0][i][1], ord_lim[0][i][2], real_err_inc).tolist()
                        if isinstance(ord_lim[0][i], tuple) else [ord_lim[0][i]] for i in range(len(ord_lim[0]))]
    test_ord_all = list(itertools.product(*test_ord_all_pre))

    real_err_fit = np.empty([1, 4], dtype=object)
    real_err_fit_y_fit_conc, real_err_fit_y_fit_rate = np.empty([1, 1], dtype=object), np.empty([1, 1], dtype=object)
    real_err_fit[0] = [ord_fit, k_fit, pois_fit, fit_param_r2]
    real_err_fit_y_fit_conc[0] = [y_fit_conc_df]
    real_err_fit_y_fit_rate[0] = [y_fit_rate_df]

    fix_pois_locs, var_locs, fit_asp_locs, fit_param_locs = \
        take_prep.get_var_locs(spec_type, num_spec, k_lim, [list(test_ord_all[0])], pois_lim, fit_asp)

    real_err = np.empty([len(test_ord_all), 4], dtype=object)
    real_err_y_fit_conc = np.empty([len(test_ord_all), 1], dtype=object)
    real_err_y_fit_rate = np.empty([len(test_ord_all), 1], dtype=object)
    for i in range(len(test_ord_all)):
        x_data_df, _, _, y_fit_conc_df, y_fit_rate_df, k_val, k_fit, k_fit_err, ord_fit, ord_fit_err, \
        pois_fit, pois_fit_err, fit_param_rss, fit_param_r2, temp_df = \
            fitting_func(num_spec, spec_name, stoich, mol0, mol_end, col, rate_eq_type, rate_loc, rxns_r,
                         k_lim, [list(test_ord_all[i])], pois_lim, inc, add_pops, vol, vol_loss_rat, rate_eq,
                         fix_pois_locs, var_locs, fit_asp_locs, fit_param_locs, x_data, x_data_add,
                         x_data_add_to_fit, y_data_to_fit, data_mod, temp, temp_to_fit, col_ext, exp_rates)
        real_err[i] = [[*test_ord_var[i]], k_fit, pois_fit, fit_param_r2]
        real_err_y_fit_conc[i] = [y_fit_conc_df]
        real_err_y_fit_rate[i] = [y_fit_rate_df]

    real_err_sort = np.concatenate((real_err_fit, real_err[real_err[:, -1].argsort()[::-1]]))
    real_err_y_fit_conc_sort = np.concatenate((real_err_fit_y_fit_conc, real_err_y_fit_conc[real_err[:, -1].argsort()[::-1]]))
    real_err_y_fit_rate_sort = np.concatenate((real_err_fit_y_fit_rate, real_err_y_fit_rate[real_err[:, -1].argsort()[::-1]]))
    headers = ["Order", "k", "Poisoning", "r^2"]
    real_err_sort_df = pd.DataFrame(real_err_sort, columns=headers)
    print(real_err_sort_df)
    return x_data_df, y_exp_conc_df, y_exp_rate_df, real_err_y_fit_conc_sort, real_err_y_fit_rate_sort, real_err_sort_df, col, ord_lim


if __name__ == "__main__":
    spec_name = ["r1", "res_r2", "p1", "c1"]
    spec_type = ["r", "r", "p", "c"]
    react_vol_init = 0.1
    stoich = [1, 1, 1, None]  # insert stoichiometry of reactant, r
    mol0 = [0.1, 0.2, 0, 0]
    mol_end = [0, 0.1, 0.1, None]
    add_sol_conc = [None, None, None, 10]
    add_cont_rate = [None, None, None, 0.001]
    t_cont = [None, None, None, 1]
    t_col = 0
    col = [1, 2, 3, None]
    ord_lim = [(1, 0, 2), 1, 0, (1, 0, 2)]
    fit_asp = ["y", "n", "y", "n"]
    file_name = r'C:\Users\Peter\Documents\Postdoctorate\Work\CAKE\Case studies\CAKE preliminary trials.xlsx'
    sheet_name = r'Test_data'
    pic_save = r'/Users/bhenders/Desktop/CAKE/take_app_test.png'
    xlsx_save = r'/Users/bhenders/Desktop/CAKE/fit_data.xlsx'

    df = take_prep.read_data(file_name, sheet_name, t_col, col)
    output = fit_take(df, spec_name=spec_name, spec_type=spec_type, react_vol_init=react_vol_init, stoich=stoich, mol0=mol0, mol_end=mol_end,
                      add_sol_conc=add_sol_conc, add_cont_rate=add_cont_rate, t_cont=t_cont,
                      t_col=t_col, col=col, ord_lim=ord_lim, fit_asp=fit_asp)
    x_data_df, y_exp_df, y_fit_conc_df, y_fit_rate_df, k_val_est, k_fit, k_fit_err, \
    ord_fit, ord_fit_err, pois_fit, pois_fit_err, ss_res, r2, col = output

    if not isinstance(col, (tuple, list)): col = [col]

    html = take_plot.plot_fit_results(x_data_df, y_exp_df, y_fit_conc_df, col,
                            f_format='svg', return_image=False, save_disk=True, save_to=pic_save)

    param_dict = take_app.make_param_dict(spec_type, react_vol_init, stoich=stoich, mol0=mol0, mol_end=mol_end,
                                 add_sol_conc=add_sol_conc, add_cont_rate=add_cont_rate, t_cont=t_cont,
                                 t_col=t_col, col=col, ord_lim=None, fit_asp=fit_asp)

    # write_fit_data(xlsx_save, df, param_dict, t, r, p, fit_p, fit_r, res_val, res_err, res_rss, res_r2, cat_pois)
    file, _ = take_app.write_fit_data_temp(df, param_dict, x_data_df, y_exp_df, y_fit_conc_df, y_fit_rate_df,
                                  k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err, ss_res, r2)
    file.seek(0)
    with open(xlsx_save, "wb") as f:
        f.write(file.getbuffer())

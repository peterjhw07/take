"""TAKE Miscellaneous Functions"""

# Imports
import numpy as np
import pandas as pd
import io
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


# define additional t values for data sets with few data points
def add_sim(s, inc):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    s_fit = np.zeros((((len(s) - 1) * (inc - 1)) + 1))
    for i in range(len(s) - 1):
        new_s_i = np.linspace(s[i], s[i + 1], num=inc)[:-1]
        s_fit[i * len(new_s_i):(i * len(new_s_i)) + len(new_s_i)] = new_s_i
    s_fit[-1] = s[-1]
    return s_fit


# smooth data (if required)
def data_smooth(arr, d_col, win=1):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    if win <= 1:
        d_ra = arr[:, d_col]
    else:
        ret = np.cumsum(arr[:, d_col], dtype=float)
        ret[win:] = ret[win:] - ret[:-win]
        d_ra = ret[win - 1:] / win
    return d_ra


# manipulate to TIC values (for MS only)
def tic_norm(data, tic=None):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    if tic is not None:
        data = data / tic
    else:
        data = data
    return data


# calculate residuals
def residuals(y_data, fit):
    """
    Function Description

    Params
    ------

    Returns
    -------


    """
    rss = np.sum((y_data - fit) ** 2)
    r2 = 1 - (rss / np.sum((y_data - np.mean(y_data)) ** 2))
    return [rss, r2]


# find nearest value
def find_nearest(array, value):
    """
        Find nearest element to value in array

        Params
        ------

        Returns
        -------


    """
    array = np.asarray(array)
    index = (np.abs(array - value)).argmin()
    return index


# return all None to ensure correct list lengths for iterative purposes
def return_all_nones(s, num_spec):
    if s is None: s = [None] * num_spec
    return s


def rearrange_list(list, indices, replacement):
    return [list[i] if i is not None else replacement for i in indices]


def insert_empty(s, ex_len, replacement):
    return s.append([replacement] * ex_len)


# convert non-list to list
def type_to_list(s):
    if not isinstance(s, list):
        s = [s]
    return s


# convert int and float into lists inside tuples
def tuple_of_lists_from_tuple_of_int_float(s):
    s_list = []
    for i in range(len(s)):
        if isinstance(s[i], (int, float)):
            s_list = [*s_list, [s[i]]]
        else:
            s_list = [*s_list, s[i]]
    return s_list


def replace_none_with_nones(item):
    if isinstance(item, list):
        return [replace_none_with_nones(sub_item) if sub_item is not None else (None, None, None) for sub_item in item]
    else:
        return item if item is not None else (None, None, None)


# read imported data
def read_data(file_name, sheet_name, t_col, col, add_col, sub_col, temp_col):
    """
    Read in data from excel filename

    Params
    ------

    Returns
    -------


    """
    df = pd.read_excel(file_name, sheet_name=sheet_name, engine='openpyxl', dtype=str)
    headers = list(pd.read_excel(file_name, sheet_name=sheet_name, engine='openpyxl').columns)
    if isinstance(col, int): col = [col]
    if add_col is None or isinstance(add_col, int): add_col = [add_col]
    conv_col = [i for i in [t_col, *col, *add_col, sub_col, temp_col] if i is not None]
    try:
        for i in conv_col:
            df[headers[i]] = pd.to_numeric(df[headers[i]], downcast="float")
        return df
    except ValueError:
        raise ValueError("Excel file must contain data rows (i.e. col specified) of numerical input with at most 1 header row.")


# prepare parameters
def param_prep(spec_name, spec_type, rxns, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot,
               t_one_shot, add_col, sub_aliq, t_aliq, t_col, col, temp_cont, t_temp, rate_eq_type,
               k_lim, ord_lim, pois_lim, fit_asp, inc):

    if rxns:
        spec, stoich, rate_loc, rxns_r, ord_lim = get_multi_rate_eq(rxns)
        num_spec = len(spec)

        spec, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col, \
        fit_asp = map(return_all_nones, [spec, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont,
                                         add_one_shot, t_one_shot, add_col, fit_asp], [num_spec] * 10)

        if spec_name is None:
            spec_name = spec
        else:
            spec_locs = [spec_name.index(i) if i in spec_name else None for i in spec]
            spec_name = spec
            mol0 = rearrange_list(mol0, spec_locs, 0)
            mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col \
                = map(lambda s: rearrange_list(s, spec_locs, None), [mol_end, add_sol_conc, add_cont_rate,
                                                                  t_cont, add_one_shot, t_one_shot, add_col])
            fit_asp = rearrange_list(fit_asp, spec_locs, None)
            if col is not None:
                col = rearrange_list(col, spec_locs, None)
            if pois_lim is None:
                pois_lim = [0] * num_spec
            else:
                pois_lim = rearrange_list(pois_lim, spec_locs, 0)

    elif spec_type:
        spec_type = type_to_list(spec_type)
        if spec_name is None:
            spec_name = spec_type
        num_spec = len(spec_name)

        if stoich is None:
            stoich = []
            for i in spec_type:
                if 'r' in i in i: stoich.append([-1])
                elif 'c' in i: stoich.append([0])
                elif 'p' in i: stoich.append([1])
        else:
            if isinstance(stoich, (int, float)): stoich = [stoich]
            for i, j in enumerate(spec_type):
                if stoich[i] is None:
                    if 'r' in j: stoich[i] = [-1]
                    elif 'c' in j: stoich[i] = [0]
                    elif 'p' in j: stoich[i] = [1]
                else:
                    if 'r' in j: stoich[i] = [abs(stoich[i]) * -1]
                    elif 'c' in j: stoich[i] = [stoich[i] * 0]
                    elif 'p' in j: stoich[i] = [stoich[i]]

        spec_name, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col, \
        fit_asp = map(return_all_nones, [spec_name, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont,
                                     add_one_shot, t_one_shot, add_col, fit_asp], [num_spec] * 10)
        for i in range(num_spec):
            if spec_name[i] is None:
                spec_name[i] = "Species " + str(i + 1)

        if ord_lim is None:
            ord_lim = []
            for i in spec_type:
                if 'r' in i or 'c' in i: ord_lim.append((1, 0, 2))
                elif 'p' in i: ord_lim.append(0)
        elif isinstance(ord_lim, (int, float)):
            ord_lim = [ord_lim]
        else:
            for i, j in enumerate(spec_type):
                if 'p' in j and ord_lim[i] is None: ord_lim[i] = 0
        ord_lim = [ord_lim]
        rate_loc = [[0]] * num_spec
        rxns_r = [tuple(range(num_spec))]

        if pois_lim is None: pois_lim = [0] * num_spec

    if k_lim is None:
        if "standard" in rate_eq_type: k_lim = [[(None, None, None)]] * len(rxns_r)
        elif "MM" or "Arrhenius" or "Eyring" in rate_eq_type: k_lim = [[(None, None, None), (None, None, None)]] * len(rxns_r)
    elif isinstance(k_lim, (int, float)): k_lim = [k_lim]
    else:
        k_lim = replace_none_with_nones(k_lim)
    if not (isinstance(k_lim, list) and all(isinstance(sublist, list) for sublist in k_lim)):
        k_lim = [k_lim]

    spec_name, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col, \
    sub_aliq, t_aliq, t_col, col, temp_cont, t_temp, ord_lim, pois_lim, fit_asp = map(type_to_list, [spec_name,
    stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col,
    sub_aliq, t_aliq, t_col, col, temp_cont, t_temp, ord_lim, pois_lim, fit_asp])
    add_cont_rate, t_cont, add_one_shot, t_one_shot = map(tuple_of_lists_from_tuple_of_int_float,
                                            [add_cont_rate, t_cont, add_one_shot, t_one_shot])

    fix_pois_locs, var_locs, fit_asp_locs, fit_param_locs = get_var_locs(spec_type, num_spec, k_lim, ord_lim, pois_lim, fit_asp)
    inc += 1

    return spec_name, num_spec, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, \
           add_one_shot, t_one_shot, add_col, sub_aliq, t_aliq, t_col, col, temp_cont, t_temp, rate_loc, rxns_r, k_lim, ord_lim, \
           pois_lim, fit_asp, fix_pois_locs, var_locs, fit_asp_locs, fit_param_locs, inc


def get_var_locs(spec_type, num_spec, k_lim, ord_lim, pois_lim, fit_asp):
    # var_k_locs = [i for i in range(len(k_lim)) if (isinstance(k_lim[i], (tuple, list)) and len(k_lim[i]) > 1)]
    var_k_locs = [(i, j) for i in range(len(k_lim)) for j in range(len(k_lim[i])) if (isinstance(k_lim[i][j], (tuple, list)) and len(k_lim[i][j]) > 1)]
    if spec_type:
        # var_ord_locs = [i for i in range(num_spec) if (isinstance(ord_lim[i], (tuple, list)) and len(ord_lim[i]) > 1)]
        var_ord_locs = [(i, j) for i in range(len(ord_lim)) for j in range(len(ord_lim[i])) if (isinstance(ord_lim[i][j], (tuple, list)) and len(ord_lim[i][j]) > 1)]
    else:
        var_ord_locs = []
    fix_pois_locs = [i for i in range(num_spec) if (isinstance(pois_lim[i], (int, float))
                    or (isinstance(pois_lim[i], (tuple, list)) and len(pois_lim[i]) == 1))]
    var_pois_locs = [i for i in range(num_spec) if (isinstance(pois_lim[i], (tuple, list, str))
                                                    and len(pois_lim[i]) > 1)]
    var_locs = [var_k_locs, var_ord_locs, var_pois_locs]
    fit_asp_locs = [i for i in range(num_spec) if fit_asp[i] is not None and 'y' in fit_asp[i]]
    fit_param_locs = [range(0, len(var_k_locs)), range(len(var_k_locs), len(var_k_locs) + len(var_ord_locs)),
                      range(len(var_k_locs) + len(var_ord_locs),
                            len(var_k_locs) + len(var_ord_locs) + len(var_pois_locs))]
    return fix_pois_locs, var_locs, fit_asp_locs, fit_param_locs


# calculate additions and subtractions of species
def get_add_pops_vol(data_org, x_data_org, num_spec, react_vol_init, add_sol_conc, add_cont_rate, t_cont,
                     add_one_shot, t_one_shot, add_col, sub_cont_rate, sub_aliq, t_aliq, sub_col, win=1, inc=1):
    add_pops = np.zeros((len(data_smooth(x_data_org, 0, win)), num_spec))
    vol = np.ones(len(add_pops)) * react_vol_init
    for i in range(num_spec):
        if add_col[i] is not None:
            add_pops[:, i] = data_smooth(data_org, add_col[i], win)
        else:
            add_pops_i = np.zeros((len(x_data_org), 1))
            if add_cont_rate[i] is not None and add_cont_rate[i] != 0:
                for j in range(len(add_cont_rate[i])):
                    index = find_nearest(x_data_org, t_cont[i][j])
                    for k in range(index + 1, len(x_data_org)):
                        add_pops_i[k] = add_pops_i[k - 1] + add_cont_rate[i][j] * \
                                        (x_data_org[k] - x_data_org[k - 1])
            if add_one_shot[i] is not None and add_one_shot[i] != 0:
                for j in range(len(add_one_shot[i])):
                    index = find_nearest(x_data_org, t_one_shot[i][j])
                    add_pops_i[index:] += add_one_shot[i][j]
            add_pops[:, i] = data_smooth(add_pops_i, 0, win)
    vol += add_pops.sum(axis=1)
    for i in range(num_spec):
        if add_sol_conc[i] is not None: add_pops[:, i] = add_pops[:, i] * add_sol_conc[i]

    if sub_col is not None:
        vol_loss = data_smooth(data_org, sub_col, win)
    else:
        vol_loss_i = np.zeros((len(x_data_org), 1))
        if sub_cont_rate is not None and sub_cont_rate != 0:
            for i in range(1, len(x_data_org)):
                vol_loss_i[i] = vol_loss_i[i - 1] + sub_cont_rate * \
                                (x_data_org[i] - x_data_org[i - 1])
        if sub_aliq[0] is not None and sub_aliq[0] != 0:
            for i in range(len(sub_aliq)):
                index = find_nearest(x_data_org, t_aliq[i])
                vol_loss_i[index:] += sub_aliq[i]
        vol_loss = data_smooth(vol_loss_i, 0, win)
    vol_loss = np.reshape(vol_loss, len(vol_loss))
    vol -= [np.float64(vol_loss[i]) for i in range(len(vol_loss))]
    vol_loss_rat = [1.0] + [1 - ((vol_loss[i] - vol_loss[i - 1]) / vol[i - 1]) for i in range(1, len(vol_loss))]

    if inc > 1:
        add_pops_add = np.zeros((len(add_sim(add_pops[:, 0], inc)), num_spec))
        for i in range(num_spec):
            add_pops_add[:, i] = add_sim(add_pops[:, i], inc)
        vol = add_sim(vol, inc)
        vol_loss_rat = add_sim(vol_loss_rat, inc)
    else:
        add_pops_add = add_pops

    add_pops_add_temp = np.zeros((len(add_pops_add), num_spec))
    for i in range(1, len(add_pops_add)):
        add_pops_add_temp[i] = add_pops_add[i] - add_pops_add[i - 1]
    add_pops = add_pops_add_temp

    return add_pops, vol, vol_loss_rat


# calculate temperature throughout
def get_temp(data_org, x_data_org, temp0, temp_cont, t_temp, temp_col, win=1, inc=1):
    if temp_col is not None:
        temp = data_smooth(data_org, temp_col, win)
    else:
        temp_i = np.zeros((len(x_data_org), 1))
        if temp_cont[0] is not None and temp_cont[0] != 0:
            for j in range(len(temp_cont)):
                index = find_nearest(x_data_org, t_temp[j])
                for k in range(index + 1, len(x_data_org)):
                    temp_i[k] = temp_i[k - 1] + temp_cont[j] * (x_data_org[k] - x_data_org[k - 1])
        temp = data_smooth(temp0 + temp_i, 0, win)
    temp_to_fit = add_sim(temp, inc)
    return temp, temp_to_fit


param_comb_dict = {
    "standard": [(-13, 13, 10, 'g')],
    "MM": [(-13, 13, 10, 'g'), (-10, 0, 10, 'g')],
    "Arrhenius": [(-15, 15, 10, 'g'), (0, 1E6, 20, 'a')],
    "Eyring": [(-1E3, 1E3, 40, 'a'), (-1E6, 1E6, 40, 'a')]
}


class FitSummary:
    def __init__(self, x_data, y_exp_conc, y_exp_rate, y_fit_conc, y_fit_rate, k_val, k_fit, k_fit_err,
                    ord_fit, ord_fit_err, pois_fit, pois_fit_err, fit_param_rss, fit_param_r2, temp, col, ord_lim):
        self.x = x_data
        self.y_exp_conc = y_exp_conc
        self.y_exp_rate = y_exp_rate
        self.y_fit_conc = y_fit_conc
        self.y_fit_rate = y_fit_rate
        self.k_est = k_val
        self.k_res = k_fit
        self.k_res_err = k_fit_err
        self.ord_res = ord_fit
        self.ord_res_err = ord_fit_err
        self.pois_res = pois_fit
        self.pois_res_err = pois_fit_err
        self.res_rss = fit_param_rss
        self.res_r2 = fit_param_r2
        self.temp = temp
        self.col = col
        self.ord_lim = ord_lim

    def __repr__(self):
        return (
            f"Optimization Result:\n"
            f"  x: {self.x, self.y_exp_conc}\n"
            f"  Experimental concentration(s): {self.y_exp_conc}\n"
            f"  Experimental rate: {self.y_exp_rate}\n"
            f"  Fitted concentration(s): {self.y_fit_conc}\n"
            f"  Fitted rate: {self.y_fit_rate}\n"
            f"  Resultant constant(s): {self.k_res}\n"
            f"  Resultant constant error(s): {self.k_res_err}\n"
            f"  Resultant order(s): {self.ord_res}\n"
            f"  Resultant order errors(s): {self.ord_res_err}\n"
            f"  Resultant poisoning(s): {self.pois_res}\n"
            f"  Resultant poisoning error(s): {self.pois_res_err}\n"
            f"  Residual sum of squares: {self.res_rss}\n"
            f"  R^2 (not recommended): {self.res_r2}\n"
            f"  Initial rate constant estimates: {self.k_est}\n"
            f"  Iterations: {self.k_est}\n"
        )


def get_multi_rate_eq(rxns):
    # Modify reactions to expand species with numbers before the first letter and remove common elements
    mod_rxns = []
    for r, p in rxns:
        if isinstance(r, str): r = (r,)
        if isinstance(p, str): p = (p,)
        mod_r, mod_p = [], []
        for i in r:
            if i[0].isdigit():
                num = int(i[0])
                letter = i[1:]
                mod_r.extend([letter] * num)
            else:
                mod_r.append(i)
        for i in p:
            if i[0].isdigit():
                num = int(i[0])
                letter = i[1:]
                mod_p.extend([letter] * num)
            else:
                mod_p.append(i)
        mod_rxns.append((mod_r, mod_p))

    # Determine unique species
    spec = []
    for r, p in mod_rxns:
        for spec_name in r + p:
            spec.append(spec_name)
    spec = [s for i, s in enumerate(spec) if spec.index(s) == i]

    rxns_r_tot, ord = [], []
    for r, _ in mod_rxns:
        rxns_r_i = [spec.index(spec_name) for spec_name in r]
        rxns_r_i_adj, ord_i = [], []
        for i in rxns_r_i:
            if i not in rxns_r_i_adj:
                rxns_r_i_adj.append(i)
                ord_i.append(1)
            else:
                index = rxns_r_i_adj.index(i)
                ord_i[index] += 1

        rxns_r_tot.append(tuple(rxns_r_i_adj))
        ord.append(tuple(ord_i))

    rxns_r, stoich, rate_loc = [], [], []
    for i in spec:
        stoich_i, spec_rxn_indices_i = [], []
        for j, (r, p) in enumerate(mod_rxns):
            r_count, p_count = r.count(i), p.count(i)

            if p_count - r_count != 0:
                rxns_r_i = tuple([spec.index(spec_name) for spec_name in r])
                stoich_i.append(p_count - r_count)
                spec_rxn_indices_i.append(j)

        rxns_r.append(rxns_r_i)
        stoich.append(stoich_i)
        rate_loc.append(spec_rxn_indices_i)

    # Print the species tuples, values, and reaction indices
    #for j, i in enumerate(spec):
    #    print(f"Species: {i}")
    #    for a, b, c, d in zip(mod_rxns, rxns_r[j], stoich[j], rate_loc[j]):
    #       print(f"Reaction: {a}, Reactant Indices: {b}, Stoich: {c}, Reaction Index: {d}")
    #    print()

    return spec, stoich, rate_loc, rxns_r_tot, ord


if __name__ == "__main__":
    rxns = [
        (['2A', 'B'], ['C']),
        (['C'], ['D']),
        (['C'], ['E']),
        (['D', 'A', 'E'], ['D', '2F']),
        (['G', 'C'], ['G', 'A'])
    ]
    spec, stoich, rate_loc, rxns_r, ord = get_multi_rate_eq(rxns)
    print(spec)
    print(stoich)
    print(rate_loc)
    print(rxns_r)
    print(ord)

def spec_err(spec_name, ord_min, ord_max, var_ord_locs, var_pois_locs, x_data_add_to_fit, y_data_to_fit, k_first_guess,
             pois_val, pois_min, pois_max):
    real_err_inc = 20  # enter number of increments between min and max orders
    real_err_inc += 1
    test_ord_pre = [np.linspace(ord_min[i], ord_max[i], real_err_inc).tolist() for i in range(len(var_ord_locs))]
    test_ord = list(itertools.product(*test_ord_pre))
    k_sec_guess = np.zeros([len(test_ord), len(var_ord_locs) + len(var_pois_locs) + 2])
    k_sec_guess[:] = [[*test_ord[i], 0, *np.zeros(len(var_pois_locs)), 0] for i in range(len(test_ord))]
    for i in range(len(k_sec_guess)):
        k_sec_guess_res = optimize.curve_fit(lambda x_data, k, *pois: eq_sim_fit(x_data, k,
                                                                                 *k_sec_guess[i,
                                                                                  :len(var_ord_locs)], *pois),
                                             x_data_add_to_fit, y_data_to_fit,
                                             [k_first_guess, *pois_val], maxfev=10000,
                                             bounds=((k_first_guess * bound_adj,
                                                      *pois_min), (k_first_guess / bound_adj, *pois_max)))
        k_sec_guess[i, len(var_ord_locs):-1] = k_sec_guess_res[0]
        fit_guess = eq_sim_fit(x_data_add_to_fit, k_sec_guess[i, len(var_ord_locs)],
                               *k_sec_guess[i, :len(var_ord_locs)],
                               *k_sec_guess[i, len(var_ord_locs) + 1:-len(var_pois_locs) - 1])
        _, k_sec_guess[i, -1] = residuals(y_data_to_fit, fit_guess)
        eq_sim_gen(stoich, mol0, mol_end, add_pops, vol, vol_loss_rat, temp, rate_eq,
                   inc, k_lim, ord_lim, r_locs, p_locs, c_locs, var_k_locs, var_ord_locs,
                   var_pois_locs, x_data_add, fit_param, fit_param_locs)
    real_err_calc_sort = k_sec_guess[k_sec_guess[:, -1].argsort()[::-1]]
    headers = [*[spec_name[i] + " order" for i in var_ord_locs], "k",
               *[spec_name[i] + " poisoning" for i in var_pois_locs], "r^2"]
    real_err_df = pd.DataFrame(real_err_calc_sort, columns=headers)
    return real_err_df

def k_sec_guess(x_data, y_data, var_ord_locs, var_pois_locs, ord_min, ord_max, pois_val,
                    bound_adj, k_first_guess):
    test_ord_pre = [list(range(round(ord_min[i]), round(ord_max[i]) + 1)) for i in range(len(var_ord_locs))]
    test_ord = list(itertools.product(*test_ord_pre))
    k_sec_guess = np.zeros([len(test_ord), len(var_ord_locs) + len(var_pois_locs) + 2])
    k_sec_guess[:] = [[*test_ord[i], 0, *np.zeros(len(var_pois_locs)), 0] for i in range(len(test_ord))]
    for i in range(len(k_sec_guess)):
        k_sec_guess_res = optimize.curve_fit(lambda x_data, k, *pois: eq_sim_fit(x_data, k,
                                                                                     *k_sec_guess[i,
                                                                                      :len(var_ord_locs)], *pois),
                                                 x_data, y_data,
                                                 [k_first_guess, *pois_val], maxfev=10000,
                                                 bounds=((k_first_guess * bound_adj,
                                                          *pois_min), (k_first_guess / bound_adj, *pois_max)))
        k_sec_guess[i, len(var_ord_locs):-1] = k_sec_guess_res[0]
        fit_guess = eq_sim_fit(x_data_add_to_fit, k_sec_guess[i, len(var_ord_locs)],
                                   *k_sec_guess[i, :len(var_ord_locs)],
                                   *k_sec_guess[i, len(var_ord_locs) + 1:-len(var_pois_locs) - 1])
        _, k_sec_guess[i, -1] = take_prep.residuals(y_data, fit_guess)
    index = np.where(k_sec_guess[:, -1] == max(k_sec_guess[:, -1]))
    k_sec_guess = float(k_sec_guess[index[0][0], 0])
    return k_sec_guess

def Bayseian_guess(ord_val, pois_val):
    def eq_sim_fit_ss(x_data_sim, *fit_param):
        return take_prep.residuals(y_data_to_fit, eq_sim_fit(x_data_sim, *fit_param))[0]
    n_random_points = 1000
    n_honing_points = 1
    hone_defaults = [-1E1, 5E3]
    hone_ranges = [(-1000, 1000), (-1E6, 1E6)]
    lam_hone = lambda k_fit: eq_sim_fit_ss(x_data_add_to_fit, k_fit, *ord_val, *pois_val)
    res_gp = gp_minimize(lam_hone, hone_ranges, x0=hone_defaults, n_initial_points=n_random_points,
                            n_calls=n_random_points + max(0, n_honing_points))
    print(res_gp)
    print(res_gp.x)
    return res_gp


def plot_other_fits_2D(x_data_df, y_exp_conc_df, y_fit_conc_df_arr, real_err_df, col, cutoff=0, f_format='svg', return_image=False,
                       save_disk=False, save_to='take_other_fits.svg', return_fig=False, transparent=False):
    for i in range(len(col_ext)):
        grid_shape = (int(round(np.sqrt(len(rows_cut)))), int(math.ceil(np.sqrt(len(rows_cut)))))
        fig = plt.figure(figsize=(grid_shape[0] * 5, grid_shape[1] * 5))
        plt.subplots_adjust(hspace=0.2, wspace=0.08)
        for j in rows_cut:
            ax = plt.subplot(grid_shape[0], grid_shape[1], j + 1)
            ax.plot(x_data * x_ax_scale, y_exp_conc[:, i] * y_ax_scale, '-o', color="black", label="Exp")
            y_fit_conc_df = y_fit_conc_df_arr[j][0]
            y_fit_conc = y_fit_conc_df.to_numpy()
            ax.plot(x_data * x_ax_scale, y_fit_conc[:, col_ext[i]] * y_ax_scale, color=std_colours[j],
                    label=real_err[j, 0])
            # color=std_colours[j]
            ax.set_xlim(calc_x_lim(x_data_adj, edge_adj))
            # ax.set_ylim([float(np.nanmin(y_fit_rate_adj[:, i]) - edge_adj * np.nanmax(y_fit_rate_adj[:, i]) * y_ax_scale),
            #    float(np.nanmax(y_fit_rate_adj[:, i])) * y_ax_scale * (1 + edge_adj)])

            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.tick_params(length=0, width=0)
            # ax.set_xlabel(x_label_text)
            # ax.set_ylabel(y_label_text)


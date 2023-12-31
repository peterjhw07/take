"""TAKE Plotting Functions"""

# Imports
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
import io
import base64
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


# calculate x limits from x data
def calc_x_lim(x_data, edge_adj):
    return [float(min(x_data) - (edge_adj * max(x_data))), float(max(x_data) * (1 + edge_adj))]


# calculate y limits from y data
def calc_y_lim(y_exp, y_fit, edge_adj):
    return [float(min(np.min(y_exp), np.min(y_fit)) - edge_adj * max(np.max(y_exp), np.max(y_fit))),
            float(max(np.max(y_exp), np.max(y_fit)) * (1 + edge_adj))]


# processes plotted data
def plot_process(return_fig, fig, f_format, save_disk, save_to, transparent):
    if return_fig:
        return fig, fig.get_axes()

    # correct mimetype based on filetype (for displaying in browser)
    if f_format == 'svg':
        mimetype = 'image/svg+xml'
    elif f_format == 'png':
        mimetype = 'image/png'
    elif f_format == 'jpg':
        mimetype = 'image/jpg'
    elif f_format == 'pdf':
        mimetype = 'application/pdf'
    elif f_format == 'eps':
        mimetype = 'application/postscript'
    else:
        raise ValueError('Image format {} not supported.'.format(format))

    # save to disk if desired
    if save_disk:
        plt.savefig(save_to, transparent=transparent)

    # save the figure to the temporary file-like object
    # plt.show()
    img = io.BytesIO()  # file-like object to hold image
    plt.savefig(img, format=f_format, transparent=transparent)
    plt.close()
    img.seek(0)
    return img, mimetype


# plot time vs conc
def plot_conc_vs_time(x_data_df, y_exp_conc_df=None, y_fit_conc_df=None, temp_df=None, col=None, show_asp=None,
                      method="lone", f_format='svg', return_image=False, save_disk=False,
                      save_to='take_fit.svg', return_fig=False, transparent=False):

    x_data = pd.DataFrame.to_numpy(x_data_df)

    if y_exp_conc_df is not None:
        y_exp_conc_headers = [i.replace(' conc. / moles_unit volume_unit$^{-1}$', '') for i in list(y_exp_conc_df.columns)]
        y_exp_conc = pd.DataFrame.to_numpy(y_exp_conc_df)
    else:
        y_exp_conc_headers = []
        y_exp_conc = pd.DataFrame.to_numpy(y_fit_conc_df)
    if y_fit_conc_df is not None:
        y_fit_conc_headers = [i.replace(' conc. / moles_unit volume_unit$^{-1}$', '') for i in list(y_fit_conc_df.columns)]
        y_fit_conc = pd.DataFrame.to_numpy(y_fit_conc_df)
    else:
        y_fit_conc_headers = []
        y_fit_conc = y_exp_conc
        y_fit_col = []
    if temp_df is not None:
        temp = pd.DataFrame.to_numpy(temp_df)

    if isinstance(show_asp, str): show_asp = [show_asp]
    if col is not None and show_asp is None:
        y_fit_col = [i for i in range(len(col)) if col[i] is not None]
        non_y_fit_col = [i for i in range(len(col)) if col[i] is None]
    if ("lone_all" in method or "sep_all" in method) and y_fit_conc_df is not None:
        show_asp = ["y"] * len(y_fit_conc_headers)
    if show_asp is not None:
        y_fit_col = [i for i in range(len(show_asp)) if 'y' in show_asp[i]]
        non_y_fit_col = [i for i in range(len(show_asp)) if 'n' in show_asp[i]]
    if "comp" in method and (len(non_y_fit_col) == 0 or y_fit_conc_df is None): method = "lone"
    if show_asp is not None and 'y' not in show_asp and 'y' not in show_asp[0]:
        print("If used, show_asp must contain at least one 'y'. Plot time_vs_conc has been skipped.")
        return

    # graph results
    x_ax_scale = 1
    y_ax_scale = 1
    edge_adj = 0.02
    std_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    del std_colours[3]
    std_colours = std_colours * 10

    x_data_adj = x_data * x_ax_scale
    y_exp_conc_adj = y_exp_conc * y_ax_scale
    y_fit_conc_adj = y_fit_conc * y_ax_scale
    temp_adj = temp

    x_label_text = list(x_data_df.columns)[0]
    y_label_text = "Concentration / moles_unit volume_unit$^{-1}$"

    cur_exp = 0
    cur_clr = 0
    if "lone" in method:  # lone plots a single figure containing all exps and fits as specified
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
        #plt.rcParams.update({'font.size': 15})
        ax1.set_xlabel(x_label_text)
        ax1.set_ylabel(y_label_text)
        for i in range(len(y_exp_conc_headers)):
            if len(x_data_adj) <= 50:
                ax1.scatter(x_data_adj, y_exp_conc_adj[:, i], label=y_exp_conc_headers[i])
            else:
                ax1.plot(x_data_adj, y_exp_conc_adj[:, i], label=y_exp_conc_headers[i])
        for i in y_fit_col:
            ax1.plot(x_data_adj, y_fit_conc_adj[:, i], label=y_fit_conc_headers[i])
        if temp_df is not None and not np.all(temp == temp[0]):
            ax2 = ax1.twinx()
            ax2.plot(x_data_adj, temp_adj, label='Temperature', color='red')
            ax2.set_ylabel('Temperature / temp. unit', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_xlim(calc_x_lim(x_data_adj, edge_adj))
            ax2.set_ylim(calc_y_lim(temp_adj, temp_adj, edge_adj))
        if len(y_fit_col) == 0: y_fit_col = range(len(y_exp_conc_headers))
        ax1.set_xlim(calc_x_lim(x_data_adj, edge_adj))
        ax1.set_ylim(calc_y_lim(y_exp_conc_adj, y_fit_conc_adj[:, y_fit_col], edge_adj))
        ax1.legend(prop={'size': 10}, frameon=False)

    elif "comp" in method:  # plots two figures, with the first containing show_asp (or col if show_asp not specified) and the second containing all fits
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        #plt.rcParams.update({'font.size': 15})
        for i in range(len(y_exp_conc_headers)):
            if len(x_data_adj) <= 50:
                ax1.scatter(x_data_adj, y_exp_conc_adj[:, i], color=std_colours[cur_clr],
                            label=y_exp_conc_headers[i])
                ax2.scatter(x_data_adj, y_exp_conc_adj[:, i], color=std_colours[cur_clr],
                            label=y_exp_conc_headers[i])
                cur_clr += 1
            else:
                ax1.plot(x_data_adj, y_exp_conc_adj[:, i], color=std_colours[cur_clr],
                            label=y_exp_conc_headers[i])
                ax2.plot(x_data_adj, y_exp_conc_adj[:, i], color=std_colours[cur_clr],
                            label=y_exp_conc_headers[i])
                cur_clr += 1
        for i in y_fit_col:
            ax1.plot(x_data_adj, y_fit_conc_adj[:, i], color=std_colours[cur_clr],
                            label=y_fit_conc_headers[i])
            ax2.plot(x_data_adj, y_fit_conc_adj[:, i], color=std_colours[cur_clr],
                            label=y_fit_conc_headers[i])
            cur_clr += 1
        for i in non_y_fit_col:
            ax2.plot(x_data_adj, y_fit_conc_adj[:, i], color=std_colours[cur_clr],
                            label=y_fit_conc_headers[i])
            cur_clr += 1
        if temp_df is not None and not np.all(temp == temp[0]):
            ax3 = ax2.twinx()
            ax3.plot(x_data_adj, temp_adj, label='Temperature', color='red')
            ax3.set_ylabel('Temperature / temp. unit', color='red')
            ax3.tick_params(axis='y', labelcolor='red')
            ax3.set_xlim(calc_x_lim(x_data_adj, edge_adj))
            ax3.set_ylim(calc_y_lim(temp_adj, temp_adj, edge_adj))

        ax1.set_xlim(calc_x_lim(x_data_adj, edge_adj))
        ax1.set_ylim(calc_y_lim(y_exp_conc_adj, y_fit_conc_adj[:, y_fit_col], edge_adj))
        ax2.set_xlim(calc_x_lim(x_data_adj, edge_adj))
        ax2.set_ylim(calc_y_lim(y_exp_conc_adj, y_fit_conc_adj, edge_adj))

        ax1.set_xlabel(x_label_text)
        ax1.set_ylabel(y_label_text)
        ax2.set_xlabel(x_label_text)
        ax2.set_ylabel(y_label_text)
        ax1.legend(prop={'size': 10}, frameon=False)
        ax2.legend(prop={'size': 10}, frameon=False)

    elif "sep" in method:
        temp_add = 0 if temp_df is None or np.all(temp == temp[0]) else 1
        num_spec = max([len(y_exp_conc_headers), len(y_fit_conc_headers)])
        grid_shape = (int(round(np.sqrt(len(y_fit_col) + temp_add))), int(math.ceil(np.sqrt(len(y_fit_col) + temp_add))))
        fig = plt.figure(figsize=(grid_shape[1] * 6, grid_shape[0] * 5))
        # plt.subplots_adjust(hspace=0.4, wspace=0.4)
        for j, i in enumerate(y_fit_col):
            ax = plt.subplot(grid_shape[0], grid_shape[1], j + 1)
            if col is not None and col[i] is not None and y_exp_conc_df is not None:
                if len(x_data_adj) <= 50:
                    ax.scatter(x_data_adj, y_exp_conc_adj[:, cur_exp], color=std_colours[cur_clr], label=y_exp_conc_headers[cur_exp])
                else:
                    ax.plot(x_data_adj, y_exp_conc_adj[:, cur_exp], color=std_colours[cur_clr], label=y_exp_conc_headers[cur_exp])
                ax.set_ylim(calc_y_lim(y_exp_conc_adj[:, cur_exp], y_fit_conc_adj[:, i], edge_adj))
                cur_exp += 1
                cur_clr += 1
            else:
                ax.set_ylim(calc_y_lim(y_fit_conc_adj[:, i], y_fit_conc_adj[:, i], edge_adj))
            if y_fit_conc_df is not None:
                ax.plot(x_data_adj, y_fit_conc_adj[:, i], color=std_colours[cur_clr], label=y_fit_conc_headers[i])
            cur_clr += 1

            ax.set_xlim(calc_x_lim(x_data_adj, edge_adj))
            ax.set_xlabel(x_label_text)
            ax.set_ylabel(y_label_text)
            plt.legend(prop={'size': 10}, frameon=False)
        if temp_df is not None and not np.all(temp == temp[0]):
            ax = plt.subplot(grid_shape[0], grid_shape[1], j + 2)
            ax.plot(x_data_adj, temp_adj, color="red", label="Temperature")

            ax.set_xlim(calc_x_lim(x_data_adj, edge_adj))
            ax.set_ylim(calc_y_lim(temp_adj, temp_adj, edge_adj))
            ax.set_xlabel(x_label_text)
            ax.set_ylabel("Temperature / temp_unit")
            plt.legend(prop={'size': 10}, frameon=False)
    else:
        print("Invalid method inputted. Please enter appropriate method or remove method argument.")
        return

    # plt.show()
    img, mimetype = plot_process(return_fig, fig, f_format, save_disk, save_to, transparent)
    if not return_image:
        graph_url = base64.b64encode(img.getvalue()).decode()
        return 'data:{};base64,{}'.format(mimetype, graph_url)
    else:
        return img, mimetype


# plot rate vs conc
def plot_rate_vs_conc(x_data_df, y_fit_conc_df, y_fit_rate_df, ord, y_exp_conc_df=None, y_exp_rate_df=None,
                      show_asp=None, f_format='svg', return_image=False, save_disk=False,
                     save_to='take_conc_vs_rate.svg', return_fig=False, transparent=False):
    x_data, y_fit_conc, y_fit_rate = map(pd.DataFrame.to_numpy, [x_data_df, y_fit_conc_df, y_fit_rate_df])
    if y_exp_conc_df is not None:
        pd.DataFrame.to_numpy(y_exp_conc_df)
    if y_exp_rate_df is not None:
        pd.DataFrame.to_numpy(y_exp_rate_df)

    if show_asp is not None:
        y_fit_col = [i for i in range(len(show_asp)) if 'y' in show_asp[i]]

    # y_exp_conc_headers = list(y_exp_conc_df.columns)
    y_fit_conc_headers = list(y_fit_conc_df.columns)
    # y_exp_rate_adj_headers = [i.replace('fit conc. / moles_unit volume_unit$^{-1}$', 'exp.') for i in list(y_fit_conc_df.columns)]
    y_fit_rate_headers = list(y_fit_rate_df.columns)
    y_fit_rate_headers_adj = [i.replace(' conc. / moles_unit volume_unit$^{-1}$', '') for i in list(y_fit_conc_df.columns)]

    x_ax_scale = 1
    y_ax_scale = 1
    edge_adj = 0.02
    for i in range(len(ord)):
        num_spec = len(ord[i])
        # y_exp_rate_adj = np.empty((len(y_exp_rate), num_spec))
        y_fit_rate_adj = np.empty((len(y_fit_rate), num_spec))
        for j in range(num_spec):
            # y_exp_rate_adj[:, j] = np.divide(y_exp_rate.reshape(len(y_exp_rate)), np.product([y_fit_conc[:, k] ** orders[k] for k in range(num_spec) if j != k], axis=0))
            y_fit_rate_adj[:, j] = np.divide(y_fit_rate.reshape(len(y_fit_rate[:, i])), np.product([y_fit_conc[:, k] ** ord[i][k] for k in range(num_spec) if j != k], axis=0))
        # y_exp_rate_adj_df = pd.DataFrame(y_exp_rate_adj, columns=y_exp_rate_adj_headers)
        y_fit_rate_adj_df = pd.DataFrame(y_fit_rate_adj, columns=y_fit_rate_headers_adj)

        grid_shape = (int(round(np.sqrt(num_spec))), int(math.ceil(np.sqrt(num_spec))))

        fig = plt.figure(figsize=(grid_shape[0] * 5, grid_shape[1] * 5))
        # plt.subplots_adjust(hspace=0.5)
        std_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
        y_label_text = "Rate / moles_unit volume_unit$^{-1}$ time_unit$^{-1}$"
        for j in range(num_spec):
            x_label_text = y_fit_conc_headers[j]
            ax = plt.subplot(grid_shape[0], grid_shape[1], j + 1)
            ax.set_xlabel(x_label_text)
            ax.set_ylabel(y_label_text)
            # ax.scatter(y_fit_conc[:, j] * x_ax_scale, y_exp_rate_adj[:, j] * y_ax_scale, color=std_colours[j])
            ax.plot(y_fit_conc[:, j] * x_ax_scale, y_fit_rate_adj[:, j] * y_ax_scale, color=std_colours[j])
            ax.set_xlim([float(min(y_fit_conc[:, j] * x_ax_scale) - (edge_adj * max(y_fit_conc[:, j] * x_ax_scale))),
                    float(max(y_fit_conc[:, j] * x_ax_scale) * (1 + edge_adj))])
            # ax.set_ylim([float(np.nanmin(y_fit_rate_adj[:, i]) - edge_adj * np.nanmax(y_fit_rate_adj[:, i]) * y_ax_scale),
            #        float(np.nanmax(y_fit_rate_adj[:, i])) * y_ax_scale * (1 + edge_adj)])
        # plt.show()
        save_to_replace = save_to.replace('.png', '_rates.png')
        img, mimetype = plot_process(return_fig, fig, f_format, save_disk, save_to_replace, transparent)
    if len(ord) == 1:
        if not return_image:
            graph_url = base64.b64encode(img.getvalue()).decode()
            return 'data:{};base64,{}'.format(mimetype, graph_url)
        else:
            return img, mimetype


# plot other fits in 2D
def plot_other_fits_2D(x_data_df, y_exp_conc_df, y_fit_conc_df_arr, real_err_df, col, cutoff=0, f_format='svg', return_image=False,
                       save_disk=False, save_to='take_other_fits.svg', return_fig=False, transparent=False):
    num_spec = len(col)
    x_data, y_exp_conc, real_err = map(pd.DataFrame.to_numpy, [x_data_df, y_exp_conc_df, real_err_df])
    # np.savetxt(r"C:\Users\Peter\Desktop\real_err.csv", real_err, delimiter="\t", fmt='%s')
    if cutoff is not None:
        cut_thresh = cutoff * real_err[0, -1]
    else:
        cut_thresh = real_err[-1, -1]
    rows_cut = [i for i, x in enumerate(real_err[:, -1] >= cut_thresh) if x]
    cur_clr = 0

    col_ext = [i for i in range(len(col)) if col[i] is not None]
    grid_shape = (int(round(np.sqrt(len(col_ext)))), int(math.ceil(np.sqrt(len(col_ext)))))

    x_ax_scale = 1
    y_ax_scale = 1
    edge_adj = 0.02

    x_data_adj = x_data * x_ax_scale
    fig = plt.figure(figsize=(grid_shape[0] * 5, grid_shape[1] * 5))
    # plt.subplots_adjust(hspace=0.5)
    std_colours = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 100

    x_label_text = "Time / time_unit"
    y_label_text = "Concentration / moles_unit volume_unit$^{-1}$"
    for i in range(len(col_ext)):
        ax = plt.subplot(grid_shape[0], grid_shape[1], i + 1)
        ax.plot(x_data * x_ax_scale, y_exp_conc[:, i] * y_ax_scale, '-o', color="black", label="Exp")
        for j in rows_cut:
            y_fit_conc_df = y_fit_conc_df_arr[j][0]
            y_fit_conc = y_fit_conc_df.to_numpy()
            ax.plot(x_data_adj, y_fit_conc[:, col_ext[i]] * y_ax_scale, label=real_err[j, 0])
            #color=std_colours[j]
        ax.set_xlim(calc_x_lim(x_data_adj, edge_adj))
        #ax.set_ylim([float(np.nanmin(y_fit_rate_adj[:, i]) - edge_adj * np.nanmax(y_fit_rate_adj[:, i]) * y_ax_scale),
        #    float(np.nanmax(y_fit_rate_adj[:, i])) * y_ax_scale * (1 + edge_adj)])

        ax.set_xlabel(x_label_text)
        ax.set_ylabel(y_label_text)
        ax.legend(prop={'size': 10}, frameon=False)

    save_to_replace = save_to.replace('.png', '_other_fits_2D.png')
    img, mimetype = plot_process(return_fig, fig, f_format, save_disk, save_to_replace, transparent)
    # plt.show()

    if not return_image:
        graph_url = base64.b64encode(img.getvalue()).decode()
        return 'data:{};base64,{}'.format(mimetype, graph_url)
    else:
        return img, mimetype


# plot other fits in 3D (contour map and 3D projection)
def plot_other_fits_3D(real_err_df, cutoff=None, f_format='svg', return_image=False, save_disk=False,
                     save_to='take_other_fits.svg', return_fig=False, transparent=False):
    real_err_arr = pd.DataFrame.to_numpy(real_err_df)
    if cutoff is None:
        real_err_arr_cut = real_err_arr
    else:
        real_err_arr_cut = real_err_arr[real_err_arr[:, -1] > cutoff, :]
    cont_x_org = [real_err_arr_cut[i, 0][0] for i in range(len(real_err_arr_cut))]
    cont_y_org = [real_err_arr_cut[i, 0][1] for i in range(len(real_err_arr_cut))]
    cont_z_org = real_err_arr_cut[:, -1]
    cont_x_add, cont_y_add = np.linspace(min(cont_x_org), max(cont_x_org), 1000), \
                             np.linspace(min(cont_y_org), max(cont_y_org), 1000)
    cont_x_plot, cont_y_plot = np.meshgrid(cont_x_add, cont_y_add)
    cont_z_plot = interpolate.griddata((cont_x_org, cont_y_org), cont_z_org, (cont_x_plot, cont_y_plot), method='linear')
    # rbf = scipy.interpolate.Rbf(cont_x_org, cont_y_org, cont_z_org, function='linear')
    # cont_z_plot = rbf(cont_x_plot, cont_y_plot)

    cont_fig = plt.imshow(cont_z_plot, vmin=cont_z_org.min(), vmax=cont_z_org.max(), origin='lower', cmap='coolwarm',
               extent=[min(cont_x_org), max(cont_x_org), min(cont_y_org), max(cont_y_org)], aspect='auto')
    # plt.scatter(cont_x_org, cont_y_org, c=cont_z_org, cmap='coolwarm')
    plt.xlabel('Order 1'), plt.ylabel('Order 2')
    plt.colorbar()
    img, mimetype = plot_process(return_fig, cont_fig, f_format, save_disk, save_to.replace('.png', '_other_fits_contour.png'), transparent)

    fig_3D = plt.axes(projection='3d')
    fig_3D.plot_surface(cont_x_plot, cont_y_plot, cont_z_plot, cmap='coolwarm')  # rstride=1, cstride=1
    fig_3D.set_xlabel('Order 1'), fig_3D.set_ylabel('Order 2'), fig_3D.set_zlabel('r^2')
    img, mimetype = plot_process(return_fig, fig_3D, f_format, save_disk, save_to.replace('.png', '_other_fits_3D.png'), transparent)
    if not return_image:
        graph_url = base64.b64encode(img.getvalue()).decode()
        return 'data:{};base64,{}'.format(mimetype, graph_url)
    else:
        return img, mimetype

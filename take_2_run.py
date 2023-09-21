"""TAKE Fitting Programme"""

# Imports
import take_2 as take
import ast
import numpy as np
import pandas as pd
import timeit
from datetime import date
import re
import pickle


# Function for sorting Excel data into pythonic format.
def input_sort(s):
    '''if ', ' in str(s):
        split = re.split(r',\s*(?![^()]*\))', str(s))
    else:
        split = s
    if isinstance(split, np.int64):
        split = np.int64.item(split)
    elif isinstance(split, np.float64):
        split = np.float64.item(split)
    if isinstance(split, (int, float)):
        s_adj = split
    elif "None" in split and isinstance(split, str):
        s_adj = None
    elif not isinstance(split, (int, str)):
        print(split)
        s_adj = [eval(str(x)) for x in split]
    else:
        s_adj = split
    return s_adj'''
    if isinstance(s, str):
        return ast.literal_eval(s)
    else:
        return s


def export_Excel(data, file_name, mode, if_sheet_exists, sheet_name):
    data_store_try = False
    while data_store_try is False:
        try:
            with pd.ExcelWriter(file_name, mode=mode, if_sheet_exists=if_sheet_exists) as writer:
                data.to_excel(writer, sheet_name=sheet_name, index=False)
            data_store_try = True
        except PermissionError:
            input('Error! Export file open. Close and then press enter.')
        except OSError:
            file_name = input('Error! Cannot save file into a non-existent folder. '
                                'Input correct file location (do not use r\'\'). \n')
        except:
            print('Unknown error! Check inputs are formatted correctly. Else examine error messages and review code.')


if __name__ == "__main__":
    sim_or_fit = "sim"
    excel_source = "y"
    export_fit = "n"
    export_param = "y"
    exp_err = "n"
    plot_type = "sep_all"  # options are lone, lone_all, sep, sep_all, comp

    if "sim" in sim_or_fit and 'n' in excel_source:
        spec_name = ["r1", "r2", "p1", "c1"]
        spec_type = ["r", "r", "p", "c"]
        react_vol_init = 4E-3
        stoich = [1, 1, 1, None]  # insert stoichiometry of reactant, r
        mol0 = [10, 0, 0, 0]
        mol_end = [0, None, 10, 0]
        add_sol_conc = [None, 1E7, None, 0.709]
        add_cont_rate = [None, 10E-6, None, None]
        t_cont = [None, 1, None, None]
        add_one_shot = [None, None, None, 100E-6]
        t_one_shot = [None, None, None, 1]
        k_lim = [3.5]
        ord_lim = [1, 0.0000001, 0, 1]
        pois_lim = [0, 0, 0, 0]
        t_param = (0, 100, 0.1)
        fit_asp = ["y", "n", "y", "y"]
        pic_save = 'take.svg'

        x_data_df, y_fit_conc_df, y_fit_rate_df = take.sim_take(t_param, spec_type, react_vol_init, spec_name=spec_name,
                                                                  stoich=stoich, mol0=mol0, mol_end=mol_end, add_sol_conc=add_sol_conc,
                                                                  add_cont_rate=add_cont_rate, t_cont=t_cont, add_one_shot=add_one_shot,
                                                                  t_one_shot=t_one_shot, k_lim=k_lim, ord_lim=ord_lim, pois_lim=pois_lim)

        plot_output = take.plot_conc_vs_time(x_data_df, y_fit_conc_df=y_fit_conc_df, show_asp=show_asp,
                                             method=plot_type, f_format='png', save_disk=True, save_to=pic_save)

    elif "sim" in sim_or_fit and 'y' in excel_source:
        df = pd.read_excel(r'C:\Users\Peter\Documents\Postdoctorate\Work\TAKE\Programmes\TAKE_Test_Input.xlsx',
                       sheet_name='Sim_testing')
        df.replace('""', None, inplace=True)

        total = np.empty([len(df), 12], object)
        for i in range(0, len(df)):
            [number, react_type, spec_name, spec_type, rxns, react_vol_init, stoich, mol0, mol_end,
             add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, sub_cont_rate, sub_aliq, t_aliq,
             temp0, temp_cont, t_temp, rate_eq_type, k_lim, ord_lim, pois_lim, t_param,
             rand_fac, show_asp, pic_save] = df.iloc[i, :28]
            print(number, react_type)
            spec_name, spec_type, rxns, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot,\
            t_one_shot, sub_cont_rate, sub_aliq, t_aliq, temp0, temp_cont, t_temp, rate_eq_type, \
            k_lim, ord_lim, pois_lim, show_asp, t_param, rand_fac \
                                    = map(input_sort, [spec_name, spec_type, rxns,
                                    stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot,
                                    t_one_shot, sub_cont_rate, sub_aliq, t_aliq, temp0, temp_cont, t_temp, rate_eq_type,
                                    k_lim, ord_lim, pois_lim, show_asp, t_param, rand_fac])
            t_param = tuple(t_param)
            x_data_df, y_fit_conc_df, y_fit_rate_df, temp_df, ord_lim = take.sim_take(t_param, spec_name=spec_name, spec_type=spec_type,
                            rxns=rxns, react_vol_init=react_vol_init, stoich=stoich, mol0=mol0, mol_end=mol_end, add_sol_conc=add_sol_conc,
                            add_cont_rate=add_cont_rate, t_cont=t_cont, add_one_shot=add_one_shot,
                            t_one_shot=t_one_shot, sub_cont_rate=sub_cont_rate, sub_aliq=sub_aliq, t_aliq=t_aliq,
                            temp0=temp0, temp_cont=temp_cont, t_temp=t_temp, rate_eq_type=rate_eq_type,
                            k_lim=k_lim, ord_lim=ord_lim, pois_lim=pois_lim, rand_fac=rand_fac)

            plot_output = take.plot_conc_vs_time(x_data_df, y_fit_conc_df=y_fit_conc_df, show_asp=show_asp,
                                                 temp_df=temp_df, method=plot_type,
                                                 f_format='png', save_disk=True, save_to=pic_save)
            #plot_output = take.plot_rate_vs_conc(x_data_df, y_fit_conc_df, y_fit_rate_df, ord_lim, show_asp=show_asp,
            #                                     f_format='png', save_disk=True, save_to=pic_save)

            if 'y' in export_fit:
                all_data = pd.concat((x_data_df, y_fit_conc_df, y_fit_rate_df), axis=1)
                exportdf = pd.DataFrame(all_data)
                export_Excel(exportdf,
                             r'C:\Users\Peter\Documents\Postdoctorate\Work\TAKE\Programmes\TAKE_Test_Results_Sim.xlsx',
                             'a', 'replace', str(number))

    elif "fit" in sim_or_fit and 'y' in excel_source:
        # create dataframe
        df = pd.read_excel(r'C:\Users\Peter\Documents\Postdoctorate\Work\TAKE\Programmes\TAKE_Test_Input.xlsx', sheet_name='Testing')
        df.replace('""', None, inplace=True)

        total = np.empty([len(df), 12], object)
        for i in range(0, len(df)):
            [number, react_type, spec_name, spec_type, rxns, react_vol_init, stoich, mol0, mol_end, add_sol_conc,
             add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col, sub_cont_rate, sub_aliq, t_aliq, sub_col,
             t_col, col, temp0, temp_cont, t_temp, temp_col, rate_eq_type, k_lim, ord_lim, pois_lim, fit_asp, TIC_col,
             scale_avg_num, win, inc, file_name, sheet_name, pic_save] = df.iloc[i, :37]
            print(number, react_type)

            #spec_type, fit_asp = map(make_char_tup, [spec_type, fit_asp])
            spec_name, spec_type, rxns, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot,\
            t_one_shot, add_col, sub_cont_rate, sub_aliq, t_aliq, sub_col, col, temp0, temp_cont, t_temp, temp_col, \
            rate_eq_type, k_lim, ord_lim, pois_lim, fit_asp = map(input_sort, [spec_name, spec_type, rxns, stoich,
                    mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col,
                    sub_cont_rate, sub_aliq, t_aliq, sub_col, col, temp0, temp_cont, t_temp, temp_col,
                    rate_eq_type, k_lim, ord_lim, pois_lim, fit_asp])
            sheet_name = str(sheet_name)

            # print(spec_type, react_vol_init, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col, t_col, col, k_lim, ord_lim, pois_lim, fit_asp, TIC_col, scale_avg_num, win, inc, file_name, sheet_name, pic_save)
            data = take.read_data(file_name, sheet_name, t_col, col, add_col, sub_col, temp_col)
            starttime = timeit.default_timer()

            if 'y' not in exp_err:
                output = take.fit_take(data, spec_name=spec_name, spec_type=spec_type, rxns=rxns,
                            react_vol_init=react_vol_init, stoich=stoich,
                            mol0=mol0, mol_end=mol_end, add_sol_conc=add_sol_conc, add_cont_rate=add_cont_rate,
                            t_cont=t_cont, add_one_shot=add_one_shot,t_one_shot=t_one_shot, add_col=add_col,
                            sub_cont_rate=sub_cont_rate, sub_aliq=sub_aliq, t_aliq=t_aliq, sub_col=sub_col, t_col=t_col,
                            col=col, temp0=temp0, temp_cont=temp_cont, t_temp=t_temp, temp_col=temp_col,
                            rate_eq_type=rate_eq_type, k_lim=k_lim, ord_lim=ord_lim, pois_lim=pois_lim, fit_asp=fit_asp,
                            scale_avg_num=scale_avg_num, win=win, inc=inc)
                with open("venv/fit_output.pkl", 'wb') as outp:
                    pickle.dump(output, outp, pickle.HIGHEST_PROTOCOL)
                x_data_df, y_exp_conc_df, y_exp_rate_df, y_fit_conc_df, y_fit_rate_df, k_val_est, k_fit, k_fit_err, \
                ord_fit, ord_fit_err, pois_fit, pois_fit_err, res_rss, res_r2, temp_df, col_proc, ord_lim_proc = output
                time_taken = timeit.default_timer() - starttime
                print(k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err, res_rss, res_r2, time_taken)

                plot_output = take.plot_conc_vs_time(x_data_df, y_exp_conc_df=y_exp_conc_df, y_fit_conc_df=y_fit_conc_df,
                                                    temp_df=temp_df, col=col_proc, method=plot_type,
                                                     f_format='png', save_disk=True, save_to=pic_save)

                '''
                total_ord, k = [], 0
                for j in range(len(ord_lim_proc)):
                    if isinstance(ord_lim_proc[j], tuple):
                        total_ord.append(ord_fit[k])
                        k += 1
                    else:
                        total_ord.append(ord_lim_proc[j])
                plot_output = take.plot_rate_vs_conc(x_data_df, y_fit_conc_df, y_fit_rate_df, total_ord,
                                                                y_exp_conc_df=y_exp_conc_df, y_exp_rate_df=y_exp_rate_df,
                                                                f_format='png', save_disk=True, save_to=pic_save)
                '''
            elif 'y' in exp_err:
                output = take.fit_err_real(data, spec_name=spec_name, spec_type=spec_type, rxns=rxns,
                        react_vol_init=react_vol_init, stoich=stoich,
                        mol0=mol0, mol_end=mol_end, add_sol_conc=add_sol_conc, add_cont_rate=add_cont_rate,
                        t_cont=t_cont, add_one_shot=add_one_shot,t_one_shot=t_one_shot, add_col=add_col,
                        sub_cont_rate=sub_cont_rate, sub_aliq=sub_aliq, t_aliq=t_aliq, sub_col=sub_col, t_col=t_col,
                        col=col, temp0=temp0, temp_cont=temp_cont, t_temp=t_temp, temp_col=temp_col,
                        rate_eq_type=rate_eq_type, k_lim=k_lim, ord_lim=ord_lim, pois_lim=pois_lim, fit_asp=fit_asp,
                        scale_avg_num=scale_avg_num, win=win, inc=inc)

                with open("venv/exp_err_output.pkl", 'wb') as outp:
                    pickle.dump(output, outp, pickle.HIGHEST_PROTOCOL)
                x_data_df, y_exp_conc_df, y_exp_rate_df, y_fit_conc, y_fit_rate, real_err_sort, col, ord_lim = output
                take.plot_other_fits_2D(x_data_df, y_exp_conc_df, y_fit_conc, real_err_sort, col, cutoff=0.997, save_disk=True, save_to=pic_save)
                take.plot_other_fits_3D(real_err_sort, cutoff=0.997, save_disk=True, save_to=pic_save)

            if 'y' in export_fit:
                all_data = pd.concat((x_data_df, y_exp_conc_df, y_fit_conc_df, y_fit_rate_df), axis=1)
                exportdf = pd.DataFrame(all_data)
                export_Excel(exportdf,
                             r'C:\Users\Peter\Documents\Postdoctorate\Work\TAKE\Programmes\TAKE_Test_Results_Fit.xlsx',
                             'a', 'replace', str(number))

            total[i] = [number, react_type, k_val_est, k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err, res_rss, res_r2, time_taken]

        if 'y' in export_param:
            exportdf = pd.DataFrame({"Number":total[:, 0],"Type": total[:, 1],"k_est":total[:, 2],
                         "k_res":total[:, 3], "k_res_err":total[:, 4], "ord_res":total[:, 5],
                         "ord_res_err":total[:, 6], "pois_res":total[:, 7], "pois_res_err":total[:, 8],
                         "res_sum_squares":total[:, 9], "R^2":total[:, 10],
                         "script_runtime":total[:, 11]})
            export_Excel(exportdf,
                         r'C:\Users\Peter\Documents\Postdoctorate\Work\TAKE\Programmes\TAKE_Test_Results.xlsx',
                         'a', 'new', date.today().strftime("%y%m%d"))

    elif "import" in sim_or_fit:
        # create dataframe
        df = pd.read_excel(r'C:\Users\Peter\Documents\Postdoctorate\Work\TAKE\Programmes\TAKE_Test_Input.xlsx', sheet_name='Testing')
        df.replace('""', None, inplace=True)

        total = np.empty([len(df), 12], object)
        for i in range(0, len(df)):
            [number, react_type, spec_name, spec_type, rxns, react_vol_init, stoich, mol0, mol_end,
             add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col,
             sub_cont_rate, sub_aliq, t_aliq, sub_col, t_col, col, temp0, temp_cont, t_temp, temp_col, rate_eq_type,
             k_lim, ord_lim, pois_lim, fit_asp,
             TIC_col, scale_avg_num, win, inc, file_name, sheet_name, pic_save] = df.iloc[i, :38]
            print(number, react_type)

            #spec_type, fit_asp = map(make_char_tup, [spec_type, fit_asp])
            spec_name, spec_type, rxns, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot,\
            t_one_shot, add_col, sub_cont_rate, sub_aliq, t_aliq, sub_col, col, \
            k_lim, ord_lim, pois_lim, fit_asp = map(input_sort, [spec_name, spec_type, rxns, stoich,
                    mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col,
                    sub_cont_rate, sub_aliq, t_aliq, sub_col, col, k_lim, ord_lim, pois_lim, fit_asp])
            sheet_name = str(sheet_name)

            # print(spec_type, react_vol_init, stoich, mol0, mol_end, add_sol_conc, add_cont_rate, t_cont, add_one_shot, t_one_shot, add_col, t_col, col, k_lim, ord_lim, pois_lim, fit_asp, TIC_col, scale_avg_num, win, inc, file_name, sheet_name, pic_save)
            data = take.read_data(file_name, sheet_name, t_col, col, add_col, sub_col, temp_col)
            starttime = timeit.default_timer()
        with open("venv/fit_output.pkl", 'rb') as inp:
            output = pickle.load(inp)
            x_data_df, y_exp_conc_df, y_exp_rate_df, y_fit_conc_df, y_fit_rate_df, k_val_est, k_fit, k_fit_err, \
            ord_fit, ord_fit_err, pois_fit, pois_fit_err, res_rss, res_r2, col, ord_lim = output
            time_taken = timeit.default_timer() - starttime
            print(k_fit, k_fit_err, ord_fit, ord_fit_err, pois_fit, pois_fit_err)
            # y_exp_conc_df=y_exp_conc_df
            plot_output = take.plot_conc_vs_time(x_data_df, y_exp_conc_df=y_exp_conc_df, y_fit_conc_df=y_fit_conc_df,
                                                col=col, method=plot_type, f_format='png', save_disk=True, save_to=pic_save)

        if 'y' in exp_err:
            with open("venv/exp_err_output.pkl", 'rb') as inp:
                output = pickle.load(inp)
                x_data, y_exp_conc, y_exp_rate, y_fit_conc, y_fit_rate, real_err_sort, col, ord_lim = output
                take.plot_other_fits_2D(x_data, y_exp_conc, y_fit_conc, real_err_sort, col, cutoff=0.6, save_disk=True, save_to=pic_save)
                take.plot_other_fits_3D(real_err_sort, cutoff=0.6, save_disk=True, save_to=pic_save)

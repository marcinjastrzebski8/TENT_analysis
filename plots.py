import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from metrics import compute_avg_metrics

def metric_vs_train_size(canvas, metric, train_sizes, job_specs: List[List[Dict]], legend):
    """
    Compare performance of different kernels wrt a given Huang metric with growing training size.
    canvas:    plt.ax
    job_specs: each entry is a list of specs (dicts) for a given model in increasing training size
               one way to think about it is: rows <-> models, columns <-> training sizes
               with a caveat that the training sizes can differ between models
               #NOTE: That functionality isn't supported yet though
    """
    y_vals = np.zeros(shape = np.shape(job_specs))
    y_errs = np.zeros(shape = np.shape(job_specs))
    x_vals = np.zeros(shape = np.shape(job_specs))
    x_errs = np.zeros(shape = np.shape(job_specs))

    for i, tr_size in enumerate(train_sizes):
        for j, job_spec in enumerate(job_specs):
            if metric in ('g', 's', 'd'):
                #TODO: ADD A SQRT(N) CURVE TO G. BUT NEED TO PLOT VS ACTUAL N FIRST RATHER THAN VS #EVENTS
                data_object = 'kernel'
            elif metric in ('acc', 'prec', 'rec', 'f1'): #TODO: ADD ROC
                data_object = 'preds'
            m_y, std_y, m_x, std_x = compute_avg_metrics(data_object, job_specs[j][i])
            ###TODO: HERE NEEDS FILLING
            y_vals[j][i] = m_y[metric]
            y_errs[j][i] = std_y[metric]
            #this is a placeholder, m_x and std_x should be passed here
            #they need to be from the lengths of the TRAINING data not TEST
            x_vals[j][i] = m_x
            x_errs[j][i] = std_x
    print(y_vals, y_errs)
    for i, job_spec in enumerate(job_specs):
        #don't plot points if they have not been trained [metric==0]
        nonzero_ids = [tr_id for tr_id in range(len(y_vals[i])) if y_vals[i][tr_id] !=0]
        y_val_to_plot = [y_vals[i][tr_id] for tr_id in range(len(y_vals[i])) if tr_id in nonzero_ids]
        y_err_to_plot = [y_errs[i][tr_id] for tr_id in range(len(y_errs[i])) if tr_id in nonzero_ids]
        x_val_to_plot = [x_vals[i][tr_id] for tr_id in range(len(y_vals[i])) if tr_id in nonzero_ids]
        x_err_to_plot = [x_errs[i][tr_id] for tr_id in range(len(y_errs[i])) if tr_id in nonzero_ids]
        canvas.errorbar(x_val_to_plot, y_val_to_plot, xerr = x_err_to_plot, yerr=y_err_to_plot, label = legend[i], linestyle = '-', markersize = 5)
        canvas.legend()
        canvas.set_xlabel('Train size [number of tracklets]')
        canvas.set_ylabel(metric)


def compare_datasets(plotting_func, metric, tracklet_type_l, train_sizes_l, test_size_l, kernel_specs, legend):
    """
    Same plot for different data sets (hence different tracklet objects)
    arguments with _l indicate a list where items depend on datasets
    """
    fig, axs = plt.subplots(1, len(tracklet_type_l))
    for canvas, tr_sizes, te_size, tracklet_type in zip(axs, train_sizes_l, test_size_l, tracklet_type_l):
        plotting_func(canvas, metric, tr_sizes, te_size, kernel_specs, tracklet_type, legend)











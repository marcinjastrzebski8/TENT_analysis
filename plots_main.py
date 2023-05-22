from plots import metric_vs_train_size, compare_datasets
import matplotlib.pyplot as plt
import argparse

def make_specs_for_metric_vs_train_size(metric, tracklet_type, te_size):
    tr_sizes = [1,2,3,4,5,6,7,8,9,10,20,30,40,50]

    #not the most compact ways of encoding this info but to me decently readable
    fid_specs = [{'tracklet_type': tracklet_type, 'kernel_type': 'fidelity', 'alpha': '0p2', 'C': '1000000p0', 'pauli_list': 'Z_YY', 'te_size': te_size, 'tr_size': tr_size} for tr_size in tr_sizes]
    proj_specs = [{'tracklet_type': tracklet_type, 'kernel_type': 'projected_1rdm', 'alpha': '0p2', 'C': '1000000p0', 'pauli_list': 'Z_YY', 'te_size': te_size, 'tr_size': tr_size} for tr_size in tr_sizes]
    class_specs = [{'tracklet_type': tracklet_type, 'kernel_type': 'classical_keep_kernel', 'C': '1000000p0', 'gamma': '1', 'te_size': te_size,'tr_size': tr_size} for tr_size in tr_sizes]
    
    if metric in ('g'):
        specs = [fid_specs, proj_specs]
        labels = ['Fidelity', 'Projected 1 RDM']
    elif metric in ('acc', 'prec', 'rec', 'f1', 's', 'd', 's_gen'):
        specs = [fid_specs, proj_specs, class_specs]
        labels = ['Fidelity', 'Projected 1 RDM', 'Classical']
    else:
        raise ValueError('wrong metric provided')
    return specs, labels, tr_sizes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Plot a given metric vs train size. Mostly hard-coded atm.')
    parser.add_argument('metric', type = str, help = 'Which metric to plot')
    parser.add_argument('tracklet_type', type = str, help = 'Which tracklet type to plot.')
    parser.add_argument('te_size', type = int, help = 'How many events were the results tested on')
    args = parser.parse_args()

    specs, labels, tr_sizes = make_specs_for_metric_vs_train_size(args.metric, args.tracklet_type, args.te_size)

    fig, ax = plt.subplots()
    metric_vs_train_size(ax, args.metric, tr_sizes, specs, labels)

    plt.savefig(f'{args.metric}_vs_train_size_{args.tracklet_type}_te_{args.te_size}', dpi = 300)


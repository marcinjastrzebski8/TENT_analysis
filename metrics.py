from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
from copy import deepcopy

from quask_metrics import calculate_approximate_dimension, calculate_geometric_difference, calculate_model_complexity, calculate_model_complexity_generalized


class Data_Object(ABC):
    """
    A Data_Object's defining attribute is a kernel (np.array) or predictions (pd.DataFrame).
    The metrics which can be defined for Data_Objects form the basis of the analysis.

    The goal is to extract all the possible data metrics once an object has been defined.
    """
    def __init__(self, job_specs):
        self.job_specs = job_specs
        self.tracklet_type = self.job_specs['tracklet_type']
        self.tr_size = job_specs['tr_size']
        self.te_size = job_specs['te_size']

        self.kernel_type = job_specs['kernel_type']
        self.C = job_specs['C']
        #if job was using classical kernel
        try:
            self.gamma = job_specs['gamma']
        except:
            self.gamma = None
        #if job was using quantum kernel
        try:
            self.alpha = job_specs['alpha']
            self.pauli_list = job_specs['pauli_list']
        except:
            self.alpha = None
            self.pauli_list = None
        #check if the spec list has all kernel-related info
        if all(mem_var is None for mem_var in [self.alpha, self.pauli_list, self.gamma]):
            raise ValueError('The spec list is missing kernel-related keys to define a run fully\n;',
                             'gamma for classical kernels\n',
                             'alpha and pauli_list for quantum kernels')
       
    @abstractmethod
    def load(self, reg_id):
        """
        setter for the main attribute of the object
        """
        pass
    
    def load_tr_labels(self, reg_id):
        saved_labels_dir = str(Path().absolute() / 'saved_tr_labels')
        labels_file = 'train_labels_'+self.tracklet_type+'_tr_'+str(self.tr_size)+'_reg_'+str(reg_id)+'_in_new_phi.npy'
        labels = np.load(saved_labels_dir+'/'+labels_file) 

        self.labels = labels
        return self.labels



class Data_Kernel(Data_Object):
    """
    Data_Kernel's main attribute is a numpy array
    """
    def __init__(self, job_specs):
        super().__init__(job_specs)
        self.reg_id = self.job_specs['reg_id']

    def load(self, reg_id):
        kernel_dir = str(Path().absolute() / 'saved_kernels')

        kernel_name = 'kernel_auto_config_'+self.tracklet_type+'_'+self.kernel_type
        #this functionality might not be necessary, most likely will only be passing various quantum kernels
        if 'classical' in self.kernel_type:
            kernel_name += '_C_'+self.C+'_gamma_'+self.gamma
        else:
            kernel_name += '_alpha_'+self.alpha+'_C_'+self.C+'_'+self.pauli_list

        #given file might not exist if loading various results in a loop, shouldnt cause the analysis to stop
        try:
            kernel_name += '_div_new_phi_reg_'+str(reg_id)+'_tr_'+str(self.tr_size)+'_te_'+str(self.te_size)+'.npy'
            kernel_to_load = kernel_dir+'/'+kernel_name
            kernel = np.load(kernel_to_load)
        except:
            ('kernel at ', kernel_to_load, ' was not found')
            kernel = np.zeros(1) #placeholder

        self.kernel = kernel
        return self.kernel
    
    def find_analogous_classic(self, C_class = '1000000p0', gamma = '1'):
        """
        For a quantum kernel, recognise the classical kernel to compare it to.
        Used for calculating geometric distance
        """
        class_specs = {'kernel_type': 'classical_keep_kernel', 
                       'tracklet_type': self.tracklet_type,
                       'tr_size': self.tr_size,
                       'te_size': self.te_size,
                       'C': C_class,
                       'gamma': gamma,
                       'reg_id': self.reg_id}
        return Data_Kernel(class_specs)
    
    def compute_metric(self, metric_type):
        #forgot why I initialise this but its in case kernels dont load or smth
        metric = 0
        if metric_type == 'd':
            #d can be found as long as a kernel exists
            try:
                metric = calculate_approximate_dimension(self.kernel)
            except:
                print(f'd couldnt be computed for {self.kernel_type} because the relevant kernel was not found')
        elif metric_type == 's':
            #s requires the train labels file to exist
            try:
                labels = self.load_tr_labels(self.reg_id)
                metric = calculate_model_complexity(self.kernel, labels)
            except:
                print(f's couldnt be computed for {self.kernel_type} because the relevant train labels were not found')
        elif metric_type == 's_gen':
            #s requires the train labels file to exist
            try:
                labels = self.load_tr_labels(self.reg_id)
                metric = calculate_model_complexity_generalized(self.kernel, labels)
            except:
                print(f's_gen couldnt be computed for {self.kernel_type} because the relevant train labels were not found')
        elif metric_type == 'g':
            #g requires an analogous classical kernel to exist  
            try:
                analogous_class = self.find_analogous_classic()
                analogous_class.load(self.reg_id)
                other_kernel = analogous_class.kernel
                metric = calculate_geometric_difference(self.kernel, other_kernel)
            except:
                print(f'g couldnt be computed for {self.kernel_type} because the analogous classical kernel was not found')
        return metric

        

class Data_Preds(Data_Object):
    """
    Data_Preds's main attribute is a dataframe
    """
    def __init__(self, job_specs, data_frame: pd.DataFrame = None):
        if data_frame == None:
            """
            Load the dataframe
            """
            super().__init__(job_specs)
            #set columns for the datafeame
            #NOTE: the columns will be updated after the decision boundary distance patch
            if self.tracklet_type == 'edge':
                self.cols = ['tracklet', 'tracklet_coords', 'label', 'eta', 'phi', 
                'true_pt', 'layer','track_length', 'hit1_id', 'hit2_id', 'particle_num', 'prediction', 'decision']
            else:
                self.cols = ['tracklet', 'tracklet_coords','label', 'phi_breaking','theta_breaking','pt_estimate', 
                'ip','pt','layers', 'track_length','particle_id', 'particle_n', 'phi', 'eta', 'prediction', 'decision']
            self.reg_id = self.job_specs['reg_id']
        else:
            """
            Initialise directly from an existing dataframe
            """
            self.df = data_frame
        

    def load(self, reg_id):
        preds_dir = str(Path().absolute() / 'predictions')
        preds_file = self.kernel_type+'_'+self.tracklet_type+'_predictions_'+str(self.tr_size)+'_'+str(self.te_size)+'_events_reg_'+str(reg_id)+'_in_new_phi.npy'
        preds_to_load = preds_dir+'/'+preds_file
        #given file might not have been created, analysis shouldnt stop
        try:
            preds_df = pd.DataFrame(np.load(preds_to_load, allow_pickle = True), columns = self.cols)
        except:
            print('predictions at ', preds_to_load, ' were not found')
            preds_df = pd.DataFrame(np.zeros((1, len(self.cols))), columns = self.cols) #placeholder


        self.df = preds_df
        return self.df
    

    def compute_metric(self, metric_type):

        def decide_conf_matrix(x):
            """
            Helper function
            label events as TN, TP, FN, FP according to their predictions and true labels
            aka decide the confusion matrix label
            """
            if x[0] == 0.0:
                if x[1] == 0.0:
                    return 'TN'
                if x[1] == 1.0:
                    return 'FP'
            if x[0] == 1.0:
                if x[1] == 1.0:
                    return 'TP'
                if x[1] == 0.0:
                    return 'FN'
                
        self.df['conf_matrix'] = self.df[['label', 'prediction']].apply(lambda x: decide_conf_matrix(x), axis = 1)

        TP = len(self.df[self.df['conf_matrix'] == 'TP'])
        TN = len(self.df[self.df['conf_matrix'] == 'TN'])
        FP = len(self.df[self.df['conf_matrix'] == 'FP'])
        FN = len(self.df[self.df['conf_matrix'] == 'FN'])
        if (TP+FP == 0) or (TP+FN == 0):
            acc = prec = rec = f1 = 0
        else:
            acc = (TP + TN)/(TP+TN+FP+FN)
            prec = (TP)/(TP+FP)
            rec = (TP)/(TP+FN)
            f1 = (2*prec*rec)/(prec+rec)
        #These will be enabled once the distance column is added to TENT
        #fpr, tpr, _ = roc_curve(results_df['a2'].tolist(), results_df['a12'].tolist())
        #roc_auc = auc(fpr, tpr)

        #feels a bit silly only saving one at a time but that flows better with the rest of the code
        if metric_type == 'acc':
            metric = acc
        elif metric_type == 'prec':
            metric = prec
        elif metric_type == 'rec':
            metric = prec
        elif metric_type == 'f1':
            metric = f1
        else:
            raise ('wrong metric type passed')        
    
        return metric

def make_name_from_job_specs(job_specs, metric_type) -> str:
    kernel_type = job_specs['kernel_type']
    tracklet_type = job_specs['tracklet_type']
    tr_size = job_specs['tr_size']
    te_size = job_specs['te_size']


    name = f'{metric_type}_{kernel_type}_{tracklet_type}_tr_{tr_size}_te_{te_size}'
    return name


def compute_avg_metrics(data_object_type, job_specs: Dict, err_type = 'std', metric_type: str = None) -> Tuple[Dict]:
    """
    Load kernels from all detector regions and find mean and std of the metrics.
    """
    metric_list = []
    data_len_list = []
    for reg_id in range(16):
        reg_job_specs = deepcopy(job_specs)
        reg_job_specs['reg_id'] = str(reg_id)
        if data_object_type == 'kernel':
            data_object = Data_Kernel(reg_job_specs)
        elif data_object_type == 'preds':
            data_object = Data_Preds(job_specs = reg_job_specs)
        else:
            raise ValueError('data_object_type should be: \'kernel\' or \'preds\'')
        data_object.load(reg_id)

        #it can take a long time to compute all the metrics for kernels so it can be useful to do one at a time
        #hence different implementations
        metric = data_object.compute_metric(metric_type = metric_type)
        metric_list.append(metric)

        data_len = len(data_object.load_tr_labels(reg_id))
        data_len_list.append(data_len)


    print('metric_list:', metric_list)
    mean_metrics = np.mean(metric_list)
    if err_type == 'std':
        err_metrics = np.std(metric_list)
    elif err_type == 'bayes_conf_int':
        #TODO: Add functionality to implement the bayesian confidence interval as error
        #should only be an option if working with predictions
        pass 

    mean_len = np.mean(data_len_list)
    std_len = np.std(data_len_list)

    #save the average metrics so they can be accessed for plotting later
    #NOTE: the whole analysis pipeline will benefit from implementing this
    #but it will require some changes to the way things are plotted
    #workflow could be: get results -> obtain all metrics needed -> start plotting
    #instead of: get results -> plot whilst getting metrics one at a time
    name_from_config = make_name_from_job_specs(job_specs, metric_type)

    np.save(f'metrics_from_results/mean_metrics_{name_from_config}', mean_metrics)
    np.save(f'metrics_from_results/err_metrics_{name_from_config}', err_metrics)
    np.save(f'metrics_from_results/mean_data_len_{name_from_config}', mean_len)
    np.save(f'metrics_from_results/std_data_len_{name_from_config}', std_len)

    return (mean_metrics, err_metrics, mean_len, std_len)

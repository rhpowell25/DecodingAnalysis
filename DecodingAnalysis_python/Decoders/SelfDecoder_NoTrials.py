
#%% Import packages
from scipy import stats
import numpy as np
from wiener_filter import format_data, train_wiener_filter, test_wiener_filter
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

class Self_Decoder_No_Trials():
    
    def __init__(self, xds, predict_what, norm_signal, shuffle_file):
        
        #%% Some of the decoding specifications
        
        # Unit quality ('All' vs. 'Stable')
        unit_quality = 'All'
        
        # Do you want to smooth the spikes ('Yes', 'No')
        smooth_units = 'No'
        
        plot_scatter = 0
  
        # How many cross validation folds?
        cv_folds = 4

        #%% Use either stable units or all units for the predictions
        
        if unit_quality == 'Stable':
            
            from Load_Excel import Load_Excel
            
            # Date
            file_name = xds.file_name
            Date = file_name.split('_', 1)[0]
            
            # Task
            if xds._lab_data__meta['task_name'] == 'multi_gadget':
                Task = 'PG'
            else:
                Task = xds._lab_data__meta['task_name']
                
            # Monkey
            Monkey = xds._lab_data__meta['monkey_name']
            
            xds_excel = Load_Excel(Date, Monkey, Task, 'Max')
            
            # Find the indices of the stable units in the excel
            insig_p_value_idx = np.asarray(np.where(xds_excel.nonlin_p_value > 0.05)).reshape(-1).T
            
            # Find the names of the stable units
            unit_names = xds_excel.unit_names[insig_p_value_idx].values
            
            # Find the indices of the stable units in xds
            unit_idxs = np.zeros(len(unit_names), dtype = 'int')
            for ii in range(len(unit_names)):
                unit_idxs[ii] = xds.unit_names.index(unit_names[ii])

            # Exclude the unstable units
            print('Excluding unstable units')
            spikes = [xds.spikes[ii] for ii in unit_idxs]
            
        elif unit_quality == 'All':
            spikes = xds.spikes
        
        #%% Binning the inputs & outputs
        
        binned_timeframe = xds.joint_angle_time_frame # Keeps binning consistent
        bin_size = (binned_timeframe[-1] - binned_timeframe[0]) / len(binned_timeframe)
        
        # If you're predicting joint angles
        if predict_what == 'Joint_Angles':
        
            # Update joint angles 
            binned_Behavior = xds.joint_angles
            self.behavior_titles = xds.joint_names
            
        # If you're predicting joint angles
        if predict_what == 'Joint Velocity':
        
            # Extract joint angles & names
            joint_angles = xds.joint_angles
            joint_names = xds.joint_names
            
            # Calculate the velocity
            binned_Behavior = np.diff(joint_angles, axis = 0) / bin_size
            
            binned_timeframe = binned_timeframe[:-1]

            self.behavior_titles = [[] for ii in range(len(joint_names))]
            for ii in range(len(joint_names)):
                self.behavior_titles[ii] = joint_names[ii] + '_velocity'
            
        # If you're predicting EMG
        if predict_what == 'EMG':
            
            if len(xds.EMG) != len(xds.joint_angle_time_frame):
                # Re-bin the EMG
                L = len(binned_timeframe)
                n = np.floor(xds.EMG.shape[0])/L
                #n = bin_size/xds.bin_width
                #L = int(np.floor(xds.EMG.shape[0]/n))
                idx = [int(np.floor(ii*n)) for ii in range(1, L)]
                binned_Behavior = xds.EMG[idx, :]
            else:
                binned_Behavior = xds.EMG
            
            self.behavior_titles = xds.EMG_names
            
        # Re-bin the spikes
        if len(xds.time_frame) == len(xds.spike_counts):
            spike_counts = []         
            for each in spikes:
                bb = each.reshape((len(each),))
                out, _ = np.histogram(bb, binned_timeframe)
                spike_counts.append(out)
            binned_spikes = np.asarray(spike_counts).T
        elif len(xds.joint_angle_time_frame) == len(xds.spike_counts):
            binned_spikes = xds.spike_counts
            
        #%% Smooth the binned spikes if selected 
        
        if smooth_units == 'Yes':
            smooth_size = 0.05 # Change the smooth window size here
            
            binned_spike_counts = binned_spikes.T.tolist()
            smoothed = []
            kernel_hl = 3 * int(smooth_size / bin_size)
            normalDistribution = stats.norm(0, smooth_size)
            x = np.arange(-kernel_hl*bin_size, (kernel_hl+1)*bin_size, bin_size)
            kernel = normalDistribution.pdf(x)
            n_sample = np.size(binned_spike_counts[0])
            nm = np.convolve(kernel, np.ones((n_sample))).T[int(kernel_hl):n_sample + int(kernel_hl)] 
            for each in binned_spike_counts:
                temp1 = np.convolve(kernel,each)
                temp2 = temp1[int(kernel_hl):n_sample + int(kernel_hl)]/nm
                smoothed.append(temp2)
            print('The spike counts have been smoothed.')
            binned_spikes = np.asarray(smoothed).T
                   
        #%% Processing the output signal if selected
        
        if norm_signal == 'Yes':

            if predict_what == 'EMG':
                
                # Zero the EMG
                from SingleSessionEMGZero import SingleSession_EMGZero
                EMG_Zero_Factor = SingleSession_EMGZero.Single_Session_EMG_Zero(xds, 'All', 'Percentile', 1)
                zeroed_EMG = binned_Behavior - EMG_Zero_Factor
                            
                # Normalize the EMG
                from SingleSessionNormalizeEMG import SingleSession_NormalizeEMG
                EMG_Norm_Factor = SingleSession_NormalizeEMG.Single_Session_Normalize_EMG(xds, 'All', 99, 1)
                binned_Behavior = zeroed_EMG / EMG_Norm_Factor*100
                
            if predict_what == 'Joint_Angles':
                
                for ii in range(len(xds.joint_names)):
                    
                    # Zero the joint angles
                    min_param = min(binned_Behavior[:,ii])
                    if min_param < 0:
                        binned_Behavior[:,ii] = binned_Behavior[:,ii] + abs(min_param)
                    else:
                        binned_Behavior[:,ii] = binned_Behavior[:,ii] - min_param
                    
                    # Normalize the joint angles
                    max_param = np.percentile(binned_Behavior[:,ii], 99)
                    binned_Behavior[:,ii] = binned_Behavior[:,ii] / max_param * 100
       
        #%% Scatter plot the spikes & behavior
        
        if plot_scatter == 1:
            unit_name = 'elec25_1'
            N = xds.unit_names.index(unit_name)
            scatter_spikes = xds.spikes[N].reshape(-1)
            
            scatter_bin_size = 0.1
            scatter_n_bins = round(binned_timeframe[-1] / scatter_bin_size)
            #scatter_binned_timeframe = np.linspace(binned_timeframe[0],binned_timeframe[-1], scatter_n_bins)
            
            # Re-bin the spikes
            scatter_spikes, bin_edges = \
                np.histogram(scatter_spikes, bins = scatter_n_bins, density = False)
            scatter_firing_rate = scatter_spikes / scatter_bin_size
                
            joint_name = 'Index_DIP_Flex_Ext'
            joint_idx = xds.joint_names.index(joint_name)
            
            # Re-bin the joint angles
            n = scatter_bin_size/bin_size
            L = int(np.floor(xds.joint_angles.shape[0]/n))
            idx = [int(np.floor(ii*n)) for ii in range(1, L - 1)]
            scatter_joint_angles = xds.joint_angles[idx, joint_idx]
            
            import matplotlib.pyplot as plt
            from Plot_Specs import Font_Specs
            
            # Font & plotting specifications
            font_specs = Font_Specs()
            
            scatter_offset = len(scatter_firing_rate) - len(scatter_joint_angles)
            # Plot the scatter
            fig, fig_axes = plt.subplots()
            trial_time = np.arange(len(scatter_firing_rate))*scatter_bin_size
            plt.scatter(scatter_firing_rate[:- scatter_offset], scatter_joint_angles, \
                        c = trial_time[:- scatter_offset], cmap='plasma_r')
             
            # Add a color bar
            color_bar = plt.colorbar()
            #color_bar.ax.set_yticklabels(['-1', '-0.5', 'Trial Go Cue', '0.5', '1', '1.5', '2', '2.5'])
            color_bar.set_label('Time (sec.)', fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
        
            # Title the scatter
            title_string = joint_name + ' vs. ' + unit_name
            plt.title(title_string, fontname = font_specs.font_name, \
                      fontsize = font_specs.title_font_size, fontweight = 'bold')
                
            # Axis Labels
            plt.xlabel(unit_name, fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
            plt.ylabel(joint_name, fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
            
        #%% K-fold CV
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=cv_folds, random_state = None, shuffle = shuffle_file)
        kf.get_n_splits(binned_spikes)
        
        #%% Initialize the empty variables
        training_spikes, training_Behavior = [[] for i in range(cv_folds)], [[] for i in range(cv_folds)]
        test_spikes, self.test_Behavior = [[] for i in range(cv_folds)], [[] for i in range(cv_folds)]
        
        formatted_training_spikes, formatted_training_Behavior = [[] for i in range(cv_folds)], [[] for i in range(cv_folds)]
        formatted_test_spikes, formatted_test_Behavior = [[] for i in range(cv_folds)], [[] for i in range(cv_folds)]
        
        self.H_baseline = [[] for i in range(cv_folds)]
        
        self.r2_values_train = np.zeros([binned_Behavior.shape[1], cv_folds])
        self.r2_values_test = np.zeros([binned_Behavior.shape[1], cv_folds])
        self.vaf_values_train = np.zeros([binned_Behavior.shape[1], cv_folds])
        self.vaf_values_test = np.zeros([binned_Behavior.shape[1], cv_folds])
        
        predicted_Behavior_train, self.predicted_Behavior_test = [[] for i in range(cv_folds)], [[] for i in range(cv_folds)]
        
        # Split the data according to their folds
        cv = 0
        for train_index, test_index in kf.split(binned_spikes):
            print("TRAIN:", train_index, "TEST:", test_index)
            training_spikes[cv], test_spikes[cv] = binned_spikes[train_index], binned_spikes[test_index]
            training_Behavior[cv], self.test_Behavior[cv] = binned_Behavior[train_index], binned_Behavior[test_index]
            cv += 1
        
        #%% Build a decoder for each fold
        n_lags = 5
        
        # Formatting the spikes & behavior as numpy arrays
        for cv in range(cv_folds):
            formatted_training_spikes[cv], formatted_training_Behavior[cv] = \
                format_data(training_spikes[cv], training_Behavior[cv], n_lags)
            formatted_test_spikes[cv], formatted_test_Behavior[cv] = \
                format_data(test_spikes[cv], self.test_Behavior[cv], n_lags)
        
        # Training the decoder
        for cv in range(cv_folds):
            self.H_baseline[cv] = train_wiener_filter(formatted_training_spikes[cv], formatted_training_Behavior[cv], 1)
        
            # Testing the trained decoder on testing data
            predicted_Behavior_train[cv] = test_wiener_filter(formatted_training_spikes[cv], self.H_baseline[cv])
            self.predicted_Behavior_test[cv] = test_wiener_filter(formatted_test_spikes[cv], self.H_baseline[cv])
        
        # Save the accuracy of the decoder
        for cv in range(cv_folds):
            self.r2_values_train[:,cv] = r2_score(formatted_training_Behavior[cv], \
                   predicted_Behavior_train[cv], multioutput = 'raw_values')
            self.r2_values_test[:,cv] = r2_score(formatted_test_Behavior[cv], \
                  self.predicted_Behavior_test[cv], multioutput = 'raw_values')
            self.vaf_values_train[:,cv] = explained_variance_score(formatted_training_Behavior[cv], \
                    predicted_Behavior_train[cv], multioutput = 'raw_values')
            self.vaf_values_test[:,cv] = explained_variance_score(formatted_test_Behavior[cv], \
                    self.predicted_Behavior_test[cv], multioutput = 'raw_values')
        
            print(f"End of training fold {cv+1}\n")
        
        











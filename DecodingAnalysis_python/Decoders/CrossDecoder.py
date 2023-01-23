
#%% Import basic packages
from scipy import stats
import numpy as np
import math
from wiener_filter import format_data_from_trials, test_wiener_filter
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

class Cross_Decoder():
    
    def __init__(self, Decoder_Vars, xds, predict_what, Zero_Factor, Norm_Factor):
        
        #%% Some of the decoding specifications
        
        # Bin size
        bin_size = 0.025
        
        # Do you want to smooth the spikes ('Yes', 'No')
        smooth_units = 'No'
 
        H_baseline = Decoder_Vars.H_baseline
        
        # How many time lags for the decoder?
        n_lags = Decoder_Vars.n_lags
        
        # How many cross validation folds?
        cv_folds = 4
        
        # Select the event to align to:
        event = 'task_onset'
        
        # Do you want the shuffled file? (True, False)
        shuffle_file = False
        
        # Save the target hold time
        TgtHold = xds._lab_data__meta['TgtHold']
        
        #%% Select which units to use for the decoder
            
        from Find_Excel import Find_Excel
        xds_excel = Find_Excel(xds)
        
        Sampling_Params = {}
        # Unit quality ('All' vs. 'Stable')
        Sampling_Params['unit_quality'] = 'Stable'
        # ISI quality ('All' vs. 'Single')
        Sampling_Params['ISI_quality'] = 'All'
        # What change in depth of modulation do you want to observe (# vs NaN)
        Sampling_Params['depth_change'] = np.nan
        # What preferred direction do you want to plot(-90, 0, 90, 180, 'All')
        Sampling_Params['pref_dir'] = 'All'
        # What minimum depth of modulation do you want to observe (# vs. NaN)
        Sampling_Params['depth_min'] = np.nan
        # Unit modulation significance ('All' vs. 'Sig')
        Sampling_Params['depth_sig'] = 'All'
        
        from Subsample_Excel import Subsample_Excel
        xds_excel = Subsample_Excel(xds_excel, Sampling_Params)
        
        # Find the names of the units
        unit_names = xds_excel.unit_names.values
        
        # Find the indices of the units in xds
        unit_idxs = np.zeros(len(unit_names), dtype = 'int')
        for ii in range(len(unit_names)):
            unit_idxs[ii] = xds.unit_names.index(unit_names[ii])
            
        spikes = [xds.spikes[ii] for ii in unit_idxs]
        
        #%% Binning the inputs & outputs
        
        binned_timeframe = np.arange(xds.time_frame[0] - bin_size/2, 
                                     xds.time_frame[-1] + bin_size/2, bin_size)
        binned_timeframe = binned_timeframe.reshape(len(binned_timeframe))
        
        # if you're predicting Force
        if predict_what == 'Force':
            n = bin_size/xds.bin_width
            L = int(np.floor(np.size(xds.force, 0)/n))
            idx = [int(np.floor(i*n)) for i in range(1, L)]
            binned_Force = xds.force[idx, :]
            # Sum the two force transducers
            binned_Behavior = (binned_Force[:,0] + binned_Force[:,1]).reshape(-1,1)
            
            behavior_titles = [predict_what]
            
        # if you're predicting cursor position
        if predict_what == 'Cursor':
            n = bin_size/xds.bin_width
            L = int(np.floor(xds.curs_p.shape[0]/n))
            idx = [int(np.floor(i*n)) for i in range(1, L)]
            binned_curs_p = xds.curs_p[idx, :]
            # Vector sum of the two cursor positions
            binned_Behavior = np.zeros(len(binned_curs_p))
            for dd in range(len(binned_Behavior)):
                binned_Behavior[dd] = math.sqrt(binned_curs_p[dd][0]**2 + binned_curs_p[dd][1]**2)
            # Normalize the cursor position
            binned_Behavior = binned_Behavior / Norm_Factor*100
            binned_Behavior = binned_Behavior.reshape(-1,1)
            
            behavior_titles = [predict_what]
            
        # If you're predicting EMG
        if predict_what == 'EMG':
            # Re-bin the EMG
            L = len(binned_timeframe)
            n = np.floor(xds.EMG.shape[0])/L
            idx = [int(np.floor(ii*n)) for ii in range(1, L)]
            binned_EMG = xds.EMG[idx, :]
            # Zero the EMG
            zeroed_EMG = binned_EMG - Zero_Factor        
            # Normalize the EMG
            binned_Behavior = zeroed_EMG / Norm_Factor*100
            
            behavior_titles = xds.EMG_names
            
        # Re-bin the spikes
        spike_counts = []         
        for each in spikes:
            bb = each.reshape((len(each),))
            out, _ = np.histogram(bb, binned_timeframe)
            spike_counts.append(out)
        binned_spikes = np.asarray(spike_counts).T
        
        #%% Smooth the binned spikes if selected 
        
        if smooth_units == 'Yes':
            smooth_size = 2*bin_size # Change the smooth window size here
            
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
            
        #%% Sort the spikes & EMG or Force
        
        from EventAlignmentTimes import EventAlignmentTimes
        from GoCueAlignmentTimes import GoCueAlignmentTimes
        from TrialEndAlignmentTimes import TrialEndAlignmentTimes
        
        # Times for rewarded trials
        target_dir = 'NaN'
        target_center = 'Max'
        Alignment_Times = EventAlignmentTimes(xds, target_dir, target_center, event)
        rewarded_gocue_time = GoCueAlignmentTimes(xds, target_dir, target_center)
        rewarded_end_time = TrialEndAlignmentTimes(xds, target_dir, target_center)
        
        sorted_spikes = [[] for ii in range(len(Alignment_Times))]
        sorted_Behavior = [[] for ii in range(len(Alignment_Times))]
        for jj in range(len(Alignment_Times)):
            bs_idxs = np.argwhere(np.logical_and(binned_timeframe.reshape(-1,) >= rewarded_gocue_time[jj] - 0.4, \
                                              binned_timeframe.reshape(-1,) <= rewarded_gocue_time[jj])).reshape(-1,)
            mp_idxs = np.argwhere(np.logical_and(binned_timeframe.reshape(-1,) >= Alignment_Times[jj], \
                                              binned_timeframe.reshape(-1,) <= rewarded_end_time[jj])).reshape(-1,)
            # Spikes
            sorted_spikes[jj] = binned_spikes[np.concatenate((bs_idxs, mp_idxs)),:]
            # Behavior
            sorted_Behavior[jj] = binned_Behavior[np.concatenate((bs_idxs, mp_idxs)), :]
            
        #%% K-fold CV
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=cv_folds, random_state = None, shuffle = shuffle_file)
        kf.get_n_splits(sorted_spikes)
        
        #%% Initialize the empty variables
        
        test_spikes, test_Behavior = [[] for ii in range(cv_folds)], [[] for ii in range(cv_folds)]
        
        formatted_test_spikes = [[] for ii in range(cv_folds)]
        formatted_test_Behavior = [[] for ii in range(cv_folds)]
        
        test_indices = [[] for ii in range(cv_folds)]
        
        per_trial_testing_Behavior = [[] for ii in range(cv_folds)]
        per_trial_predicted_Behavior = [[] for ii in range(cv_folds)]
        
        r2_values_test = np.zeros([sorted_Behavior[0].shape[1], cv_folds])
        vaf_values_test = np.zeros([sorted_Behavior[0].shape[1], cv_folds])
        
        predicted_Behavior_test = [[] for ii in range(cv_folds)]
        
        #%% Split the data according to their folds
        cv = 0
        trial_lim = math.floor(len(sorted_Behavior) / cv_folds) * cv_folds
        for train_index, test_index in kf.split(np.arange(0, trial_lim)):
            print("TEST:", test_index)
            test_spikes[cv] = [x for ii, x in enumerate(sorted_spikes) if ii in test_index]
            test_Behavior[cv] = [x for ii, x in enumerate(sorted_Behavior) if ii in test_index]
            test_indices[cv] = test_index
            cv += 1
            
        #%% Test the decoder on each fold
        
        # Formatting the spikes & behavior as numpy arrays
        for cv in range(cv_folds):
            formatted_test_spikes[cv], formatted_test_Behavior[cv] = \
                format_data_from_trials(test_spikes[cv], test_Behavior[cv], n_lags)
        
        # Testing the trained decoder on testing data
        for cv in range(cv_folds):
            predicted_Behavior_test[cv] = test_wiener_filter(formatted_test_spikes[cv], H_baseline[0])
        
        # Save the accuracy of the decoder
        for cv in range(cv_folds):
            r2_values_test[:,cv] = r2_score(formatted_test_Behavior[cv], \
                  predicted_Behavior_test[cv], multioutput = 'raw_values')
            vaf_values_test[:,cv] = explained_variance_score(formatted_test_Behavior[cv], \
                    predicted_Behavior_test[cv], multioutput = 'raw_values')
        print(f"End of testing fold {cv+1}\n")
            
        #%% Seperating & testing by trial
        
        for cv in range(cv_folds):
            for each in zip(test_spikes[cv], test_Behavior[cv]):
                a_, b_ = format_data_from_trials(each[0], each[1], n_lags)
                b_pred = test_wiener_filter(a_, H_baseline[0])
                per_trial_testing_Behavior[cv].append(b_)
                per_trial_predicted_Behavior[cv].append(b_pred)  

        #%% Save the necessary variables
        
        self.n_lags = n_lags
        self.TgtHold = TgtHold
        self.binned_timeframe = binned_timeframe
        self.behavior_titles = behavior_titles
        self.test_indices = test_indices
        self.formatted_test_Behavior = formatted_test_Behavior
        self.per_trial_testing_Behavior = per_trial_testing_Behavior
        self.per_trial_predicted_Behavior = per_trial_predicted_Behavior
        self.r2_values_test = r2_values_test
        self.vaf_values_test = vaf_values_test
        self.predicted_Behavior_test = predicted_Behavior_test






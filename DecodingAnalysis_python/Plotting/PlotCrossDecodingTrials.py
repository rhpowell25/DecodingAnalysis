
#%% Import basic packages
import matplotlib.pyplot as plt
from Plot_Specs import Font_Specs
import numpy as np
import matplotlib.font_manager as fm

def Plot_CrossDecodingTrials(Within_Decoder_Vars, Cross_Decoder_Vars, Save_Figs):
    
    #%% Extracting the variables
    
    # Font & plotting specifications
    font_specs = Font_Specs()
    
    # Do you want to manually set the y-axis?
    man_y_axis = 'No'
    #man_y_axis = [-20, 25]
    
    behavior_titles = Within_Decoder_Vars.behavior_titles
    
    # Convert the force millivolts to cursor position
    if behavior_titles == ['Force']:
        force_convert = 1000
        force_gain = 5
    else:
        force_convert = 1
        force_gain = 1
    
    # Decoder accuracy
    try:
        Within_r2_values = Within_Decoder_Vars.r2_values_test
    except:
           Within_r2_values = Within_Decoder_Vars.r2_values_train
    Cross_r2_values = Cross_Decoder_Vars.r2_values_test
    
    test_Behavior_timeframe = Within_Decoder_Vars.binned_timeframe
    
    # Behaviors
    #per_trial_Within_predictions = Within_Decoder_Vars.per_trial_predicted_Behavior[0]
    Within_predicted_Behavior = Within_Decoder_Vars.predicted_Behavior_test
    Cross_predicted_Behavior = Cross_Decoder_Vars.predicted_Behavior_test
    Within_test_Behavior = Within_Decoder_Vars.formatted_test_Behavior
    
    test_idxs = Within_Decoder_Vars.test_indices[0]
    # Extract the corresponding cross predictions
    per_trial_Cross_predictions = [Cross_Decoder_Vars.per_trial_predicted_Behavior[0][jj] for jj in test_idxs]
        
    for jj in range(len(per_trial_Cross_predictions)):
        if jj == 0:
            Cross_predicted_Behavior = per_trial_Cross_predictions[jj]
        else:
            Cross_predicted_Behavior = np.vstack((Cross_predicted_Behavior, per_trial_Cross_predictions[jj]))
    
    #%% Plotting the concatenated predicted behavior
    
    for ii in range(len(behavior_titles)):
        
        plot_start = 0
        plot_end = len(Within_predicted_Behavior[0]) #500 #
        
        Behavior = behavior_titles[ii].replace('EMG_', '', 1)
        #EMG = EMG.replace('1', '', 1)
        #EMG = EMG.replace('2', '', 1)
        
        fig, fig_axes = plt.subplots(figsize=(10, 5))
        
        plt.plot(test_Behavior_timeframe[plot_start:plot_end], \
                 Within_test_Behavior[0][plot_start:plot_end , ii] / force_convert * force_gain, 'k', label = 'Actual')
        
        plt.plot(test_Behavior_timeframe[plot_start:plot_end], \
                 Within_predicted_Behavior[0][plot_start:plot_end, ii] / force_convert * force_gain, 'r', label = 'Within Predicted')
            
        plt.plot(test_Behavior_timeframe[plot_start:plot_end], \
                 Cross_predicted_Behavior[plot_start:plot_end, ii] / force_convert * force_gain, 'g', label = 'Cross Predicted')
            
        title_string = 'Decoding Results - ' + Behavior
        plt.title(title_string, fontname = font_specs.font_name, fontsize = font_specs.title_font_size, fontweight = 'bold')
        # Axis Labels
        plt.xlabel('Time', fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
        plt.ylabel(Behavior, fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
        
        # Collect the current axis limits
        x_limits = fig_axes.get_xlim()
        plt.xlim(x_limits)
        
        if isinstance(man_y_axis, str):
            y_limits = fig_axes.get_ylim() 
            axis_expansion = y_limits[1] + 5*(np.std(Within_predicted_Behavior[0][plot_start:plot_end, ii])) / force_convert * force_gain
            # Reset the axis limits
            plt.ylim(y_limits[0], axis_expansion)
        else:
            plt.ylim(man_y_axis[0], man_y_axis[1])

        legend_font = fm.FontProperties(family = font_specs.font_name, size = font_specs.legend_font_size + 5)
        plt.legend(prop = legend_font)
        plt.legend(frameon = False)
        
        # Annotation of the r^2
        plt.text(0.15, 0.9, 'r\u00b2 = ' + str(round(np.mean(Within_r2_values[ii,:]), 2)), \
                 verticalalignment = 'center', horizontalalignment = 'center', color = 'red', \
                     transform = fig_axes.transAxes, fontname = font_specs.font_name, fontsize = font_specs.legend_font_size)
        plt.text(0.15, 0.8, 'r\u00b2 = ' + str(round(np.mean(Cross_r2_values[ii,:]), 2)), \
                 verticalalignment = 'center', horizontalalignment = 'center', color = 'green', \
                     transform = fig_axes.transAxes, fontname = font_specs.font_name, fontsize = font_specs.legend_font_size)
        
        # Figure Saving
        if Save_Figs != 0:
            save_dir = 'C:/Users/rhpow/Desktop/'
            fig_title = title_string
            fig_title = str.replace(fig_title, ':', '')
            fig_title = str.replace(fig_title, 'vs.', 'vs')
            fig_title = str.replace(fig_title, 'mg.', 'mg')
            fig_title = str.replace(fig_title, 'kg.', 'kg')
            fig_title = str.replace(fig_title, '.', '_')
            fig_title = str.replace(fig_title, '/', '_')
            plt.savefig(save_dir + fig_title + '.' + Save_Figs)
            plt.close()
        
        












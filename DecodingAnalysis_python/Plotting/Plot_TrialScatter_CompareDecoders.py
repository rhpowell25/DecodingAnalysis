
#%% Import basic packages
import matplotlib.pyplot as plt
import numpy as np
from Plot_Specs import Font_Specs

def Scatter_CompareDecoders(Self_Decoder_Vars, Cross_Decoder_Vars, Save_Figs):
    
    #%% Extracting the variables
    
    # Font & plotting specifications
    font_specs = Font_Specs()
    
    # Do you want to manually set the y-axis?
    man_axis = 'No'
    #man_axis = [-20, 300]
    axis_expansion = 2
    
    behavior_titles = Self_Decoder_Vars.behavior_titles
    
    # Convert the force millivolts to cursor position
    if behavior_titles == ['Force']:
        force_convert = 1000
        force_gain = 5
    else:
        force_convert = 1
        force_gain = 1
        
    # Extract the self predictions
    per_trial_Within_predictions = Self_Decoder_Vars.per_trial_predicted_Behavior
    for ii in range(len(per_trial_Within_predictions)):
        if ii == 0:
            Within_predictions = Self_Decoder_Vars.predicted_Behavior_test[0]
        else:
            Within_predictions = np.vstack((Within_predictions, Self_Decoder_Vars.predicted_Behavior_test[ii]))
    
    
    # Extract the corresponding cross predictions
    per_trial_Cross_predictions = Cross_Decoder_Vars.per_trial_predicted_Behavior
    for ii in range(len(per_trial_Within_predictions)):
        if ii == 0:
            Cross_predictions = Cross_Decoder_Vars.predicted_Behavior_test[0]
        else:
            Cross_predictions = np.vstack((Cross_predictions, Cross_Decoder_Vars.predicted_Behavior_test[ii]))
    
    # Convert the force
    Cross_predictions = Cross_predictions / force_convert * force_gain
    Within_predictions = Within_predictions / force_convert * force_gain
    
    # Bin size
    bin_size = 0.025
    offset = 0
    
    #%% Plotting the concatenated predicted behaviors
    
    for ii in range(len(behavior_titles)):
        
        Behavior = behavior_titles[ii].replace('EMG_', '', 1)
        #EMG = EMG.replace('1', '', 1)
        #EMG = EMG.replace('2', '', 1)
        
        # Plot the scatter
        fig, fig_axes = plt.subplots()
        for tt in range(len(per_trial_Cross_predictions)): # Number of folds
            for jj in range(len(per_trial_Cross_predictions[tt])): # Number of trials
                trial_time = np.arange(len(per_trial_Cross_predictions[tt][jj]))*bin_size - offset
                plt.scatter(per_trial_Within_predictions[tt][jj][:, ii] / force_convert * force_gain, \
                            per_trial_Cross_predictions[tt][jj][:, ii] / force_convert * force_gain, \
                            c = trial_time, cmap='plasma_r')
                    
        # Title the scatter
        title_string = 'Wiener Scatter - ' + Behavior
        plt.title(title_string, fontname = font_specs.font_name, \
                  fontsize = font_specs.title_font_size, fontweight = 'bold')

        # Add a color bar
        color_bar = plt.colorbar()
        color_bar.ax.set_yticklabels(['-0.5', 'Task Onset', '0.5', '1', '1.5', '2', '2.5', '3'])
        color_bar.set_label('Time (sec.)', fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
        
        # Axis Labels
        plt.xlabel('Within Predicted ' + Behavior, fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
        plt.ylabel('Cross Predicted ' + Behavior, fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
        
        # Set the axis
        if isinstance(man_axis, str):
            # Collect the current axis limits
            x_limits = fig_axes.get_xlim()
            y_limits = fig_axes.get_ylim()
            axis_min = round(min(min(x_limits, y_limits))/5)*5 - axis_expansion
            axis_max = round(max(max(x_limits, y_limits))/5)*5 + axis_expansion
            
        else:
            axis_min = man_axis[0]
            axis_max = man_axis[1]

        # Reset the axis limits
        plt.xlim(axis_min, axis_max)
        plt.ylim(axis_min, axis_max)  
        # Draw the unity line
        plt.plot([axis_max, axis_min], [axis_max, axis_min], linestyle = 'dashed', color = 'k')
        
        # Draw the line of best fit
        best_fit_slope, best_fit_intercept = \
            np.polyfit(Within_predictions[:,ii], Cross_predictions[:,ii], 1)
        plt.plot(Within_predictions[:,ii], \
                 best_fit_slope*Within_predictions[:,ii] + best_fit_intercept, color = 'r')
        
        plt.tight_layout()
        
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

        












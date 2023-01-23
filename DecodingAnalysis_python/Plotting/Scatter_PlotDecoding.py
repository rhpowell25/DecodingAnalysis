
#%% Import basic packages
import matplotlib.pyplot as plt
import numpy as np
from Plot_Specs import Font_Specs

class Plot_DecodingScatter():
    
    def __init__(self, Decoder_Vars, Save_Figs):
        #%% Extracting the variables
        
        # Font & plotting specifications
        font_specs = Font_Specs()
        
        # Do you want to manually set the y-axis?
        man_axis = 'No'
        #man_axis = [-20, 300]
        
        behavior_titles = Decoder_Vars.behavior_titles
        r2_values = Decoder_Vars.r2_values
        
        per_trial_testing_Behavior = Decoder_Vars.per_trial_testing_Behavior
        per_trial_predicted_Behavior = Decoder_Vars.per_trial_predicted_Behavior
        
        bin_size = 0.025
        offset = 1
        trial_time = np.arange(len(per_trial_testing_Behavior[0]))*bin_size - offset
        
        #%% Plotting the concatenated predicted behaviors
        
        for ii in range(len(behavior_titles)):
            
            Behavior = behavior_titles[ii].replace('EMG_', '', 1)
            #EMG = EMG.replace('1', '', 1)
            #EMG = EMG.replace('2', '', 1)
            
            # Plot the scatter
            fig, fig_axes = plt.subplots()
            for jj in range(len(per_trial_testing_Behavior)): # Number of trials
                trial_time = np.arange(len(per_trial_testing_Behavior[jj]))*bin_size - offset
                plt.scatter(per_trial_testing_Behavior[jj][:, ii], per_trial_predicted_Behavior[jj][:, ii], \
                            c = trial_time, cmap='plasma_r')
                    
            # Title the scatter
            title_string = 'Wiener Scatter - ' + Behavior
            plt.title(title_string, fontname = font_specs.font_name, \
                      fontsize = font_specs.title_font_size, fontweight = 'bold')

            # Add a color bar
            color_bar = plt.colorbar()
            color_bar.ax.set_yticklabels(['-1', '-0.5', 'Trial Go Cue', '0.5', '1', '1.5', '2', '2.5'])
            color_bar.set_label('Time (sec.)', fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
            
            # Axis Labels
            plt.xlabel('Actual ' + Behavior, fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
            plt.ylabel('Predicted ' + Behavior, fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
            
            # Set the axis
            if isinstance(man_axis, str):
                # Collect the current axis limits
                x_limits = fig_axes.get_xlim()
                y_limits = fig_axes.get_ylim()
                axis_min = round(min(min(x_limits, y_limits))/5)*5
                axis_max = round(max(max(x_limits, y_limits))/5)*5
                
            else:
                axis_min = man_axis[0]
                axis_max = man_axis[1]

            # Reset the axis limits
            plt.xlim(axis_min, axis_max)
            plt.ylim(axis_min, axis_max)  
            # Draw the unity line
            plt.plot([axis_max, axis_min], [axis_max, axis_min], linestyle = 'dashed', color = 'k')
            
            # Annotation of the r^2
            plt.text(0.15, 0.9, 'r\u00b2 = ' + str(round(r2_values[ii], 2)), \
                     verticalalignment = 'center', horizontalalignment = 'center', \
                         transform = fig_axes.transAxes, fontname = font_specs.font_name, fontsize = font_specs.legend_font_size)
                
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

        












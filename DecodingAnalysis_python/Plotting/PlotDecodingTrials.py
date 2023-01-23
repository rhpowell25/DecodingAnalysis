
#%% Import basic packages
import matplotlib.pyplot as plt
from Plot_Specs import Font_Specs
import numpy as np
import matplotlib.font_manager as fm

class Plot_DecodingTrials():
    
    def __init__(self, Decoder_Vars, Save_Figs):
        #%% Extracting the variables
        
        # Font & plotting specifications
        font_specs = Font_Specs()
        
        # Do you want to manually set the y-axis?
        man_y_axis = 'No'
        #man_y_axis = [-20, 25]
        
        behavior_titles = Decoder_Vars.behavior_titles
        r2_values = Decoder_Vars.r2_values_test
        
        test_Behavior_timeframe = Decoder_Vars.binned_timeframe
        predicted_Behavior = Decoder_Vars.predicted_Behavior_test[0]
        
        if 'H_baseline' in dir(Decoder_Vars):
            test_Behavior = Decoder_Vars.formatted_test_Behavior[0]
        else:
            test_Behavior = Decoder_Vars.formatted_test_Behavior
        
        #%% Plotting the concatenated predicted behavior
        
        for ii in range(len(behavior_titles)):
            
            plot_start = 0
            plot_end = len(predicted_Behavior)
            
            Behavior = behavior_titles[ii].replace('EMG_', '', 1)
            #EMG = EMG.replace('1', '', 1)
            #EMG = EMG.replace('2', '', 1)
            
            fig, fig_axes = plt.subplots(figsize=(10, 5))
            
            plt.plot(test_Behavior_timeframe[plot_start:plot_end], \
                     test_Behavior[plot_start:plot_end, ii], 'k', label = 'Actual')
            
            plt.plot(test_Behavior_timeframe[plot_start:plot_end], \
                     predicted_Behavior[plot_start:plot_end, ii], 'r', label = 'Predicted')
                
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
                axis_expansion = y_limits[1] + 5*(np.std(predicted_Behavior[plot_start:plot_end, ii]))
                # Reset the axis limits
                plt.ylim(y_limits[0], axis_expansion)
            else:
                plt.ylim(man_y_axis[0], man_y_axis[1])
    
            legend_font = fm.FontProperties(family = font_specs.font_name, size = font_specs.legend_font_size + 5)
            plt.legend(prop = legend_font)
            plt.legend(frameon = False)
            
            # Annotation of the r^2
            plt.text(0.15, 0.9, 'r\u00b2 = ' + str(round(np.mean(r2_values[ii,:]), 2)), \
                     verticalalignment = 'center', horizontalalignment = 'center', \
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
      
        
        #%% Plotting the per-trial predicted EMG's
        #fig, fig_axes = plt.subplots()
        
        #plt.plot(per_trial_testing_EMG[0][:,0], 'k')
        #plt.plot(per_trial_predicted_EMG[0][:,0], 'r')
        
        
        













#%% Import basic packages
import matplotlib.pyplot as plt
import numpy as np
from Plot_Specs import Font_Specs
import matplotlib.font_manager as fm
import scipy.stats as stats

def Hist_CompareDecoders(Self_Decoder_Vars, Cross_Decoder_Vars, Save_Figs):
    
    #%% Extracting the variables
    
    # What do you want to plot
    Plot_Touchpad = 0
    Plot_Ramp = 1
    Plot_TgtHold = 1
    
    # Font & plotting specifications
    font_specs = Font_Specs()
    
    # Do you want to manually set the y-axis?
    man_axis = 'No'
    #man_axis = [-20, 300]
    axis_expansion = 8
    
    # Annotation x-axis
    annot_x = 0.025
    
    behavior_titles = Self_Decoder_Vars.behavior_titles
    
    # Convert the force millivolts to cursor position
    if behavior_titles == ['Force']:
        force_convert = 1000
        force_gain = 5
    else:
        force_convert = 1
        force_gain = 1
        
    # Extract the per trial predictions
    per_trial_Within_predictions = Self_Decoder_Vars.per_trial_predicted_Behavior
    per_trial_Cross_predictions = Cross_Decoder_Vars.per_trial_predicted_Behavior
    
    # Bin size
    bin_size = 0.025
    
    # Touchpad Hold
    TouchpadHold = 0.4
    
    # Target Hold
    TgtHold = Self_Decoder_Vars.TgtHold
    
    # Define the epochs
    baseline_phase = int(TouchpadHold / bin_size)
    ramp_phase = int(TgtHold / bin_size)
    
    # Seperate the predictions based on the epoch
    for tt in range(len(per_trial_Cross_predictions)): # Number of folds
        for jj in range(len(per_trial_Cross_predictions[tt])): # Number of trials
            if jj == 0 and tt == 0:
                Cross_predicted_Baseline = per_trial_Cross_predictions[tt][jj][0:baseline_phase]
                Within_predicted_Baseline = per_trial_Within_predictions[tt][jj][0:baseline_phase]
                Cross_predicted_Ramp = per_trial_Cross_predictions[tt][jj][baseline_phase + 1:-1 - ramp_phase]
                Within_predicted_Ramp = per_trial_Within_predictions[tt][jj][baseline_phase + 1:-1 - ramp_phase]
                Cross_predicted_TgtHold = per_trial_Cross_predictions[tt][jj][-1 - ramp_phase + 1:-1]
                Within_predicted_TgtHold = per_trial_Within_predictions[tt][jj][-1 - ramp_phase + 1:-1]
            else:
                Cross_predicted_Baseline = \
                    np.vstack((Cross_predicted_Baseline, per_trial_Cross_predictions[tt][jj][0:baseline_phase]))
                Within_predicted_Baseline = \
                    np.vstack((Within_predicted_Baseline, per_trial_Within_predictions[tt][jj][0:baseline_phase]))
                Cross_predicted_Ramp = \
                    np.vstack((Cross_predicted_Ramp, per_trial_Cross_predictions[tt][jj][baseline_phase + 1:-1 - ramp_phase]))
                Within_predicted_Ramp = \
                    np.vstack((Within_predicted_Ramp, per_trial_Within_predictions[tt][jj][baseline_phase + 1:-1 - ramp_phase]))
                Cross_predicted_TgtHold = \
                    np.vstack((Cross_predicted_TgtHold, per_trial_Cross_predictions[tt][jj][-1 - ramp_phase + 1:-1]))
                Within_predicted_TgtHold = \
                    np.vstack((Within_predicted_TgtHold, per_trial_Within_predictions[tt][jj][-1 - ramp_phase + 1:-1]))
                
    # Convert the force
    Cross_predicted_Baseline = Cross_predicted_Baseline / force_convert * force_gain
    Within_predicted_Baseline = Within_predicted_Baseline / force_convert * force_gain
    Cross_predicted_Ramp = Cross_predicted_Ramp / force_convert * force_gain
    Within_predicted_Ramp = Within_predicted_Ramp / force_convert * force_gain
    Cross_predicted_TgtHold = Cross_predicted_TgtHold / force_convert * force_gain
    Within_predicted_TgtHold = Within_predicted_TgtHold / force_convert * force_gain
    
    # Run the statistics
    Baseline_p_val = stats.ttest_ind(Cross_predicted_Baseline, Within_predicted_Baseline)[1]
    Ramp_p_val = stats.ttest_ind(Cross_predicted_Ramp, Within_predicted_Ramp)[1]
    TgtHold_p_val = stats.ttest_ind(Cross_predicted_TgtHold, Within_predicted_TgtHold)[1]
    
    # Find the percent change in the predictions
    Cross_Baseline_mean = np.zeros(len(behavior_titles))
    Within_Baseline_mean = np.zeros(len(behavior_titles))
    Cross_Ramp_mean = np.zeros(len(behavior_titles))
    Within_Ramp_mean = np.zeros(len(behavior_titles))
    Cross_TgtHold_mean = np.zeros(len(behavior_titles))
    Within_TgtHold_mean = np.zeros(len(behavior_titles))
    for ii in range(len(behavior_titles)):
        Cross_Baseline_mean[ii] = np.mean(Cross_predicted_Baseline[:,ii])
        Within_Baseline_mean[ii] = np.mean(Within_predicted_Baseline[:,ii])
        Cross_Ramp_mean[ii] = np.mean(Cross_predicted_Ramp[:,ii])
        Within_Ramp_mean[ii] = np.mean(Within_predicted_Ramp[:,ii])
        Cross_TgtHold_mean[ii] = np.mean(Cross_predicted_TgtHold[:,ii])
        Within_TgtHold_mean[ii] = np.mean(Within_predicted_TgtHold[:,ii])
    
    Baseline_perc_change = (Cross_Baseline_mean - Within_Baseline_mean) / Cross_Baseline_mean
    Ramp_perc_change = (Cross_Ramp_mean - Within_Ramp_mean) / Cross_Ramp_mean
    TgtHold_perc_change = (Cross_TgtHold_mean - Within_TgtHold_mean) / Cross_TgtHold_mean
    
    #%% Plotting the baseline histogram
    
    if Plot_Touchpad == 1:
    
        for ii in range(len(behavior_titles)):
            
            Behavior = behavior_titles[ii].replace('EMG_', '', 1)
            #EMG = EMG.replace('1', '', 1)
            #EMG = EMG.replace('2', '', 1)
            
            # Plot the baseline histogram
            fig, fig_axes = plt.subplots()
            plt.hist(Within_predicted_Baseline[:,ii], alpha = font_specs.hist_transparency, 
                     edgecolor = 'k', color = 'r', label = 'Within Decoder')
            plt.hist(Cross_predicted_Baseline[:,ii], alpha = font_specs.hist_transparency, 
                     edgecolor = 'k', color = 'g', label = 'Cross Decoder')
                    
            # Title the baseline histogram
            title_string = 'Touchpad Hold - ' + Behavior
            plt.title(title_string, fontname = font_specs.font_name, \
                      fontsize = font_specs.title_font_size, fontweight = 'bold')
                
            # Axis Labels
            plt.xlabel('Predicted ' + Behavior, fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
            plt.ylabel('Indices - ' + Behavior, fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
            
            # Set the axis
            y_limits = fig_axes.get_ylim()
            if isinstance(man_axis, str):
                # Collect the current axis limits
                axis_min = min(min(Within_predicted_Baseline[:,ii]), min(Cross_predicted_Baseline[:,ii]))
                axis_min = round(axis_min/5)*5 - axis_expansion
                axis_max = max(max(Within_predicted_Baseline[:,ii]), max(Cross_predicted_Baseline[:,ii]))
                axis_max = round(axis_max/5)*5 + axis_expansion
                
            else:
                axis_min = man_axis[0]
                axis_max = man_axis[1]
    
            # Reset the axis limits
            plt.ylim(y_limits[0], y_limits[1]) 
            plt.xlim(axis_min, axis_max)
            
            # Lines indicating the means
            plt.plot([Cross_Baseline_mean[ii], Cross_Baseline_mean[ii]], [y_limits[0], y_limits[1]], \
                     color = 'g', linewidth = 3, linestyle = 'dashed')
            plt.plot([Within_Baseline_mean[ii], Within_Baseline_mean[ii]], [y_limits[0], y_limits[1]], \
                     color = 'r', linewidth = 3, linestyle = 'dashed')
            
            # Add the legend
            legend_font = fm.FontProperties(family = font_specs.font_name, size = font_specs.legend_font_size)
            plt.legend(prop = legend_font)
            plt.legend(frameon = False)
            
            # Only label every other tick
            for label in fig_axes.xaxis.get_ticklabels()[::2]:
                label.set_visible(False)
            for label in fig_axes.yaxis.get_ticklabels()[::2]:
                label.set_visible(False)
                
            # Set The Font
            plt.xticks(fontname = font_specs.font_name, fontsize = font_specs.label_font_size - 5)
            plt.yticks(fontname = font_specs.font_name, fontsize = font_specs.label_font_size - 5)
            
            # Annotation of the p-value
            if round(Baseline_p_val[ii], 3) > 0:
                plt.text(annot_x, 0.95, 'p = ' + str(round(Baseline_p_val[ii], 3)), 
                         verticalalignment = 'center', horizontalalignment = 'left', 
                         transform = fig_axes.transAxes, fontname = font_specs.font_name, fontsize = font_specs.legend_font_size)
            if round(Baseline_p_val[ii], 3) == 0:
                plt.text(annot_x, 0.95, 'p < 0.001', verticalalignment = 'center', horizontalalignment = 'left', 
                         transform = fig_axes.transAxes, fontname = font_specs.font_name, fontsize = font_specs.legend_font_size)
            # Annotation of the percent change
            plt.text(annot_x, 0.85, 'Δ% = ' + str(round(Baseline_perc_change[ii], 3)), 
                     verticalalignment = 'center', horizontalalignment = 'left', 
                     transform = fig_axes.transAxes, fontname = font_specs.font_name, fontsize = font_specs.legend_font_size)
            
            plt.tight_layout()
            
            # Print the baseline percent change
            print('The touchpad hold effect size is ' + str(round(Baseline_perc_change[ii], 3)))
            
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

    #%% Plotting the ramp phase histogram
    
    if Plot_Ramp == 1:
        
        for ii in range(len(behavior_titles)):
            
            Behavior = behavior_titles[ii].replace('EMG_', '', 1)
            #EMG = EMG.replace('1', '', 1)
            #EMG = EMG.replace('2', '', 1)
            
            # Plot the ramp phase histogram
            fig, fig_axes = plt.subplots()
            plt.hist(Within_predicted_Ramp[:,ii], alpha = font_specs.hist_transparency, 
                     edgecolor = 'k', color = 'r', label = 'Within Decoder')
            plt.hist(Cross_predicted_Ramp[:,ii], alpha = font_specs.hist_transparency, 
                     edgecolor = 'k', color = 'g', label = 'Cross Decoder')
                    
            # Title the baseline histogram
            title_string = 'Ramp Phase - ' + Behavior
            plt.title(title_string, fontname = font_specs.font_name, \
                      fontsize = font_specs.title_font_size, fontweight = 'bold')
                
            # Axis Labels
            plt.xlabel('Predicted ' + Behavior, fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
            plt.ylabel('Indices - ' + Behavior, fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
            
            # Set the axis
            y_limits = fig_axes.get_ylim()
            if isinstance(man_axis, str):
                # Collect the current axis limits
                axis_min = min(min(Within_predicted_Ramp[:,ii]), min(Cross_predicted_Ramp[:,ii]))
                axis_min = round(axis_min/5)*5 - axis_expansion
                axis_max = max(max(Within_predicted_Ramp[:,ii]), max(Cross_predicted_Ramp[:,ii]))
                axis_max = round(axis_max/5)*5 + axis_expansion
                
            else:
                axis_min = man_axis[0]
                axis_max = man_axis[1]
    
            # Reset the axis limits
            plt.ylim(y_limits[0], y_limits[1]) 
            plt.xlim(axis_min, axis_max)
            
            # Lines indicating the means
            plt.plot([Cross_Ramp_mean[ii], Cross_Ramp_mean[ii]], [y_limits[0], y_limits[1]], \
                     color = 'g', linewidth = 3, linestyle = 'dashed')
            plt.plot([Within_Ramp_mean[ii], Within_Ramp_mean[ii]], [y_limits[0], y_limits[1]], \
                     color = 'r', linewidth = 3, linestyle = 'dashed')
            
            # Add the legend
            legend_font = fm.FontProperties(family = font_specs.font_name, size = font_specs.legend_font_size)
            plt.legend(prop = legend_font)
            plt.legend(frameon = False)
            
            # Only label every other tick
            for label in fig_axes.xaxis.get_ticklabels()[::2]:
                label.set_visible(False)
            for label in fig_axes.yaxis.get_ticklabels()[::2]:
                label.set_visible(False)
                
            # Set The Font
            plt.xticks(fontname = font_specs.font_name, fontsize = font_specs.label_font_size - 5)
            plt.yticks(fontname = font_specs.font_name, fontsize = font_specs.label_font_size - 5)
            
            # Annotation of the p-value
            if round(Ramp_p_val[ii], 3) > 0:
                plt.text(annot_x, 0.95, 'p = ' + str(round(Ramp_p_val[ii], 3)), 
                         verticalalignment = 'center', horizontalalignment = 'left', 
                         transform = fig_axes.transAxes, fontname = font_specs.font_name, fontsize = font_specs.legend_font_size)
            if round(Ramp_p_val[ii], 3) == 0:
                plt.text(annot_x, 0.95, 'p < 0.001', verticalalignment = 'center', horizontalalignment = 'left', 
                         transform = fig_axes.transAxes, fontname = font_specs.font_name, fontsize = font_specs.legend_font_size)
            # Annotation of the percent change
            plt.text(annot_x, 0.85, 'Δ% = ' + str(round(Ramp_perc_change[ii], 3)), 
                     verticalalignment = 'center', horizontalalignment = 'left', 
                     transform = fig_axes.transAxes, fontname = font_specs.font_name, fontsize = font_specs.legend_font_size)
            
            plt.tight_layout()
            
            # Print the baseline percent change
            print('The ramp phase effect size is ' + str(round(Ramp_perc_change[ii], 3)))
            
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
            
    #%% Plotting the target hold histogram
    
    if Plot_TgtHold == 1:
        
        for ii in range(len(behavior_titles)):
            
            Behavior = behavior_titles[ii].replace('EMG_', '', 1)
            #EMG = EMG.replace('1', '', 1)
            #EMG = EMG.replace('2', '', 1)
            
            # Plot the target hold histogram
            fig, fig_axes = plt.subplots()
            plt.hist(Within_predicted_TgtHold[:,ii], alpha = font_specs.hist_transparency, 
                     edgecolor = 'k', color = 'r', label = 'Within Decoder')
            plt.hist(Cross_predicted_TgtHold[:,ii], alpha = font_specs.hist_transparency, 
                     edgecolor = 'k', color = 'g', label = 'Cross Decoder')
                    
            # Title the baseline histogram
            title_string = 'Target Hold - ' + Behavior
            plt.title(title_string, fontname = font_specs.font_name, \
                      fontsize = font_specs.title_font_size, fontweight = 'bold')
                
            # Axis Labels
            plt.xlabel('Predicted ' + Behavior, fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
            plt.ylabel('Indices - ' + Behavior, fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
            
            # Set the axis
            y_limits = fig_axes.get_ylim()
            if isinstance(man_axis, str):
                # Collect the current axis limits
                axis_min = min(min(Within_predicted_TgtHold[:,ii]), min(Cross_predicted_TgtHold[:,ii]))
                axis_min = round(axis_min/5)*5 - axis_expansion
                axis_max = max(max(Within_predicted_TgtHold[:,ii]), max(Cross_predicted_TgtHold[:,ii]))
                axis_max = round(axis_max/5)*5 + axis_expansion
                
            else:
                axis_min = man_axis[0]
                axis_max = man_axis[1]
    
            # Reset the axis limits
            plt.ylim(y_limits[0], y_limits[1]) 
            plt.xlim(axis_min, axis_max)
            
            # Lines indicating the means
            plt.plot([Cross_TgtHold_mean[ii], Cross_TgtHold_mean[ii]], [y_limits[0], y_limits[1]], \
                     color = 'g', linewidth = 3, linestyle = 'dashed')
            plt.plot([Within_TgtHold_mean[ii], Within_TgtHold_mean[ii]], [y_limits[0], y_limits[1]], \
                     color = 'r', linewidth = 3, linestyle = 'dashed')
            
            # Add the legend
            legend_font = fm.FontProperties(family = font_specs.font_name, size = font_specs.legend_font_size)
            plt.legend(prop = legend_font)
            plt.legend(frameon = False)
            
            # Only label every other tick
            for label in fig_axes.xaxis.get_ticklabels()[::2]:
                label.set_visible(False)
            for label in fig_axes.yaxis.get_ticklabels()[::2]:
                label.set_visible(False)
                
            # Set The Font
            plt.xticks(fontname = font_specs.font_name, fontsize = font_specs.label_font_size - 5)
            plt.yticks(fontname = font_specs.font_name, fontsize = font_specs.label_font_size - 5)
            
            # Annotation of the p-value
            if round(TgtHold_p_val[ii], 3) > 0:
                plt.text(annot_x, 0.95, 'p = ' + str(round(TgtHold_p_val[ii], 3)), 
                         verticalalignment = 'center', horizontalalignment = 'left', 
                         transform = fig_axes.transAxes, fontname = font_specs.font_name, fontsize = font_specs.legend_font_size)
            if round(TgtHold_p_val[ii], 3) == 0:
                plt.text(annot_x, 0.95, 'p < 0.001', verticalalignment = 'center', horizontalalignment = 'left', 
                         transform = fig_axes.transAxes, fontname = font_specs.font_name, fontsize = font_specs.legend_font_size)
            # Annotation of the percent change
            plt.text(annot_x, 0.85, 'Δ% = ' + str(round(TgtHold_perc_change[ii], 3)), 
                     verticalalignment = 'center', horizontalalignment = 'left', 
                     transform = fig_axes.transAxes, fontname = font_specs.font_name, fontsize = font_specs.legend_font_size)
            
            plt.tight_layout()
            
            # Print the baseline percent change
            print('The target hold effect size is ' + str(round(TgtHold_perc_change[ii], 3)))
            
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










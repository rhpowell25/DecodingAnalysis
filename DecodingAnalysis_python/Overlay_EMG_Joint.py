# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:10:29 2022

@author: rhpow
"""

#%% Import basic packages
import matplotlib.pyplot as plt
from Plot_Specs import Font_Specs
import numpy as np
import matplotlib.font_manager as fm

class Overlay_EMG_Joint():
    
    def __init__(self, EMG_Decoder, Joint_Decoder, Save_Figs):
        #%% Extracting the variables
        
        # Font & plotting specifications
        font_specs = Font_Specs()
        
        behavior_titles = EMG_Decoder.behavior_titles
        r2_values = EMG_Decoder.r2_values_test
        
        test_Behavior_timeframe = EMG_Decoder.test_Behavior_timeframe
        test_Behavior = EMG_Decoder.test_Behavior
        predicted_Behavior = EMG_Decoder.predicted_Behavior_test
        
        #%% Plotting the concatenated predicted behavior
        
        ii = 7
        plot_start = 0
        plot_end = len(predicted_Behavior[0])#-1
        
        Behavior = behavior_titles[ii].replace('EMG_', '', 1)
        #EMG = EMG.replace('1', '', 1)
        #EMG = EMG.replace('2', '', 1)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize = (10, 5))
        
        ax1.plot(test_Behavior_timeframe[plot_start:plot_end], \
                 test_Behavior[0][plot_start:plot_end, ii], 'k')
        
        ax1.plot(test_Behavior_timeframe[plot_start:plot_end], \
                 predicted_Behavior[0][plot_start:plot_end, ii], 'r', label = 'Predicted Lumb EMG')
            
        # Overlay the corresponding joint
        ax1.plot(test_Behavior_timeframe[plot_start:plot_end], \
                 Joint_Decoder.test_Behavior[0][plot_start:plot_end, 8], 'k')
        
        ax1.plot(test_Behavior_timeframe[plot_start:plot_end], \
                 Joint_Decoder.predicted_Behavior_test[0][plot_start:plot_end, 8], 'g', label = 'Predicted Lumb EMG')
            
        ax2.plot(test_Behavior_timeframe[plot_start:plot_end], \
                 test_Behavior[0][plot_start:plot_end, ii], 'k')
        
        ax2.plot(test_Behavior_timeframe[plot_start:plot_end], \
                 predicted_Behavior[0][plot_start:plot_end, ii], 'r', label = 'Predicted Lumb EMG')
            
        # Overlay the corresponding joint
        ax2.plot(test_Behavior_timeframe[plot_start:plot_end], \
                 Joint_Decoder.test_Behavior[0][plot_start:plot_end, 8], 'k', label = 'Actual')
        
        ax2.plot(test_Behavior_timeframe[plot_start:plot_end], \
                 Joint_Decoder.predicted_Behavior_test[0][plot_start:plot_end, 8], 'g', label = 'Predicted MCP1 Angle')
        
        ax1.set_ylim(65, 90)
        ax2.set_ylim(-5, 35)
        
        # hide the spines between ax and ax2
        ax1.spines.bottom.set_visible(False)
        ax2.spines.top.set_visible(False)
        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False)  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()
        
        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
        
        save_string = 'MCP1 vs Lumbricals'
        
        # Axis Labels
        plt.xlabel('Time', fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
        
        ax1.set_ylim(65, 90)
        ax2.set_ylim(-5, 40)
        
        legend_font = fm.FontProperties(family = font_specs.font_name, size = font_specs.legend_font_size + 5)
        plt.legend(prop = legend_font)
        plt.legend(frameon = False)
        
        # Annotation of the r^2
        plt.text(0.15, 0.75, 'r\u00b2 = ' + str(round(np.mean(r2_values[ii,:]), 2)), \
                 verticalalignment = 'center', horizontalalignment = 'center', \
                     transform = ax2.transAxes, fontname = font_specs.font_name, fontsize = font_specs.legend_font_size)
        plt.text(0.15, 0.9, 'r\u00b2 = ' + str(round(np.mean(Joint_Decoder.r2_values_test[8,:]), 2)), \
                 verticalalignment = 'center', horizontalalignment = 'center', \
                     transform = ax2.transAxes, fontname = font_specs.font_name, fontsize = font_specs.legend_font_size)
            
        # Figure Saving
        if Save_Figs != 0:
            save_dir = 'C:/Users/rhpow/Desktop/'
            fig_title = save_string
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
        
        
        













#%% Import basic packages
import matplotlib.pyplot as plt
from Plot_Specs import Font_Specs
import numpy as np

class BarChart_DecoderAccuracy():
    
    def __init__(self, Decoder_Vars, Save_Figs):
        #%% Extracting the variables
        
        # Font & plotting specifications
        font_specs = Font_Specs()
        
        behavior_titles = Decoder_Vars.behavior_titles
        
        # What accuracy measure do you want to print ('r^2', 'vaf')
        acc_choice = 'r^2'
        
        if acc_choice == 'vaf':
            Acc_Var = Decoder_Vars.vaf_values_test
        if acc_choice == 'r^2':
            Acc_Var = Decoder_Vars.r2_values_test
            
        # Do you want to manually set the axis?
        man_axis = 'No'
        #man_axis = [-0.1, 0.6]
            
        # Remove Thumb_IP_Flex_Ext
        #remove_joint = 'Thumb_IP_Flex_Ext'
        #joint_idx = behavior_titles.index(remove_joint)
        
        #Acc_Var = np.delete(Acc_Var, joint_idx, 0)
        #behavior_titles.remove(remove_joint)
            
        #%% Find the mean & standard deviations of the accuracy variables
        Acc_avg = np.zeros(len(behavior_titles))
        Acc_std = np.zeros(len(behavior_titles))
        for ii in range(len(behavior_titles)):
            Acc_avg[ii] = np.mean(Acc_Var[ii,:])
            Acc_std[ii] = np.std(Acc_Var[ii,:])

        #%% Plotting the predicted behaviors
        
        fig, fig_axes = plt.subplots(figsize=(len(behavior_titles) / 2, 6))
        
        plt.bar(behavior_titles, Acc_avg, yerr = Acc_std)
        
        # Title the bar chart
        title_string = 'Predictive Accuracy - ' + acc_choice
        plt.title(title_string, fontname = font_specs.font_name, \
                  fontsize = font_specs.title_font_size, fontweight = 'bold')
            
        # Change of fontsize and angle of xticklabels
        plt.setp(fig_axes.get_xticklabels(), fontsize = 10, rotation = 90)
        fig_axes.tick_params(axis='x', which='major', pad=5)

        # Axis Labels
        plt.ylabel('Accuracy', fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
        plt.xlabel('Output Variable', fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
        
        if isinstance(man_axis, str):
            plt.ylim(bottom = -0.1)
        else:
            plt.ylim(man_axis[0], man_axis[1])
        
        plt.tight_layout()
             
        #%% Figure Saving
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

        












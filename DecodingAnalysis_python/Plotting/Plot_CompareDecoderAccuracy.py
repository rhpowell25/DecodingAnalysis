
#%% Import basic packages
import matplotlib.pyplot as plt
from Plot_Specs import Font_Specs
import numpy as np

def CompareDecoderAccuracy(Decoder_Vars, Cross_Decoder_Vars, Save_Figs):
    
    #%% Extracting the variables
    
    # Plot method ('Bar' vs 'Scatter')
    plot_method = 'Bar'
    
    # Font & plotting specifications
    font_specs = Font_Specs()
    
    behavior_titles = Decoder_Vars.behavior_titles
    
    # What accuracy measure do you want to print ('r^2', 'vaf')
    acc_choice = 'r^2'
    
    if acc_choice == 'vaf':
        try:
            Within_Acc_Var = Decoder_Vars.vaf_values_test
        except:
            Within_Acc_Var = Decoder_Vars.vaf_values_train
        Cross_Acc_Var = Cross_Decoder_Vars.vaf_values_test
    if acc_choice == 'r^2':
        try:
            Within_Acc_Var = Decoder_Vars.r2_values_test
        except:
            Within_Acc_Var = Decoder_Vars.r2_values_train
        Cross_Acc_Var = Cross_Decoder_Vars.r2_values_test
        
    # Do you want to manually set the axis?
    man_axis = 'No'
    #man_axis = [-0.1, 0.25]
        
    #%% Find the mean & standard deviations of the accuracy variables
    Untrimmed_Acc_avg = np.zeros(len(behavior_titles))
    Untrimmed_Acc_std = np.zeros(len(behavior_titles))
    Trimmed_Acc_avg = np.zeros(len(behavior_titles))
    Trimmed_Acc_std = np.zeros(len(behavior_titles))
    for ii in range(len(behavior_titles)):
        Untrimmed_Acc_avg[ii] = np.mean(Within_Acc_Var[ii,:])
        Untrimmed_Acc_std[ii] = np.std(Within_Acc_Var[ii,:])
        Trimmed_Acc_avg[ii] = np.mean(Cross_Acc_Var[ii,:])
        Trimmed_Acc_std[ii] = np.std(Cross_Acc_Var[ii,:])
        
    #%% Plotting the predicted behaviors
    
    # Label locations
    label_loc = np.arange(len(behavior_titles))
    
    if plot_method == 'Bar':
        fig, fig_axes = plt.subplots(figsize=(len(behavior_titles) / 2, 6))
        
        # Bar chart width
        bar_width = 0.35
        
        plt.bar(label_loc - bar_width / 2, Untrimmed_Acc_avg, label = 'Morning')
        plt.bar(label_loc + bar_width / 2, Trimmed_Acc_avg, label = 'Afternoon')
        
        # Change of fontsize and angle of xticklabels
        fig_axes.set_xticks(label_loc, behavior_titles, fontsize = 10, rotation = 90)
        
        # Axis Labels
        fig_axes.set_ylabel('Accuracy', fontname = font_specs.font_name, fontsize = font_specs.label_font_size + 10)
        fig_axes.set_xlabel('Output Variable', fontname = font_specs.font_name, fontsize = font_specs.label_font_size + 10)
        
        # Add a legend
        fig_axes.legend(frameon = False)
        
    if plot_method == 'Scatter':
        
        fig, fig_axes = plt.subplots(figsize=(8, 8))
        
        plt.scatter(Untrimmed_Acc_avg, Trimmed_Acc_avg, color ='r')
        
        # Axis Labels
        fig_axes.set_xlabel('Self Decoder Accuracy', fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
        fig_axes.set_ylabel('Cross Decoder Accuracy', fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
  
        if isinstance(man_axis, str):
            # Set the axis limits
            y_max = max(Untrimmed_Acc_avg)
            x_max =  max(Trimmed_Acc_avg)
            y_min = min(Untrimmed_Acc_avg)
            x_min =  min(Trimmed_Acc_avg)
            
            axis_expansion = 0.1
            plt.ylim(min(x_min, y_min) - axis_expansion, max(x_max, y_max) + axis_expansion)
            plt.xlim(min(x_min, y_min) - axis_expansion, max(x_max, y_max) + axis_expansion)
            plt.ylim(min(x_min, y_min) - axis_expansion, max(x_max, y_max) + axis_expansion)
            plt.xlim(min(x_min, y_min) - axis_expansion, max(x_max, y_max) + axis_expansion)
            
            # Plot the unity line
            unity_x = [min(x_min, y_min) - axis_expansion, max(x_max, y_max) + axis_expansion]
            unity_y = [min(x_min, y_min) - axis_expansion, max(x_max, y_max) + axis_expansion]
            plt.plot(unity_x, unity_y, color = 'k', linestyle = 'dashed')
        else:
            plt.xlim(man_axis[0], man_axis[1])
            plt.ylim(man_axis[0], man_axis[1])
            # Plot the unity line
            unity_x = [man_axis[0], man_axis[1]]
            unity_y = [man_axis[0], man_axis[1]]
            plt.plot(unity_x, unity_y, color = 'k', linestyle = 'dashed')
            
  
    # Title the bar chart
    if len(Trimmed_Acc_avg) > 12:
        title_string = 'Joint Angle Decoder Accuracy - ' + acc_choice
    else:
        title_string = 'EMG Decoder Accuracy - ' + acc_choice
            
    plt.title(title_string, fontname = font_specs.font_name, \
              fontsize = font_specs.title_font_size + 10, fontweight = 'bold')

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

        












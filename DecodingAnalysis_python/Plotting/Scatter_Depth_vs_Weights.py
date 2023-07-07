#%% Import basic packages
import numpy as np
from Plot_Specs import Font_Specs
import matplotlib.pyplot as plt
import scipy.stats as stats

def Scatter_Depth_vs_Weights(Monkey, Date, Task, predict_what, Save_Figs):

    #%% Load the decoders
    import pickle
    base_path = 'C:/Users/rhpow/Documents/Work/Northwestern/Monkey_Data/' + Monkey + '/' + Date + '/Decoders/'
    
    # Morning decoder
    Decoder_File_Name = Date + '_' + Monkey + '_' + Task + '_' + 'Morn' + '_' + predict_what + '_Decoder.pkl'
    Decoder_Path = base_path + Decoder_File_Name
    with open(Decoder_Path, 'rb') as decoder_file:
        Morn_Decoder_Vars = pickle.load(decoder_file)
        
    # Afternoon decoder
    Decoder_File_Name = Date + '_' + Monkey + '_' + Task + '_' + 'Noon' + '_' + predict_what + '_Decoder.pkl'
    Decoder_Path = base_path + Decoder_File_Name
    with open(Decoder_Path, 'rb') as decoder_file:
        Noon_Decoder_Vars = pickle.load(decoder_file)
    
    #%% Load the weights of the neurons 
    from Heatmap_Decoder_Weights import decoder_mesh
    Plot_Figs = 0
    morn_weight = decoder_mesh('Morn Decoder Weights', Morn_Decoder_Vars, Noon_Decoder_Vars, Plot_Figs, 'png')
    noon_weight = decoder_mesh('Noon Decoder Weights', Noon_Decoder_Vars, Morn_Decoder_Vars, Plot_Figs, 'png')
    
    # How do you want to look at the decoder weights ('Mean' vs. 'Max')
    weight_choice = 'Max'
    
    # Extract the decoder weights
    morn_weights = np.zeros(len(morn_weight))
    noon_weights = np.zeros(len(noon_weight))
    for ii in range(len(morn_weight)):
        if weight_choice == 'Max':
            max_weight_idx = np.abs(morn_weight[ii]) == max(np.abs(morn_weight[ii]))
            morn_weights[ii] = morn_weight[ii][max_weight_idx]
            max_weight_idx = np.abs(noon_weight[ii]) == max(np.abs(noon_weight[ii]))
            noon_weights[ii] = noon_weight[ii][max_weight_idx]
        elif weight_choice == 'Mean':
          morn_weights[ii] = np.mean(morn_weight[ii])
          noon_weights[ii] = np.mean(noon_weight[ii])
    
    #%% Use either stable units or all units for the predictions
    
    from Load_Excel import Load_Excel
    xds_excel = Load_Excel(Date, Monkey, Task)
    
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
    
    #%% Extracting the variables
    
    # Annotation x-axis
    annot_x = 0.025
    
    # Do you want to manually set the y-axis?
    man_axis = 'No'
    #man_axis = [-20, 300]
    
    # Font & plotting specifications
    font_specs = Font_Specs()
    
    # Size of the markers
    sz = 20
    
    # Which firing rate phase do you want to plot? ('Baseline', 'Ramp', 'TgtHold', 'Depth')?
    fire_rate_phase = 'TgtHold';
    
    # Reassign variables according to what you're plotting
    if fire_rate_phase == 'Baseline':
        print('Baseline Firing Rate')
        fire_rate_morn = xds_excel['bsfr_morn']
        fire_rate_noon = xds_excel['bsfr_noon']

    if fire_rate_phase == 'Ramp':
        print('Ramp Phase')
        fire_rate_morn = xds_excel['ramp_morn']
        fire_rate_noon = xds_excel['ramp_noon']
    
    if fire_rate_phase == 'TgtHold':
        print('TgtHold Phase')
        fire_rate_morn = xds_excel['TgtHold_morn']
        fire_rate_noon = xds_excel['TgtHold_noon']
    
    if fire_rate_phase == 'Depth':
        print('Depth of Modulation')
        fire_rate_morn = xds_excel['depth_morn']
        fire_rate_noon = xds_excel['depth_noon']

    fire_rate_change = fire_rate_noon - fire_rate_morn
    
    # Decoder weights
    weight_change = noon_weights - morn_weights
    
    # Fire Rate * weights
    fire_rate_weight_morn = fire_rate_morn*morn_weights
    fire_rate_weight_noon = fire_rate_morn*noon_weights

    # Run the statistics
    fire_rate_weight_p_val = stats.ttest_rel(fire_rate_weight_morn, fire_rate_weight_noon)[1]
    Weights_p_val = stats.ttest_rel(morn_weights, noon_weights)[1]

    #%% Scatter plot the morning depth / weights

    # Morning scatter plot
    fig, fig_axes = plt.subplots()
    plt.scatter(fire_rate_morn, morn_weights, s = sz, edgecolors = 'k', )
    
    # Mornin scatter title
    title_string = 'Decoder Weights vs. Morning ' + fire_rate_phase
    plt.title(title_string, fontname = font_specs.font_name, fontsize = \
              font_specs.title_font_size, fontweight = 'bold')
    # Axis labels
    plt.xlabel('Morning  ' + fire_rate_phase, \
               fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
    plt.ylabel(weight_choice + ' Decoder Weights', \
               fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
    

    # Collect the current axis limits
    y_limits = fig_axes.get_ylim()
        
    # Draw the zero line
    plt.plot([0, 0], [y_limits[0], y_limits[1]], linestyle = 'dashed', color = 'k')
    
    # Reset the axis limits
    plt.ylim(y_limits[0], y_limits[1])
    
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
    
    #%% Scatter plot the afternoon depth / weights

    # Afternoon scatter plot
    fig, fig_axes = plt.subplots()
    plt.scatter(fire_rate_noon, noon_weights, s = sz, edgecolors = 'k', )
    
    # Mornin scatter title
    title_string = 'Decoder Weights vs. Afternoon  ' + fire_rate_phase
    plt.title(title_string, fontname = font_specs.font_name, fontsize = \
              font_specs.title_font_size, fontweight = 'bold')
    # Axis labels
    plt.xlabel('Afternoon  ' + fire_rate_phase, \
               fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
    plt.ylabel(weight_choice + ' Decoder Weights', \
               fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
    
    # Collect the current axis limits
    y_limits = fig_axes.get_ylim()
        
    # Draw the zero line
    plt.plot([0, 0], [y_limits[0], y_limits[1]], linestyle = 'dashed', color = 'k')
    
    # Reset the axis limits
    plt.ylim(y_limits[0], y_limits[1])
    
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
    
    #%% Scatter plot the morning vs afternoon weights

    # Decoder weight change scatter plot
    fig, fig_axes = plt.subplots()
    plt.scatter(morn_weights, noon_weights, s = sz, edgecolors = 'k', )
    
    # Scatter title
    title_string = 'Afternoon vs. Morning Decoder Weights'
    plt.title(title_string, fontname = font_specs.font_name, fontsize = \
              font_specs.title_font_size, fontweight = 'bold')
    # Axis labels
    plt.xlabel('Morning Decoder Weights', \
               fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
    plt.ylabel('Afternoon Decoder Weights', \
               fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
    
    # Set the axis
    if isinstance(man_axis, str):
        # Collect the current axis limits
        x_limits = fig_axes.get_xlim()
        y_limits = fig_axes.get_ylim()
        x_min = min(x_limits)
        x_max = max(x_limits)
        y_min = min(y_limits)
        y_max = max(y_limits)
        axis_min = round(min(x_min, y_min)/5)*5
        axis_max = round(max(x_max, y_max)/5)*5
    else:
        axis_min = man_axis[0]
        axis_max = man_axis[1]
        
    # Reset the axis limits
    plt.xlim(axis_min, axis_max)
    plt.ylim(axis_min, axis_max)
    
    # Draw the unity line
    plt.plot([axis_max, axis_min], [axis_max, axis_min], linestyle = 'dashed', color = 'k')
    
    # Annotation of the p-value
    if round(Weights_p_val, 3) > 0:
        plt.text(annot_x, 0.95, 'p = ' + str(round(Weights_p_val, 3)), 
                 verticalalignment = 'center', horizontalalignment = 'left', 
                 transform = fig_axes.transAxes, fontname = font_specs.font_name, fontsize = font_specs.legend_font_size)
    if round(Weights_p_val, 3) == 0:
        plt.text(annot_x, 0.95, 'p < 0.001', verticalalignment = 'center', horizontalalignment = 'left', 
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
        
    #%% Scatter plot the change in modulation / weights

    # Depth change scatter plot
    fig, fig_axes = plt.subplots()
    plt.scatter(fire_rate_change, weight_change, s = sz, edgecolors = 'k', )
    
    # Mornin scatter title
    title_string = 'Δ Decoder Weights vs. Δ ' + fire_rate_phase
    plt.title(title_string, fontname = font_specs.font_name, fontsize = \
              font_specs.title_font_size, fontweight = 'bold')
    # Axis labels
    plt.xlabel('Δ ' + fire_rate_phase, \
               fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
    plt.ylabel('Δ ' + weight_choice + ' Decoder Weights', \
               fontname = font_specs.font_name, fontsize = font_specs.label_font_size)
    
    # Set the axis
    if isinstance(man_axis, str):
        # Collect the current axis limits
        x_limits = fig_axes.get_xlim()
        y_limits = fig_axes.get_ylim()
        x_min = min(x_limits)
        x_max = max(x_limits)
        y_min = min(y_limits)
        y_max = max(y_limits)
        axis_min = round(min(x_min, y_min)/5)*5
        axis_max = round(max(x_max, y_max)/5)*5
    else:
        axis_min = man_axis[0]
        axis_max = man_axis[1]
        
    # Reset the axis limits
    plt.xlim(axis_min, axis_max)
    plt.ylim(axis_min, axis_max) 
        
    # Draw the unity line
    plt.plot([axis_max, axis_min], [axis_max, axis_min], linestyle = 'dashed', color = 'k')
    
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
        
    #%% Violin plots of the depth / decoder weights
    
    # Put the data into a dataframe
    import pandas as pd
    depth_d = {fire_rate_phase: np.hstack((fire_rate_morn, fire_rate_noon)), \
               'Decoder Weights': np.hstack((morn_weights,noon_weights)), \
                   'Firing Rate Phase x Weights': np.hstack((fire_rate_weight_morn, fire_rate_weight_noon)), \
                   'Experiment': np.hstack((['Morning']*len(morn_weights), \
                                            ['Afternoon']*len(noon_weights)))}
    fire_rate_df = pd.DataFrame(data = depth_d, index = range(len(morn_weights)*2))
    
    # Depth change violin plot
    import seaborn as sns
    fig, fig_axes = plt.subplots()
    sns.violinplot(data= fire_rate_df, x = "Experiment", y = "Firing Rate Phase x Weights", inner = "stick")
    
    # Annotation of the p-value
    if round(fire_rate_weight_p_val, 3) > 0:
        plt.text(annot_x, 0.95, 'p = ' + str(round(fire_rate_weight_p_val, 3)), 
        verticalalignment = 'center', horizontalalignment = 'left', 
        transform = fig_axes.transAxes, fontname = font_specs.font_name, fontsize = font_specs.legend_font_size)
    if round(fire_rate_weight_p_val, 3) == 0:
        plt.text(annot_x, 0.95, 'p < 0.001', verticalalignment = 'center', horizontalalignment = 'left', 
                 transform = fig_axes.transAxes, fontname = font_specs.font_name, fontsize = font_specs.legend_font_size)
        
    # Title the violin plot
    title_string = fire_rate_phase + ' x Morning Decoder Weights'
    plt.title(title_string, fontname = font_specs.font_name, \
              fontsize = font_specs.title_font_size, fontweight = 'bold')

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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
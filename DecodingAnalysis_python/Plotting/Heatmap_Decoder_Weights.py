#%% Import basic packages
import numpy as np
import matplotlib.pyplot as plt

def decoder_mesh(fig_title, Morn_Decoder_Vars, Noon_Decoder_Vars, Plot_Figs):
    
    """
    This function is used to extract the decoder weights for each neuron and plot
    It will plot the decoder matrix (reordered according to neuron)
    Also return a list of weights for neurons
    fig_title:
    H: decoder matrix
    n_lags:
    vmin:
    vmax:
    """
    
    # Extract the decoder variables
    H_morn = Morn_Decoder_Vars.H_baseline[0]
    H_noon = Noon_Decoder_Vars.H_baseline[0]
    n_lags = Morn_Decoder_Vars.n_lags
    
    vmin = min(np.min(H_morn[1:, :]), np.min(H_noon[1:, :]))
    vmax = min(np.max(H_morn[1:, :]), np.max(H_noon[1:, :]))

    # Extract the decoder matrix after the first row
    M = H_morn[1:, :]
    # Collect the matrix's dimensions
    Rows, Columns = M.shape
    # How many neurons in the decoder? 
    N = int(Rows/n_lags)
    
    # Extract the weights of each neuron
    w_all = []
    for ii in range(N):
        neuron_weight = []
        for jj in range(n_lags):
            neuron_weight.append(M[jj*N+ii, :])
        if ii == 0:
            lag_by_N = np.array(neuron_weight).T
        else:
            lag_by_N = np.vstack((lag_by_N, np.array(neuron_weight).T))
        w_all.append(np.array(neuron_weight))
    
    # Plot a heat map of the decoder weights
    if Plot_Figs == 1:

        from Plot_Specs import Font_Specs
        # Font & plotting specifications
        font_specs = Font_Specs()
        
        plt.figure(fig_title, figsize = (4, 8))
        x_axis = np.arange(n_lags)
        y_axis = np.arange(N)
        ax = plt.pcolormesh(x_axis, y_axis, lag_by_N, \
                            vmin = vmin, vmax = vmax, cmap = 'seismic')

        cbar = plt.colorbar(ax)
        cbar.ax.tick_params(labelsize=10)
        plt.yticks([])
        
        for ii in range(1, n_lags):
            plt.axvline(x = ii-0.5, ls="-", c="white")

        plt.title(fig_title, fontname = font_specs.font_name, fontsize = \
                  font_specs.title_font_size, fontweight = 'bold')
            
        plt.tight_layout() 
        
    return w_all

#%%
def matrix_mesh(fig_title, Morn_Decoder_Vars, Noon_Decoder_Vars):
    """
    This function is to visualize a "small" matrix for the weights of one neuron in the decoder
    """
    # Extract the decoder variables
    H_morn = Morn_Decoder_Vars.H_baseline[0]
    H_noon = Noon_Decoder_Vars.H_baseline[0]
    
    # Select your neuron
    neuron_choice = 0
    
    vmin = min(np.min(H_morn[1:, :]), np.min(H_noon[1:, :]))
    vmax = min(np.max(H_morn[1:, :]), np.max(H_noon[1:, :]))

    plt.figure(fig_title, figsize = (5, 4))
    R, C = H_morn[neuron_choice].shape
    x_axis = np.arange(R)
    y_axis = np.arange(C)
    ax = plt.pcolormesh(x_axis, y_axis , H_morn[neuron_choice].T, \
                        vmin = vmin, vmax = vmax, cmap = 'seismic')
    cbar = plt.colorbar(ax)
    cbar.ax.tick_params(labelsize=10)
    plt.xlabel('Time lags', fontsize = 20)
    plt.yticks([])
    plt.tight_layout()



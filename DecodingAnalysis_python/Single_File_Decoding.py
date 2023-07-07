
def Single_File_Decoding(Monkey, Date, Task, Morn_vs_Noon, predict_what):
    
    #%% Loading the file
    
    from Load_XDS import Load_XDS
    from Process_XDS import Process_XDS
    
    xds_morn = Load_XDS(Monkey, Date, Task, 1, 'Morn')
    xds_noon = Load_XDS(Monkey, Date, Task, 1, 'Noon')
    
    # Process the xds files
    if Task == 'PG':
        Match_The_Targets = 1
        xds_morn, xds_noon = Process_XDS(xds_morn, xds_noon, Match_The_Targets)
    
    #%% Do you want to look at the morning file or the afternoon file ('Morn', 'Noon')
    
    if Morn_vs_Noon == 'Morn':
        xds = xds_morn
        # Do you want to reserve trials for testing or train on all trials ('Yes', 'No')
        #reserve_trials = 'No'
        reserve_trials = 'Yes'
    if Morn_vs_Noon == 'Noon':
        xds = xds_noon
        # Do you want to reserve trials for testing or train on all trials ('Yes', 'No')
        reserve_trials = 'Yes'
        
    #%% Decode the file
    
    if predict_what == 'EMG':
        from MultiSessionEMGZero import MultiSession_EMGZero
        Zero_Factor = MultiSession_EMGZero.Multi_Session_EMG_Zero(xds_morn, xds_noon, 'All', 'Percentile', 1)
        from MultiSessionNormalizeEMG import MultiSession_NormalizeEMG
        Norm_Factor = MultiSession_NormalizeEMG.Multi_Session_Normalize_EMG(xds_morn, xds_noon, 'All', 95, 1)
        
    elif predict_what == 'Cursor':
        Zero_Factor = 0
        from Multi_Session_NormalizeCursor import Multi_Session_NormalizeCursor
        Norm_Factor = Multi_Session_NormalizeCursor(xds_morn, xds_noon, 1)
        
    else:
        Zero_Factor = 0
        Norm_Factor = 0
    
    from SelfDecoder import Self_Decoder
    Decoder_Vars = Self_Decoder(xds, predict_what, Zero_Factor, Norm_Factor, reserve_trials)
    
    #%% Save the decoder
    import pickle
    import os
    
    base_path = 'C:/Users/rhpow/Documents/Work/Northwestern/Monkey_Data/' + Monkey + '/' + Date + '/Decoders/'
    if os.path.exists(base_path) == False:
        os.mkdir(base_path)
        
    Decoder_File_Name = Date + '_' + Monkey + '_' + Task + '_' + Morn_vs_Noon + '_' + predict_what + '_Decoder.pkl'
        
    Decoder_Path = base_path + Decoder_File_Name
    
    with open(Decoder_Path, 'wb') as decoder_file:
        pickle.dump(Decoder_Vars, decoder_file)
    
    
     
     
     
     
     
     
     
     
     
     
     
     
     
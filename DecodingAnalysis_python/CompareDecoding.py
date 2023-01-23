# -*- coding: utf-8 -*-

def CompareDecoding(Monkey, Date, Task, predict_what):
    
    #%% Loading the files
    from Load_XDS import Load_XDS
    from Process_XDS import Process_XDS
    
    
    xds_morn = Load_XDS(Monkey, Date, Task, 'Morn')
    xds_noon = Load_XDS(Monkey, Date, Task, 'Noon')
    
    # Process the xds files
    if Task == 'PG':
        Match_The_Targets = 1
        xds_morn, xds_noon = Process_XDS(xds_morn, xds_noon, Match_The_Targets)
    
    #%% Load the self decoders
    import pickle

    base_path = 'C:/Users/rhpow/Documents/Work/Northwestern/Monkey_Data/' + Monkey + '/' + Date + '/Decoders/'
    
    # Morning decoder
    Decoder_File_Name = Date + '_' + Monkey + '_' + Task + '_' + 'Morn' + '_' + predict_what + '_Decoder.pkl'
    
    Decoder_Path = base_path + Decoder_File_Name
    with open(Decoder_Path, 'rb') as decoder_file:
        Morn_Decoder_Vars = pickle.load(decoder_file)
        
    # Afternoon decoder
    Decoder_File_Name = Date + '_' + Monkey + '_' + Task + '_' + 'Noon' + '_' + predict_what + '_Decoder.pkl'
    
    #%% Calculate the cross decoder
    if predict_what == 'EMG':
        from MultiSessionEMGZero import MultiSession_EMGZero
        Zero_Factor = MultiSession_EMGZero.Multi_Session_EMG_Zero(xds_morn, xds_noon, 'All', 'Percentile', 1)
        from MultiSessionNormalizeEMG import MultiSession_NormalizeEMG
        Norm_Factor = MultiSession_NormalizeEMG.Multi_Session_Normalize_EMG(xds_morn, xds_noon, 'All', 95, 1)
        
    elif predict_what == 'Cursor':
        Zero_Factor = 0
        from MultiSessionNormalizeCursor import MultiSession_NormalizeCursor
        Norm_Factor = MultiSession_NormalizeCursor.Multi_Session_Normalize_Cursor(xds_morn, xds_noon, 'All', 95, 1)
        
    else:
        Zero_Factor = 0
        Norm_Factor = 0
        
    from CrossDecoder import Cross_Decoder
    Cross_Decoder_Vars = Cross_Decoder(Morn_Decoder_Vars, xds_noon, predict_what, Zero_Factor, Norm_Factor)
    
    #%% Save the decoder
    import pickle
    import os
    
    base_path = 'C:/Users/rhpow/Documents/Work/Northwestern/Monkey_Data/' + Monkey + '/' + Date + '/Decoders/'
    if os.path.exists(base_path) == False:
        os.mkdir(base_path)
        
    Decoder_File_Name = 'Cross_Decoder_' + Date + '_' + Monkey + '_' + Task + '_' + predict_what + '_Decoder.pkl'
        
    Decoder_Path = base_path + Decoder_File_Name
    
    with open(Decoder_Path, 'wb') as decoder_file:
        pickle.dump(Cross_Decoder_Vars, decoder_file)


 

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
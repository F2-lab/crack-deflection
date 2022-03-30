# -*- coding: utf-8 -*-
"""
Created on Sat May 15 11:46:15 2021

@author: Yoshihiro Obata
"""
# %% importing packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob

# %% functions for preprocessing segmented crack
#############################################
def read_scan(files):
    """
    Reads binary image of crack and puts image data into 3D numpy array.

    Parameters
    ----------
    files : str
        image stack names

    Returns
    -------
    scan : numpy array (depth x height x width)
        binary numpy array of crack image data

    """
    # getting image dimensions and initializing array
    H, W = plt.imread(files[0]).shape
    D = len(files)
    scan = np.zeros((D, H, W))

    # reading images and putting values into array
    for i, fl in enumerate(files):
        scan[i] = plt.imread(fl)
        
    return scan

#############################################
def get_median_coord(img, use_nan=False):
    
    no_crack = np.float('nan') if use_nan else 0
    
    coord=[]
    for i in range(img.shape[0]):

        vec = img[i].copy()

        if len(np.unique(vec)) < 2:
            coord.append(no_crack)

        else:
            med = np.median(np.where(vec==1)[0])
            coord.append(int(round(med)))

    return np.array(coord)

#############################################
def get_stack_coord(files, use_nan=False):
    
    crack_coord=[]
    
    for fl in files:
        img = plt.imread(fl)
        img = img.copy()
        img[img>0]=1
        coord = get_median_coord(img, use_nan=use_nan)
        crack_coord.append(coord)
        
    return np.array(crack_coord)

#############################################
def center_at_notch(crack_coord):

    crack_def = crack_coord.copy()
    
    # get notch values
    vec_0 = crack_def[-1].copy()
    # make notch mask
    notch = np.zeros_like(vec_0)
    notch[vec_0 > 0] = 1

    # remove values outside notch mask
    crack_def *= notch
    # center at notch
    crack_def -= vec_0
    
    crack_def[crack_coord==0]=0
    
    return crack_def

#############################################
def process_no_crack(coord):

    new_coord = np.zeros_like(coord)

    for idx in range(coord.shape[1]):
        
        if np.all(np.isnan(coord[:, idx])):
            pass
        
        else:
            
            # set top non-crack values to 0
            count=0
            next_val = coord[count, idx]
            while np.isnan(next_val):
                coord[count, idx]=0
                count+=1
                next_val = coord[count, idx]

            # fill other gaps with neighboring value (bottom)
            df_def = pd.DataFrame({'d': coord[:, idx]})
            df_def.fillna(method='ffill', inplace=True)

            new_coord[:, idx]=df_def['d'].values
    
    return new_coord

#############################################
def preproc(fldrname, write=False, path=False):
    print(f'Starting preprocessing for {fldrname}...')    
    #  preprocessing images
    # get files for scan (update later)
    if path==False:
        files = glob('./' + fldrname + '/*')
    else:
        files = glob(fldrname + '/*')
        fldrname = fldrname.split('\\')[-1]
    
    # get stack median coordinates (takes about 15 to 30 seconds to run)
    coord = get_stack_coord(files, use_nan=True)
    
    # print(coord.shape)
    
    # # process areas with no crack
    coord = process_no_crack(coord)
    
    if write:
        df = pd.DataFrame(coord)
        parts = np.array(fldrname.split('_'))[[5,4,9,10]]
        sample_name = '_'.join(parts)
        df.to_csv('raw_'+sample_name+'.csv', index=False)
    
    print('Done')
    
    return coord
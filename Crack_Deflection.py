# package imports
import numpy as np
import pandas as pd
from Crack_Preprocess import preproc

from scipy.signal import savgol_filter
from sklearn.cluster import OPTICS

def apply_savgol_filter(coord, window=11, poly=2, axis=None):
    
    new_coord = np.zeros_like(coord)
    
    if axis==0:
        for idx in range(coord.shape[0]):
            new_coord[idx, :] = savgol_filter(coord[idx, :], window, poly)
    if axis==1:
        for idx in range(coord.shape[1]):
            new_coord[:, idx] = savgol_filter(coord[:, idx], window, poly)
    else:
        for idx in range(coord.shape[1]):
            new_coord[:, idx] = savgol_filter(coord[:, idx], window, poly)

        for idx in range(coord.shape[0]):
            new_coord[idx, :] = savgol_filter(coord[idx, :], window, poly)
        
    return new_coord


# %% functions for getting crack properties

#############################################
def apply_rolling_average(arr, window=10, fill_value=0, correct_top=False,
                          center=True):
    
    df = pd.DataFrame(arr)
    
    for j in range(df.shape[1]):
        df.iloc[:, j] = df.iloc[:, j].rolling(window=window, center=center).mean()
        
    df = df.fillna(value=fill_value)
    arr = df.values
    
    if correct_top:
        for j in range(arr.shape[1]):
            if arr[:, j].sum() > 0:
                # find first non-zero term
                idx0 = np.where(arr[:, j]!=0)[0].min()
                idx1 = idx0 + window + 2
                mean = arr[idx1:(idx1+window), j].mean()
                arr[:idx1, j] = mean
                arr[:idx0, j] = 0
            
    return arr

#############################################
def get_sections(c_grad_val, img_idx, window=50, view='side'):
    # 2 or -2 when sign flips so take abs
    sign_grad = np.sign(c_grad_val)
    flip = np.abs(sign_grad[:-1] - sign_grad[1:])
    
    # find index of deflection points
    deflect_points = np.where(flip != 0)[0]

    return deflect_points

#############################################
def get_angles(coord, img_idx, deflect_points):
    
    z_vals = coord[:, img_idx]
    start = 0
    # preallocate
    angles = np.zeros(len(deflect_points))
    vectors_o = np.zeros((len(deflect_points),2))
    vectors_u = np.zeros(len(deflect_points))
    vectors_v = np.zeros(len(deflect_points))
    # loop over deflections
    for i, section in enumerate(deflect_points):
        end = int(section)
        # get z and x dir change for this deflection chunk
        z_change = z_vals[end] - z_vals[start]
        x_change = end - start
        # get vector components and origin
        vectors_o[i,:] = start, z_vals[start]
        vectors_u[i] = x_change
        vectors_v[i] = z_change
        # calc angle
        angle = np.arctan(z_change/x_change)
        angles[i] = angle*180/np.pi
        
        start = int(section)
        
    return angles, vectors_o, vectors_u, vectors_v

#############################################
def crack_extension_length(x):
    '''
    Measures crack extension and length in the x, z view.
    
    Input: 
    x -- 1D np.array of the encoded crack
    
    Output:
    extension -- number of pixels the crack extends in the z direction
    length    -- length of the crack
    '''
    
    # check if there is a crack to measure
    if np.all(x==0):
        return 0, 0
    
    # find idx of crack tip
    idx_tip = np.where(x != 0)[0].min()
    
    # subset x to only contain the crack
    x_sub = x[idx_tip:].copy()
    
    # offset data
    x0 = x_sub[:-1]
    x1 = x_sub[1:]
    
    # get y term (1 for each entry)
    y = np.ones_like(x0)
    
    # compute total extension
    extension = y.sum()
    
    # compute total distance
    length = np.sqrt((x0 - x1)**2 + y).sum()
    
    return extension, length

#############################################   
def gradAndCoord(coord, window):
    #  gradient
    # get gradient
    crack_grad_side, _ = np.gradient(coord)
    # change direction so color map cooresponds to deflection
    crack_grad_side = -1*crack_grad_side
    
    # apply rolling average to the side
    crack_grad_side = apply_rolling_average(crack_grad_side, window=window)
    
    #  cropping data
    # shift to 0
    coord2 = np.where(coord == 0, np.nan, coord) # CHANGED
    coord2 -= coord2[-1,:]
    coord2 = -np.flipud(coord2) # NOTE: There was a negative sign in front (mistake?)
    
    
    low_dim = np.isnan(coord2)
    low_dim = low_dim == False
    low_dim = low_dim.sum(axis=0)
    
    cols = np.where(low_dim != 0)[0]
    start = cols[0]
    end = cols[-1]
    
    # chop off
    xrange = end - start
    # 5% of data on ends
    chop = 0.05
    chop_ind = int(xrange*chop)
    new_idx = cols[chop_ind:-chop_ind]
    coord = coord2[:,new_idx]
    
    return crack_grad_side, coord

#############################################
def get_raw_angles(coord, crack_grad_side, skip, window):
    #  getting angles and deflections
    # getting the angles and the vectors for increments of 10 slices
    num_imgs = coord.shape[1]
    slices = np.arange(num_imgs)[::skip]
    all_deflects = []
    angle_info = []
    
    no_nan = np.where(coord == 0, np.nan, coord)
    crack_grad_side = np.gradient(no_nan, axis=0)
    # change direction so color map cooresponds to deflection
    crack_grad_side = -1*crack_grad_side
    # apply rolling average to the side (larger window)
    crack_grad_side2 = apply_rolling_average(crack_grad_side.copy(), window=window)
    
    for num in slices:
        deflect_points = get_sections(crack_grad_side2[:, num], num)
        all_deflects.append(deflect_points)
        angles, vectors_o, vectors_u, vectors_v = get_angles(coord, num, deflect_points)
        angle_info.append((angles, vectors_o, vectors_u, vectors_v))
        
    return all_deflects, angle_info

#############################################
def unpackAndFilter(coord, angle_info, skip, angle_thresh, mag_thresh):
    a, x3d, y3d, z3d, u3d, v3d, w3d = [], [], [], [], [], [], []
    num_imgs = coord.shape[1]
    slices = np.arange(num_imgs)[::skip]

    for i, info in enumerate(angle_info):
        if len(info[1]) == 0:
            continue
        a.append(info[0])
        y3d.append(info[1][:,0])
        z3d.append(info[1][:,1])
        x3d.append([slices[i]]*len(info[1]))
        v3d.append(info[2])
        w3d.append(info[3])
        u3d.append([0]*len(info[2]))
    
    a3d = np.hstack(a)
    x3d = np.hstack(x3d)
    y3d = np.hstack(y3d)
    z3d = np.hstack(z3d)
    u3d = np.hstack(u3d)
    v3d = np.hstack(v3d)
    w3d = np.hstack(w3d)
    
    idx1 = np.abs(a3d) > angle_thresh
    mag = np.linalg.norm(np.array([u3d,v3d,w3d]), axis=0)
    idx2 = mag > mag_thresh
    idx_final = np.logical_and(idx1, idx2)
    a3d = a3d[idx_final]
    x3d = x3d[idx_final]
    y3d = y3d[idx_final]
    z3d = z3d[idx_final]
    u3d = u3d[idx_final]
    v3d = v3d[idx_final]
    w3d = w3d[idx_final]
    mag3d = mag[idx_final]
    
    return a3d, x3d, y3d, z3d, u3d, v3d, w3d, mag3d

#############################################
def do_clustering(a3d, x3d, y3d):
    df = pd.DataFrame(dict(angle=a3d, x=x3d, y=y3d))
    # Building the OPTICS Clustering model
    optics_model = OPTICS(min_samples=4, xi=0.05, min_cluster_size=5)
    # Training the model
    optics_model.fit(df)
    
    idxs = optics_model.ordering_
    labels = optics_model.labels_[idxs]
    unique = np.unique(labels)
    label_idx = []

    for label in unique:
        idx_lab = labels == label
        idx_true = idxs[idx_lab]
        label_idx.append(idx_true)   
    #  mean angles and magnitudes
    d_labels = unique[1:]
    i_labels = label_idx[1:]
    
    return d_labels, i_labels

#############################################
def get_cluster_data(d_labels, i_labels, coord, a3d, mag3d):
    mean_angles = []
    mean_mags = []
    for clust in range(len(d_labels)):
        c_idx = i_labels[clust]
        c_angles = a3d[c_idx]
        c_mag = mag3d[c_idx]
        mean_angles.append(np.mean(c_angles))
        mean_mags.append(np.mean(c_mag))
        
    # print(mean_angles, mean_mags)
    labeled = np.hstack(i_labels)
    a3d = a3d[labeled]
    mag3d = mag3d[labeled]
    
    # 
    crack_extension, crack_length = [], []
    coord_flip = coord.copy()
    coord_flip[np.isnan(coord)] = 0
    for j in range(coord.shape[1]):    
        e, l = crack_extension_length(coord_flip[:, j])
        crack_extension.append(e)
        crack_length.append(l)
    
    px2mm = 1.6*0.001
    crack_sa = np.sum(crack_length)*px2mm**2
    mean_len = np.mean(crack_length)*px2mm
    mean_exten = np.mean(crack_extension)*px2mm
    
    return mean_angles, mean_mags, mean_len, mean_exten, crack_sa

#############################################    
def make_df(fname, mean_angles, mean_mags, mean_len, mean_exten, crack_sa):
    crack_props = {}
    
    crack_props['Angle'] = mean_angles
    crack_props['Mag'] = mean_mags
    
    
    df = pd.DataFrame(crack_props)
    df['Exten (mm)'] = mean_exten
    df['Length (mm)'] = mean_len
    parts = fname[4:]
    sample_name = 'processed_' + parts
    df['Sample'] = sample_name
    df['SA (mm2)'] = crack_sa
    
    cols = list(df.drop('Sample', axis=1).columns)
    cols = ['Sample'] + cols
    df = df[cols]
    df.to_csv(sample_name, index=False)

#############################################    
def get_sample_props(fname, window=50, angle_thresh=35, mag_thresh=30, 
                     skip=10, get_slice=False, tune_clustering=False):
    # fname is the csv name of the raw coord data after preprocessing
    # folder is no longer accepted
    try:
        coord = pd.read_csv(fname)
    except:
        raise ValueError(f'{fname} does not exist. Try preprocessing.')

    crack_grad_side, coord = gradAndCoord(coord, window=window) 
    
    #  getting angles and deflections
    all_deflects, angle_info = get_raw_angles(coord, crack_grad_side, skip=skip, window=window)
    
    # if tuning desired, return the info needed to tune
    if tune_clustering:
        return coord, angle_info
    else:
        pass
    
    # unpack the angle info and apply the thresholds
    a3d, x3d, y3d, z3d, u3d, v3d, w3d, mag3d = unpackAndFilter(coord, angle_info, 
                skip=skip, angle_thresh=angle_thresh, mag_thresh=mag_thresh)

    if len(a3d) > 10:    
        # perform optical clustering on the filtered angles and x-y locations
        d_labels, i_labels = do_clustering(a3d, x3d, y3d)
        
        # get mean cluster values
        mean_angles, mean_mags, mean_len, mean_exten, crack_sa = get_cluster_data(d_labels, 
                                            i_labels, coord, a3d, mag3d)
    else:
        mean_angles, mean_mags, mean_len, mean_exten, crack_sa = [0], [0], [0], [0], [0]
    
    # create dataframe from clustered large deflections
    make_df(fname, mean_angles, mean_mags, mean_len, mean_exten, crack_sa)
    print(f'Got properties for {fname}.')
    print('All done.')

#############################################    
def get_slice_props(filename, window=50, angle_thresh=35, mag_thresh=30, write=True, skip=10):
    '''
    Extracts crack properties for each slice
    
    Input:
    filename     -- filename for the csv containing the pre-processed crack should follow "raw_..."
    window       -- window to be used in the rolling average
    angle_thresh -- threshold used to filter out minumum angle deflection
    mag_thresh   -- threshold used to filter out minimum magnitude of deflection
    write        -- boolean stating whether to write results
    
    Output:
    df -- data frame of results
    '''

    # read in raw coord data
    coord = pd.read_csv(filename).values

    # set 0s to nans
    coord = np.where(coord == 0, np.nan, coord)
    # center at notch
    coord -= coord[-1,:]
    coord = np.flipud(coord)


    # get edges with no crack
    low_dim = np.isnan(coord)
    low_dim = low_dim == False
    low_dim = low_dim.sum(axis=0)

    # get start and stop values
    cols = np.where(low_dim != 0)[0]
    start = cols[0]
    end = cols[-1]

    # chop off ends with additional percent
    xrange = end - start
    # 5% of data on ends
    chop = 0.05
    chop_ind = int(xrange*chop)
    new_idx = cols[chop_ind:-chop_ind]
    coord = coord[:,new_idx]

#########################
    # get gradient
    crack_grad_side = np.gradient(coord, axis=0)
    crack_grad_side = -1*crack_grad_side
    
    # apply rolling average to gradient
    crack_grad_side2 = apply_rolling_average(crack_grad_side.copy(), window=window)
#######################
#     no_nan = np.where(coord == 0, np.nan, coord)
#     crack_grad_side = np.gradient(no_nan, axis=0)
#     # change direction so color map cooresponds to deflection
#     crack_grad_side = -1*crack_grad_side
#     # apply rolling average to the side (larger window)
#     crack_grad_side2 = apply_rolling_average(crack_grad_side.copy(), window=window)
########################

    # dictionary to store results
    results = {'slice':[],
              'num_deflect':[], 
              'ang_avg':[],
              'ang_std':[],
              'mag_avg':[],
              'mag_std':[],
              'extension':[],
              'length':[]}

    # loop over each slice
    for num in range(coord.shape[1]):
        # get deflection information
        deflect_points = get_sections(crack_grad_side2[:, num], num)
        angles, vectors_o, vectors_u, vectors_v = get_angles(coord, num, deflect_points)

        # get the magnitude of the vectors
        uv = np.concatenate([vectors_u.reshape(-1, 1), vectors_v.reshape(-1, 1)], axis=1)
        mag = np.linalg.norm(uv, axis=1)
        
        # get absolute values of angles
        angles = np.abs(angles)
        
        # make filter using magnitude and angle
#         print(mag)
#         print(angles)
        
        mag[np.isnan(mag)]=0
        angles[np.isnan(angles)]=0
        mag_bool = mag > mag_thresh
        angle_bool = angles > angle_thresh
        keep = mag_bool & angle_bool

        # record slice number
        results['slice'].append(num+new_idx[0])

        # record angle info
        if keep.sum() > 0:
            angles = angles[keep]
            mag = mag[keep]
            results['num_deflect'].append(keep.sum())
            results['ang_avg'].append(angles.mean())
            results['ang_std'].append(angles.std())
            results['mag_avg'].append(mag.mean())
            results['mag_std'].append(mag.std())

        else:
            results['num_deflect'].append(0)
            results['ang_avg'].append(0)
            results['ang_std'].append(0)
            results['mag_avg'].append(0)
            results['mag_std'].append(0)

        # get crack array for current slice
        crack = np.flipud(coord[:, num].copy())
        crack[np.isnan(crack)]=0
        
        # get extension and length
        extension, length = crack_extension_length(crack)
        results['extension'].append(extension)
        results['length'].append(length)

    # put results in a dataframe
    df = pd.DataFrame(results)

    # make the output name based on the filename
    outname = ['sliceProps'] + filename.split('_')[1:]
    temp, load = outname[2:4]
    outname = '_'.join(outname)

    # get load information
    load = load.lower()
    load = load.replace('load', '')
    load = load.replace('pt', '.')
    load = load.replace('n', '')

    df['load'] = load
    df['temp'] = temp

    # write data
    if write:
        df.to_csv(outname, index=False)
    
    return df

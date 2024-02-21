import numpy as np

def read_input(inputfile):
    import h5py
    import os
    list_input = open("%s"%inputfile)
    nfiles = 0
    for line in list_input:
        fname = line.rstrip()
        if fname.startswith('#'):
            continue
        if not os.path.getsize(fname):
            continue
        print("read file", fname)
        h5f = h5py.File( fname, 'r')
        if nfiles == 0:
           X = h5f['X'][:]
           Y = h5f['Y'][:]
    
        else:
           X = np.concatenate((X, h5f['X']), axis=0)
           Y = np.concatenate((Y, h5f['Y']), axis=0)
        h5f.close()
        nfiles += 1
    
    print("finish reading files")
    return X, Y

def read_input_New(inputfile):
    # based on the hdf5 file from convertNanoToHDF5New.py
    import h5py
    import os
    list_input = open("%s"%inputfile)
    nfiles = 0
    for line in list_input:
        fname = line.rstrip()
        if fname.startswith('#'):
            continue
        if not os.path.getsize(fname):
            continue
        print("read file", fname)
        h5f = h5py.File( fname, 'r')
        if nfiles == 0:
           X = h5f['X'][:]
           X_c_0 = h5f['X_c_0'][:]
           X_c_1 = h5f['X_c_1'][:]
           X_c_2 = h5f['X_c_2'][:]
           Y = h5f['Y'][:]
        else:
           X = np.concatenate((X, h5f['X']), axis=0)
           X_c_0 = np.concatenate((X_c_0, h5f['X_c_0']), axis=0)
           X_c_1 = np.concatenate((X_c_1, h5f['X_c_1']), axis=0)
           X_c_2 = np.concatenate((X_c_2, h5f['X_c_2']), axis=0)
           Y = np.concatenate((Y, h5f['Y']), axis=0)
        h5f.close()
        nfiles += 1
        
    X = np.concatenate((X, X_c_0, X_c_1, X_c_2), axis=2)
    print("finish reading files")
    return X, Y

def preProcessing(X, EVT=None):
    """ pre-processing input """
    norm = 50.0

    dxy = X[:,:,5:6]
    dz  = X[:,:,6:7].clip(-100, 100)
    eta = X[:,:,3:4]
    mass = X[:,:,8:9]
    pt = X[:,:,0:1] / norm
    puppi = X[:,:,7:8]
    px = X[:,:,1:2] / norm
    py = X[:,:,2:3] / norm

    # remove outliers
    pt[ np.where(np.abs(pt>200)) ] = 0.
    px[ np.where(np.abs(px>200)) ] = 0.
    py[ np.where(np.abs(py>200)) ] = 0.

    if EVT is not None:
        # environment variables
        evt = EVT[:,0:4]
        evt_expanded = np.expand_dims(evt, axis=1)
        evt_expanded = np.repeat(evt_expanded, X.shape[1], axis=1)
        # px py has to be in the last two columns
        inputs = np.concatenate((dxy, dz, eta, mass, pt, puppi, evt_expanded, px, py), axis=2)
    else:
        inputs = np.concatenate((dxy, dz, eta, mass, pt, puppi, px, py), axis=2)

    inputs_cat0 = X[:,:,11:12] # encoded PF pdgId
    inputs_cat1 = X[:,:,12:13] # encoded PF charge
    inputs_cat2 = X[:,:,13:14] # encoded PF fromPV

    return inputs, inputs_cat0, inputs_cat1, inputs_cat2

def preProcessingNew(X, EVT=None, PVrobust=False, NoPUPPI=False):
    """ pre-processing input based on convertNanoToHDF5New"""
    # norm_factor of 50 already applied to all pt variables
    dxy = X[:,:,0:1]
    dz  = X[:,:,2:3].clip(-100, 100)
    eta = X[:,:,4:5]
    mass = X[:,:,6:7]
    pt = X[:,:,7:8]
    puppi = X[:,:,8:9]
    px = X[:,:,20:21]
    py = X[:,:,21:22]

    # remove outliers
    pt[ np.where(np.abs(pt>200)) ] = 0.
    px[ np.where(np.abs(px>200)) ] = 0.
    py[ np.where(np.abs(py>200)) ] = 0.

    if EVT is not None:
        # environment variables
        evt = EVT[:,0:4]
        evt_expanded = np.expand_dims(evt, axis=1)
        evt_expanded = np.repeat(evt_expanded, X.shape[1], axis=1)
        # px py has to be in the last two columns
        if not NoPUPPI:
            if not PVrobust:
                inputs = np.concatenate((dxy, dz, eta, mass, pt, puppi, evt_expanded, px, py), axis=2)
            else:
                # drop dxy
                inputs = np.concatenate((dz, eta, mass, pt, puppi, evt_expanded, px, py), axis=2)
        else:
            if not PVrobust:
                inputs = np.concatenate((dxy, dz, eta, mass, pt, evt_expanded, px, py), axis=2)
            else:
                # drop dxy
                inputs = np.concatenate((dz, eta, mass, pt, evt_expanded, px, py), axis=2)
    else:
        if not NoPUPPI:
            if not PVrobust:
                inputs = np.concatenate((dxy, dz, eta, mass, pt, puppi, px, py), axis=2)
            else:
                inputs = np.concatenate((dz, eta, mass, pt, puppi, px, py), axis=2)
        else:
            if not PVrobust:
                inputs = np.concatenate((dxy, dz, eta, mass, pt, px, py), axis=2)
            else:
                inputs = np.concatenate((dz, eta, mass, pt, px, py), axis=2)

    inputs_cat0 = X[:,:,22:23] # encoded PF charge
    inputs_cat1 = X[:,:,23:24] # encoded PF pdgId
    inputs_cat2 = X[:,:,24:25] # encoded PF fromPV

    if not PVrobust:
        return inputs, inputs_cat0, inputs_cat1, inputs_cat2
    else:
        # drop fromPV
        return inputs, inputs_cat0, inputs_cat1

def plot_history(history, path):
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.use('Agg')

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.ylim([0.00001, 20])
    plt.yscale('log')
    plt.legend()
    plt.savefig(path+'/history.pdf', bbox_inches='tight')

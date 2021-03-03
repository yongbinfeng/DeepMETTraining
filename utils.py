import numpy as np
import tables

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

def get_features(file_name, features, number_of_cands):
    # load file
    h5file = tables.open_file(file_name, 'r')
    nevents = getattr(h5file.root,features[0]).shape[0]
    nfeatures = len(features)
    print(nevents)

    # allocate arrays
    feature_array = np.zeros((nevents,number_of_cands,nfeatures))

    # load feature arrays
    for j in range(number_of_cands):
        for (i, feat) in enumerate(features):
            feature_array[:,j,i] = getattr(h5file.root, feat)[:,j]
    h5file.close()
    return feature_array

def get_targets(file_name, targets):
    # load file
    h5file = tables.open_file(file_name, 'r')
    nevents = getattr(h5file.root,targets[0]).shape[0]
    ntargets = len(targets)
    print(nevents)

    # allocate arrays
    target_array = np.zeros((nevents,ntargets))

    # load target arrays
    for (i, targ) in enumerate(targets):
        target_array[:,i] = getattr(h5file.root,targ)[:]

    h5file.close()
    return target_array

def convertXY2PtPhi(arrayXY):
    # convert from array with [:,0] as X and [:,1] as Y to [:,0] as pt and [:,1] as phi
    nevents = arrayXY.shape[0]
    arrayPtPhi = np.zeros((nevents, 2))
    arrayPtPhi[:,0] = np.sqrt((arrayXY[:,0]**2 + arrayXY[:,1]**2))
    arrayPtPhi[:,1] = np.sign(arrayXY[:,1])*np.arccos(arrayXY[:,0]/arrayPtPhi[:,0])
    return arrayPtPhi

def MakePlots(truth_XY, predict_XY, baseline_XY):
    # make the 1d distribution, response, resolution, 
    # and response-corrected resolution plots
    # assume the input has [:,0] as X and [:,1] as Y
    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)
    truth_PtPhi = convertXY2PtPhi(truth_XY)
    predict_PtPhi = convertXY2PtPhi(predict_XY)
    baseline_PtPhi = convertXY2PtPhi(baseline_XY)
    Make1DHists(truth_XY[:,0], predict_XY[:,0], baseline_XY[:,0], -100, 100, 40, False, 'MET X [GeV]', 'A.U.', 'MET_x.png')
    Make1DHists(truth_XY[:,1], predict_XY[:,1], baseline_XY[:,1], -100, 100, 40, False, 'MET Y [GeV]', 'A.U.', 'MET_y.png')
    Make1DHists(truth_PtPhi[:,0], predict_PtPhi[:,0], baseline_PtPhi[:,0], 0, 400, 40, False, 'MET Pt [GeV]', 'A.U.', 'MET_pt.png')
    # do statistics
    from scipy.stats import binned_statistic
    binnings = np.linspace(0, 400, num=21)
    print(binnings)
    truth_means,    bin_edges, binnumber = binned_statistic(truth_PtPhi[:,0], truth_PtPhi[:,0],    statistic='mean', bins=binnings, range=(0,400))
    predict_means,  _,         _ = binned_statistic(truth_PtPhi[:,0], predict_PtPhi[:,0],  statistic='mean', bins=binnings, range=(0,400))
    baseline_means, _,         _ = binned_statistic(truth_PtPhi[:,0], baseline_PtPhi[:,0], statistic='mean', bins=binnings, range=(0,400))
    # plot response
    plt.figure()
    plt.hlines(truth_means/truth_means, bin_edges[:-1], bin_edges[1:], colors='k', lw=5,
           label='Truth', linestyles='solid')
    plt.hlines(predict_means/truth_means, bin_edges[:-1], bin_edges[1:], colors='r', lw=5,
           label='Predict', linestyles='solid')
    plt.hlines(baseline_means/truth_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=5,
           label='Baseline', linestyles='solid')
    plt.xlim(0,400.0)
    plt.ylim(0,1.1)
    plt.xlabel('Truth MET [GeV]')
    plt.legend(loc='lower right')
    plt.ylabel('<MET Estimation>/<MET Truth>')
    plt.savefig("MET_response.png")
    plt.close()
    # response correction factors
    sfs_truth    = np.take(truth_means/truth_means,    np.digitize(truth_PtPhi[:,0], binnings)-1, mode='clip')
    sfs_predict  = np.take(predict_means/truth_means,  np.digitize(truth_PtPhi[:,0], binnings)-1, mode='clip')
    sfs_baseline = np.take(baseline_means/truth_means, np.digitize(truth_PtPhi[:,0], binnings)-1, mode='clip')
    # resolution defined as (q84-q16)/2.0
    def resolqt(y):
        return(np.percentile(y,84)-np.percentile(y,16))/2.0
    bin_resolX_predict, bin_edges, binnumber = binned_statistic(truth_PtPhi[:,0], truth_XY[:,0] - predict_XY[:,0] * sfs_predict, statistic=resolqt, bins=binnings, range=(0,400))
    bin_resolY_predict, _, _                 = binned_statistic(truth_PtPhi[:,0], truth_XY[:,1] - predict_XY[:,1] * sfs_predict, statistic=resolqt, bins=binnings, range=(0,400))
    bin_resolX_baseline, _, _                = binned_statistic(truth_PtPhi[:,0], truth_XY[:,0] - baseline_XY[:,0] * sfs_predict, statistic=resolqt, bins=binnings, range=(0,400))
    bin_resolY_baseline, _, _                = binned_statistic(truth_PtPhi[:,0], truth_XY[:,1] - baseline_XY[:,1] * sfs_predict, statistic=resolqt, bins=binnings, range=(0,400))
    plt.figure()
    plt.hlines(bin_resolX_predict, bin_edges[:-1], bin_edges[1:], colors='r', lw=5,
           label='Predict', linestyles='solid')
    plt.hlines(bin_resolX_baseline, bin_edges[:-1], bin_edges[1:], colors='g', lw=5,
           label='Baseline', linestyles='solid')
    plt.legend(loc='lower right')
    plt.xlim(0,400.0)
    plt.ylim(0,200.0)
    plt.xlabel('Truth MET [GeV]')
    plt.ylabel('RespCorr $\sigma$(METX) [GeV]')
    plt.savefig("resolution_metx.png")
    plt.close()
    plt.figure()
    plt.hlines(bin_resolY_predict, bin_edges[:-1], bin_edges[1:], colors='r', lw=5,
           label='Predict', linestyles='solid')
    plt.hlines(bin_resolY_baseline, bin_edges[:-1], bin_edges[1:], colors='g', lw=5,
           label='Baseline', linestyles='solid')
    plt.legend(loc='lower right')
    plt.xlim(0,400.0)
    plt.ylim(0,200.0)
    plt.xlabel('Truth MET [GeV]')
    plt.ylabel('RespCorr $\sigma$(METY) [GeV]')
    plt.savefig("resolution_mety.png")
    plt.close()


def Make1DHists(truth, predict, baseline, xmin=0, xmax=400, nbins=100, density=False, xname="pt [GeV]", yname = "A.U.", outputname="1ddistribution.png"):
    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)
    plt.figure(figsize=(10,8))
    plt.hist(truth,    bins=nbins, range=(xmin, xmax), density=density, histtype='step', facecolor='k', label='Truth')
    plt.hist(predict,  bins=nbins, range=(xmin, xmax), density=density, histtype='step', facecolor='r', label='Predict')
    plt.hist(baseline, bins=nbins, range=(xmin, xmax), density=density, histtype='step', facecolor='g', label='Baseline')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig(outputname)
    plt.close()

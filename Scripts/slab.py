import os
import glob
import cPickle as pickle
import collections
import numpy as np
import scipy as sp
import nibabel as nib
from scipy import stats
from scipy.ndimage import label, generate_binary_structure
import errno

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def LoadPermResults(path,idx):
    with open(os.path.join(path,'perms.pickle')) as f:
        data = pickle.load(f)[idx]
    return data

def SavePermResults(path,*args):
    data = []
    for thing in enumerate(args):
        data.append(thing)
    with open(os.path.join(path,'perms.pickle'),'w') as f:
        pickle.dump(data,f)

def CalculatePermutation(flatdata, design, mask, thresh, i):
    permflatdata = SimpleGLM(flatdata,design[:,i])[0]
    permdata = UnflattenandUnmask(permflatdata,mask)
    cci = ClusterizeImage(permdata,thresh)
    unique, counts = np.unique(cci,return_counts=True)
    return sorted(counts[1:])

def flatten(l):
    for el in l:
        if isinstance(el,collections.Iterable) and not isinstance(el,basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

def ClusterizeImage(image,thresh=None,connectivity=3):
    if thresh is None:
        thresh = 0
    image[np.where(image<=thresh)] = 0
    s = generate_binary_structure(3,connectivity)
    larray, nf = label(image,s)
    return larray

def LoadImageList(path,pattern='*.nii.gz'):
    files = glob.glob(os.path.join(path,pattern))
    data = []
    dim = []
    for i in xrange(0,len(files)):
        tempnii = nib.load(files[i])
        dim = tempnii.shape
        if i==0:
            data = tempnii.get_data()
        elif i==1:
            data = np.stack((data,tempnii.get_data()),axis=3)
        else:
            data = np.concatenate((data[...],tempnii.get_data()[...,np.newaxis]),axis=3)
    return data, dim

def FlattenandMask(data,mask=None):    
    if mask is None:
        mask = np.ones(data.shape[0:3])
    dx = data.shape[0]
    dy = data.shape[1]
    dz = data.shape[2]
    n  = data.shape[3]
    rdata = np.reshape(data,(dx*dy*dz,n))
    rmask = np.reshape(mask,(dx*dy*dz,1))
    return np.transpose(rdata[np.nonzero(rmask)[0],:])

def UnflattenandUnmask(flatdata,mask):
    dx = mask.shape[0]
    dy = mask.shape[1]
    dz = mask.shape[2]
    unmasked = np.zeros((1,dx*dy*dz))
    flatmask = np.reshape(mask,(1,dx*dy*dz))
    unmasked[np.nonzero(flatmask)] = flatdata
    data = np.reshape(unmasked,(dx,dy,dz))
    return data

def SimpleGLM(Y,X=None):
    if X is None:
        X = np.ones((Y.shape[0],1))
    nFeat = Y.shape[1]
    nSub = Y.shape[0]
    nPred = X.shape[1]
    Y = np.matrix(Y)
    X = np.matrix(X)
    b = np.array((X.T*X).I*X.T*Y)
    pred = np.array(X*b)
    res = np.array(Y-pred)
    C = (X.T*X).I
    xvar_inv = np.diag(C)
    xvar_inv = np.tile(xvar_inv,(1,nFeat))
    sse = np.sum(res**2,axis=0)/(nSub-nPred)
    bSE = np.sqrt(xvar_inv * sse)
    t = b / bSE
    return t,b,pred,res


#!/usr/bin/python

import os
import glob
import pickle
import collections
import numpy as np
import scipy as sp
import nibabel as nib
from scipy import stats
from scipy.ndimage import label, generate_binary_structure
import multiprocessing
from functools import partial

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


p = 1
cores = 6
seed = 1234

zthreshes = [2.3,3.1]
Tasks = ['RhymeJudgment','MixedGamblesTask','LivingNonliving','WordObject']
Contrasts = [[1,2,3,4],[1,4],[1,2,3],[1,2,3,4,5,6]]

zthreshes = [3.1]
Tasks = ['RhymeJudgment']
Contrasts = [[1]]

Exp = '/net/pepper/Eklund/'
ResultsFolder = 'GroupAnalyses/randomise/'
OutputFolder1 = '/net/pepper/Eklund/FDR_perms/'
OutputFolder2 = 'perms_py_'

LoadResults = 0

np.random.seed(seed)

for iTask in xrange(0,len(Tasks)):
    for iContrast in xrange(0,len(Contrasts[iTask])):
        for iThresh in xrange(0,len(zthreshes)):
            Task = Tasks[iTask]
            zthresh = zthreshes[iThresh]
            OutputPath = os.path.join(OutputFolder1,Task,'contrast'+str(iContrast+1),OutputFolder2+str(zthresh))
            print(OutputPath)

            InputPath = os.path.join(Exp,Task,ResultsFolder)
            data = LoadImageList(InputPath,'contrast'+str(iContrast+1)+'_0*.nii.gz')[0]
            mask = LoadImageList(InputPath,'contrast'+str(iContrast+1)+'_mask.nii.gz')[0]
            n = data.shape[3]
            #reshape data & mask

            flatdata = FlattenandMask(data,mask)
            
            tthresh = stats.t.ppf(stats.norm.cdf(zthresh),n-1)

            #if loadresults
            if LoadResults==1:
                with open(os.path.join(OutputPath,'perms.pickle')) as f:
                    PermDesign = pickle.load(f)[2]
            else:
                PermDesign = np.matrix(np.sign(np.random.rand(n,p)-0.5))
            
            PermClusters = []

            perms = xrange(0,p)
            pool = multiprocessing.Pool(cores)
            PartialCalculatePermutation = partial(CalculatePermutation,flatdata,PermDesign,mask,tthresh)
            pool.map(PartialCalculatePermutation, iterable)
            pool.close()
            pool.join()


            for i in xrange(0,p):
                #calculate b-values and tstats
                permflatdata = SimpleGLM(flatdata,PermDesign[:,i])[0]
                permdata = UnflattenandUnmask(permflatdata,mask)
                cci = ClusterizeImage(permdata,tthresh)
                unique, counts = np.unique(cci,return_counts=True)
                PermClusters.append(sorted(counts[1:]))

            Clusters = sorted(list(flatten(PermClusters)))
            print(Clusters)

            #save output
            os.mkdir(OutputPath)
            #imgout = nib.Nifti1Image(permdata,np.eye(4))
            #imgout.to_filename(os.path.join(OutputPath,'test.nii.gz'))
            with open(os.path.join(OutputPath,'perms.pickle'),'w') as f:
                pickle.dump([PermClusters,Clusters,PermDesign,zthresh,tthresh,n,p],f)

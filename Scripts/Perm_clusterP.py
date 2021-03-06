#!/usr/bin/python

import slab
import os
import numpy as np
import scipy
from scipy import stats
import multiprocessing
from functools import partial

p = 5000
cores = 3
seed = 1980

zthreshes = [2.3,3.1]
Tasks = ['RhymeJudgment','MixedGamblesTask','LivingNonliving','WordObject']
Contrasts = [[1,2,3,4],[1,4],[1,2,3],[1,2,3,4,5,6]]

Exp = '../Data'
ResultsFolder = 'Contrasts'
OutputFolder1 = '../Results/'
OutputFolder2 = 'perms_py_1980_'

LoadResults = 0

np.random.seed(seed)

for iTask in xrange(0,len(Tasks)):
    for iContrast in xrange(0,len(Contrasts[iTask])):
        for iThresh in xrange(0,len(zthreshes)):
            Task = Tasks[iTask]
            Contrast = Contrasts[iTask][iContrast]
            zthresh = zthreshes[iThresh]
            OutputPath = os.path.join(OutputFolder1,Task,'contrast'+str(Contrast),OutputFolder2+str(zthresh))
            print('Working on '+OutputPath)

            InputPath = os.path.join(Exp,Task,ResultsFolder)
            data = slab.LoadImageList(InputPath,'contrast'+str(Contrast)+'_0*.nii.gz')[0]
            mask = slab.LoadImageList(InputPath,'contrast'+str(Contrast)+'_mask.nii.gz')[0]
            n = data.shape[3]

            #reshape data & mask
            flatdata = slab.FlattenandMask(data,mask)
            
            tthresh = stats.t.ppf(stats.norm.cdf(zthresh),n-1)

            #if you are loading existing PermDesign
            if LoadResults==1:
                PermDesign = np.asmatrix(slab.LoadPermResults(OutputPath,'perms','msgpack',2)[1])
            else:
                PermDesign = np.matrix(np.sign(np.random.rand(n,p)-0.5))
            
            #calculate permutations distributing over cores
            PermClusters = []

            perms = xrange(0,p)
            #for i in perms:
            #    PermClusters.append(slab.CalculateSinglePermutation(flatdata,PermDesign[:,i],mask,tthresh))
            
            pool = multiprocessing.Pool(cores)
            PartialCalculatePermutation = partial(slab.CalculatePermutation,flatdata,PermDesign,mask,tthresh)
            PermClusters = pool.map(PartialCalculatePermutation, perms)
            pool.close()
            pool.join()

            #save a flattened version of all discovered clusters as well
            Clusters = sorted(list(slab.flatten(PermClusters)))

            #save output
            slab.mkdir_p(OutputPath)
            slab.SavePermResults(OutputPath,'perms','msgpack',PermClusters,Clusters,PermDesign.tolist(),zthresh,tthresh,n,p)

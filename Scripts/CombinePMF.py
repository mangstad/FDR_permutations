#!/usr/bin/python

import os
import slab
import numpy as np

zthreshes = [2.3,3.1]
Tasks = ['RhymeJudgment','MixedGamblesTask','LivingNonliving','WordObject']
Contrasts = [[1,2,3,4],[1,4],[1,2,3],[1,2,3,4,5,6]]
Exp = '/net/pepper/Eklund/temp'
OutputFolder1 = '/net/pepper/Eklund/temp/FDR_perms/'
OutputFolder2 = 'perms_py_'

for iThresh in xrange(0,len(zthreshes)):
    for iTask in xrange(0,len(Tasks)):
        for iContrast in xrange(0,len(Contrasts[iTask])):
            Task = Tasks[iTask]
            Contrast = Contrasts[iTask][iContrast]
            zthresh = zthreshes[iThresh]
            OutputPath = os.path.join(OutputFolder1,Task,'contrast'+str(Contrast),OutputFolder2+str(zthresh))
            print(OutputPath)

            PermClusters = slab.LoadPermResults(OutputPath,'perms','msgpack',0)[1]
            
            max = np.max(list(slab.flatten(PermClusters)))
            pmf = np.zeros(max+1)
            
            for iPerm in xrange(0,5000):
                m = len(PermClusters[iPerm]) #number of clusters in perm
                if m==0:
                    pmf[0] = pmf[0] + 1
                else:
                    u,c = np.unique(PermClusters[iPerm],return_counts=True)
                    cs = c / float(m) #scale each count by number of clusters
                    pmf[u] = pmf[u] + cs
                    
            slab.SavePermResults(OutputPath,'PMF','msgpack',pmf.tolist())

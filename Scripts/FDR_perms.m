%make sure SPM is on path

p = 5000;
cores = 6;
seed = 1234;

zthreshes = [2.3 3.1];
Tasks = {
    'RhymeJudgment';
    'MixedGamblesTask';
    'LivingNonliving';
    'WordObject';
    };

Contrasts = {
    [1 2 3 4];
    [1 4];
    [1 2 3];
    [1 2 3 4 5 6];
    };

Exp = '/net/pepper/Eklund/FDR_perms/';
ResultsFolder = 'perms_3_';

LoadResults = 1;

rng(seed);

for iTask = 1:numel(Tasks)
    for iContrast = Contrasts{iTask}
        for iThresh = 1:numel(zthreshes)
            clear permdata permstats rawstats data dataflat mask maskflat maskeddataflat Design;
            clear PermClusters Clusters rawdata rawflat;
            
            Task = Tasks{iTask};
            sNum = sprintf('%d',iContrast);
            zthresh = zthreshes(iThresh);
            sThresh = sprintf('%2.1f',zthresh);
            
            fprintf(1,'Task: %s, Contrast: %d, Threshold: %2.1f, Permutations: %d\n',Task,iContrast,zthresh,p);

            OutputPath = [Exp Task '/contrast' sNum '/' ResultsFolder sThresh];
            mkdir(OutputPath);
            
            tic;
            %load images
            InputPath = [Exp Task '/contrast' sNum];
            
            datafile = spm_select('FPList',InputPath,'.*_0.*\.nii');
            Vf = spm_vol(datafile);
            data = spm_read_vols(Vf);
            
            [dx dy dz n] = size(data);
            dataflat = reshape(data,dx*dy*dz,n)';
            
            maskfile = spm_select('FPList',InputPath,'.*mask.*\.nii');
            Vm = spm_vol(maskfile);
            mask = spm_read_vols(Vm);
            maskflat = reshape(mask,dx*dy*dz,1)';
            
            maskeddataflat = dataflat(:,logical(maskflat));

            tthresh = icdf('t',cdf('norm',zthresh),n-1);

            Design = [ones(n,1)];
            
            if (LoadResults && exist(fullfile(OutputPath,'perms.mat'),'file'))
                load(fullfile(OutputPath,'perms.mat'),'PermDesign');
            else
                PermDesign = sign(rand(n,p)-0.5);
            end
            
            clear permstats
            PermClusters = cell(p,1);
            Clusters = [];
            
            if (matlabpool('size') == 0)
                matlabpool('open',cores);
            end
            
            parfor i = 1:p
                permstats(i) = mc_glm(maskeddataflat,PermDesign(:,i));
                permstats(i).b = [];
                permstats(i).res = [];
                permstats(i).pred = [];
                
                permflat = zeros(1,dx*dy*dz);
                permflat(logical(maskflat)) = permstats(i).t;
                permdata = reshape(permflat,dx,dy,dz);
                [cci num] = spm_bwlabel(double(permdata>tthresh),26);
                clust = sort(crosstab(cci(cci>0)));
                PermClusters{i} = clust;
            end
            
            matlabpool close
            
            Clusters = sort(vertcat(PermClusters{:}));
            
            save(fullfile(OutputPath,'perms.mat'),'PermClusters','Clusters','PermDesign','rawstats','zthresh','tthresh','n','p','dx','dy','dz','maskflat','-v7.3');
            
            toc
            
        end
    end
end

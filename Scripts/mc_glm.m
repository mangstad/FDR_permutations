function [stat] = mc_glm(Y,X)

    nFeat = size(Y,2);
    nSub = size(Y,1);
    nPred = size(X,2);
    stat.b =   pinv(X)*Y;
    stat.pred = X * stat.b;
    stat.res = Y - stat.pred;
    C = pinv(X'*X);
    xvar_inv = diag(C);
    xvar_inv = repmat(xvar_inv,1,nFeat);
    sse = sum(stat.res.^2,1) ./ (nSub - nPred);
    sse = repmat(sse,nPred,1);
    bSE = sqrt(xvar_inv .* sse);
    stat.t = stat.b ./ bSE;

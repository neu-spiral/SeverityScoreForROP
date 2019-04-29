function [alpha,lambda] = selectParameter(AUCMatrix,alphaWeights,lambdaWeights)
% This function is to find the paratemer(alpha, lambda) combination which generate the
% highest AUC.

% Input: 
%       AUCMatrix: length(alphaWeights) by length(lambdaWeights) matrix.
%           Each element contains the AUC generated the corresponding alpha and
%           lambda.
%       alphaWeights: 1 by nAlpha matrix contains the value of alpha.
%       lambdaWeights: 1 by nLambda cell matrix contains the value of lambda.
% Output:
%       alpha: the alpha value which generated the higheset AUC.
%       lambda: the lambda value which generated the hightest AUC.

[maxAUC,~] = max(AUCMatrix(:));
[alphaInd,lambdaInd] = find(AUCMatrix==maxAUC);
alpha = alphaWeights(alphaInd);
lambda = lambdaWeights(lambdaInd);
end
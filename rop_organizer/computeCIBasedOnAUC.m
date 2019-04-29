function [AUCSE,AUCCI]=computeCIBasedOnAUC(AUC,labels,alpha)
%% Standard Deviation of AUC

nPoss = sum(labels);
nNegs = sum(~labels);
% .. assuming exponential distributions for scores
pxxy = AUC / (2 - AUC);
pxyy = (2*AUC^2) / (1+AUC);
%
AUCSE = sqrt((AUC*(1-AUC) + (nPoss-1)*(pxxy-AUC^2) + ...
    (nNegs-1)*(pxyy-AUC^2)) / (nPoss*nNegs));

%% Confidence Intervals for AUC

% transferring with logit function, so that the output interval would be
% zero and one
logitAUC = log(AUC / (1-AUC));
% bounds for the logit
lowerBound = logitAUC - norminv(1-alpha/2,0,1)*AUCSE/(AUC*(1-AUC));
upperBound = logitAUC + norminv(1-alpha/2,0,1)*AUCSE/(AUC*(1-AUC));
% transfer back to AUC
AUCCI = zeros(2,1);
AUCCI(1) = exp(lowerBound) / (1+exp(lowerBound));
AUCCI(2) = exp(upperBound) / (1+exp(upperBound));
end
function [absAUC, cmpAUC] = get_auc(model,loss)


inputFileName = ['../../../result/netflix/',model,'/',loss,'/','netflix__',loss,'_'];
figSave = ['../fig/netflix','_',loss];
isSaveFig = 1;
xAxis = 0:0.1:1;
legendStr = {};
xLabel = 'Weight \alpha on Absolute Label Data';

alphaValue = 0.0:0.1:1.0;
lambdaStr = {'10000.0' '1000.0' '100.0' '10.0' '9.0' '7.0' '5.0' '3.0' '1.0' '0.9' '0.8' '0.7' '0.6' '0.5' '0.4' '0.3' '0.2' '0.1' '0.01' '0.001' '0.0001' '1e-05' '1e-06' };
CIParameter = 0.10;
lambdaValue = cellfun(@str2num,lambdaStr(1:end));
numOfAlphaValue = size(alphaValue,2);
numOfLambdaValue = size(lambdaValue,2); 

absAUC = zeros(numOfAlphaValue,numOfLambdaValue);
cmpAUC = zeros(numOfAlphaValue,numOfLambdaValue);

for alphaInd = 1:numOfAlphaValue
    for lambdaInd = 1:numOfLambdaValue
        try
        file = load([inputFileName, num2str(alphaValue(alphaInd),'%.1f'), '_',lambdaStr{lambdaInd},'.mat']);
        absAUC(alphaInd, lambdaInd) = mean(file.abs_auc,1);
        cmpAUC(alphaInd, lambdaInd) = mean(file.cmp_auc,1);
        catch
        end
    end
end

% bestAbsAUC(:,1) = max(absAUC,[],2);
% bestCmpAUC(:,1) = max(absAUC,[],2);
% if strcmp(dataset, 'happy')
%     nAbsPoss = 488;
%     nAbsNeg = 883;
%     nCmpPoss = 4823;
%     nCmpNeg = 4794;
% elseif strcmp('dataset','pleasure')
%     nAbsPoss = 488;
%     nAbsNeg = 883;
%     nCmpPoss = 3944;
%     nCmpNeg =  4056;
% end

% for countAlphaInd = 1:numOfAlphaValue
%     [bestAbsAUC(countAlphaInd,2),~] = computeCIBasedOnAUCNegPos(bestAbsAUC(countAlphaInd,1),CIParameter,nAbsPoss,nAbsNeg);
%     [bestCmpAUC(countAlphaInd,2),~] = computeCIBasedOnAUCNegPos(bestCmpAUC(countAlphaInd,1),CIParameter,nCmpPoss,nCmpNeg);
% end
%     bestAbsAUC(countAlphaInd,2) = 1.96*bestAbsAUC(countAlphaInd,2);
%     bestCmpAUC(countAlphaInd,2) = 1.96*bestCmpAUC(countAlphaInd,2);


    
end

loss = 'SVMSVM';
inputFileName = ['../../../result/FAC',loss,'/FAC_','_',loss,'_'];
figSave = ['../../../pic/FAC','_',loss];
isSaveFig = 1;
xAxis = 0:0.1:1;
legendStr = {};
xLabel = 'Weight \alpha on Class Label Data';

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

bestAbsAUC(:,1) = max(absAUC,[],2);
bestCmpAUC(:,1) = max(cmpAUC,[],2);

nAbsPoss = 488;
nAbsNeg = 883;
nCmpPoss = 4823;
nCmpNeg = 4794;


for countAlphaInd = 1:numOfAlphaValue
    bestAbsAUC(countAlphaInd,2) = computeCIBasedOnAUCNegPos(bestAbsAUC(countAlphaInd,1),CIParameter,nAbsPoss,nAbsNeg);
    bestCmpAUC(countAlphaInd,2) = computeCIBasedOnAUCNegPos(bestCmpAUC(countAlphaInd,1),CIParameter,nCmpPoss,nCmpNeg);
end
bestAbsAUC(:,2) = 1.96*bestAbsAUC(:,2);
bestCmpAUC(:,2) = 1.96*bestCmpAUC(:,2);


fig_abs = plotAUCCI(bestAbsAUC(:,1),bestAbsAUC(:,2),'xLabel',xLabel,...
    'yLabel','AUC on Class Labels','legendStr',legendStr,'isSaveFig',isSaveFig,...
    'figName',[figSave,'_abs_'],'xAxis',xAxis,'titleStr',{},'figLowerBound',0.6,'figUpperBound',0.85);

fig_cmp = plotAUCCI(bestCmpAUC(:,1),bestCmpAUC(:,2),'xLabel',xLabel,...
    'yLabel','AUC on Comparison Labels','legendStr',legendStr,'isSaveFig',isSaveFig,...
    'figName',[figSave,'_cmp_'],'xAxis',xAxis,'titleStr',{},'figLowerBound',0.6,'figUpperBound',0.85);




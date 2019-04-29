
loss = 'LogLog';
models = {'globalModel','globalModelExpertBias','expertModel'};

[absAucGM, cmpAucGM] = get_auc('globalModel',loss);
[absAucGMEB, cmpAucGMEB] = get_auc('globalModelExpertBias',loss);
[absAucEM, cmpAucEM] = get_auc('expertModel',loss);
legendStr = {'GM','GMEB','EM'};
xAxis = 0:0.1:1;
isSaveFig = 1;


bestAbsAucGM(:,1) = max(absAucGM,[],2);
bestCmpAucGM(:,1) = max(cmpAucGM,[],2);
bestAbsAucGMEB(:,1) = max(absAucGMEB,[],2);
bestCmpAucGMEB(:,1) = max(cmpAucGMEB,[],2);
bestAbsAucEM(:,1) = max(absAucEM,[],2);
bestCmpAucEM(:,1) = max(cmpAucEM,[],2);

nAbsPoss = 5374;
nAbsNeg = 3037;
nCmpPoss = 42783;
nCmpNeg = 43874;

xLabel = 'Weight \alpha on Class Label Data';
figSave = ['../../../figSource/netflix/netflix','_',loss];
numOfAlphaValue = length(xAxis);
CIParameter = 0.1;
for countAlphaInd = 1:numOfAlphaValue
    bestAbsAucGM(countAlphaInd,2) = computeCIBasedOnAUCNegPos(bestAbsAucGM(countAlphaInd,1),CIParameter,nAbsPoss,nAbsNeg);
    bestCmpAucGM(countAlphaInd,2) = computeCIBasedOnAUCNegPos(bestCmpAucGM(countAlphaInd,1),CIParameter,nCmpPoss,nCmpNeg);
    bestAbsAucGMEB(countAlphaInd,2) = computeCIBasedOnAUCNegPos(bestAbsAucGMEB(countAlphaInd,1),CIParameter,nAbsPoss,nAbsNeg);
    bestCmpAucGMEB(countAlphaInd,2) = computeCIBasedOnAUCNegPos(bestCmpAucGMEB(countAlphaInd,1),CIParameter,nCmpPoss,nCmpNeg);
    bestAbsAucEM(countAlphaInd,2) = computeCIBasedOnAUCNegPos(bestAbsAucEM(countAlphaInd,1),CIParameter,nAbsPoss,nAbsNeg);
    bestCmpAucEM(countAlphaInd,2) = computeCIBasedOnAUCNegPos(bestCmpAucEM(countAlphaInd,1),CIParameter,nCmpPoss,nCmpNeg);
end
    bestAbsAucGM(countAlphaInd,2) = 1.96*bestAbsAucGM(countAlphaInd,2);
    bestCmpAucGM(countAlphaInd,2) = 1.96*bestCmpAucGM(countAlphaInd,2);
    bestAbsAucGMEB(countAlphaInd,2) = 1.96*bestAbsAucGMEB(countAlphaInd,2);
    bestCmpAucGMEB(countAlphaInd,2) = 1.96*bestCmpAucGMEB(countAlphaInd,2);
    bestAbsAucEM(countAlphaInd,2) = 1.96*bestAbsAucEM(countAlphaInd,2);
    bestCmpAucEM(countAlphaInd,2) = 1.96*bestCmpAucEM(countAlphaInd,2);

bestAbsAUC = [bestAbsAucGM(:,1),bestAbsAucGMEB(:,1),bestAbsAucEM(:,1)];
bestAbsAUCCI = [bestAbsAucGM(:,2),bestAbsAucGMEB(:,2),bestAbsAucEM(:,2)];
bestCmpAUC = [bestCmpAucGM(:,1),bestCmpAucGMEB(:,1),bestCmpAucEM(:,1)];
bestCmpAUCCI = [bestCmpAucGM(:,2),bestCmpAucGMEB(:,2),bestCmpAucEM(:,2)];

fig_abs = plotAUCCI(bestAbsAUC,bestAbsAUCCI,'xLabel',xLabel,...
    'yLabel','AUC on Class Labels','legendStr',legendStr,'isSaveFig',isSaveFig,...
    'figName',[figSave,'_abs_'],'xAxis',xAxis,'titleStr',{},'figLowerBound',0.5,'figUpperBound',1);

fig_cmp = plotAUCCI(bestCmpAUC,bestCmpAUCCI,'xLabel',xLabel,...
    'yLabel','AUC on Comparison Labels','legendStr',legendStr,'isSaveFig',isSaveFig,...
    'figName',[figSave,'_cmp_'],'xAxis',xAxis,'titleStr',{},'figLowerBound',0.5,'figUpperBound',1);
save([figSave,'.mat']);

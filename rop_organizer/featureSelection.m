featureFile = load('featureSelection.mat');
featsNameFile = load('../../Data/FeatureExtraction/Manual/SetOf100/featsName.mat');
fileName = '../../Data/Result/L1/RSD&Bias2RSD&Exp13_L1_CV1_manual_';
featsNameOrigin = featsNameFile.featsName;
[namesN,~] = size(featsNameOrigin);
featsName = cell(namesN-1,1);
d = 156;
gapValue = 200;
numOfFeats = 20;
for i = 2 : namesN
featsName{i-1,:} = char(strcat(char(featsNameOrigin{i,2}), {'-'}, char(featsNameOrigin{i,3})));
end
xAxisRange = 1 : gapValue:  numOfFeats*gapValue;


%% 1 Plus Predict RSD Here training Exp13Bias
% figure('rend','painters','pos',[-10 -200 900 600])
alphaExp132RSDPlus = 0.0;
lambdaExp132RSDPlus = 100;
betaExp132RSDPlusNoB = featureFile.betaExp132RSDPlusNoB;
[~,betaExp132RSDPlusNoBInd] = sort(abs(betaExp132RSDPlusNoB),'descend');
betaExp132RSDPlusNoBInd = betaExp132RSDPlusNoBInd(1:numOfFeats);
title = [featsName(betaExp132RSDPlusNoBInd,1),num2cell(betaExp132RSDPlusNoB(betaExp132RSDPlusNoBInd,1))];
bar(xAxisRange,abs(cell2mat(title(:,2))),0.1);
a = title(:,1);
set(gca,'XTickLabel',a,'XTick',xAxisRange);
fileExp132RSDPlus = load([fileName, num2str(alphaExp132RSDPlus,'%.1f'), '_',num2str(lambdaExp132RSDPlus,'%.1f'),'.mat']);
betaAllExp132RSDPlus = fileExp132RSDPlus.betaValueExp13Plus{1,1};
[~,NCV] = size(betaAllExp132RSDPlus);
diff = (sum(sqrt(sum((betaAllExp132RSDPlus(1:d,:)-repmat(betaExp132RSDPlusNoB,1,NCV)).^2,1))))./NCV./sqrt(sum(betaExp132RSDPlusNoB.^2))

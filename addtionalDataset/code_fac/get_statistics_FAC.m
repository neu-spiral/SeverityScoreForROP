



close all;
[maxAbsGM,indAbsGM] = max(bestAbsAUC(:,1),[],1);
[maxCmpGM,indCmpGM] = max(bestCmpAUC(:,1),[],1);
disp(loss)
disp(['Absolute ',num2str(maxAbsGM), ' Std ',num2str(bestAbsAUC(indAbsGM,2)), ' Alpha ',num2str(xAxis(indAbsGM))])
disp(['Comparison ',num2str(maxCmpGM), ' Std ',num2str(bestCmpAUC(indCmpGM,2)), ' Alpha ',num2str(xAxis(indCmpGM))])
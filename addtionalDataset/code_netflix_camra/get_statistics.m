



close all;
[maxAbsGM,indAbsGM] = max(bestAbsAucGM(:,1),[],1);
[maxCmpGM,indCmpGM] = max(bestCmpAucGM(:,1),[],1);
disp('GM')
disp(['Absolute ',num2str(maxAbsGM), ' Std ',num2str(bestAbsAucGM(indAbsGM,2)), ' Alpha ',num2str(xAxis(indAbsGM))])
disp(['Comparison ',num2str(maxCmpGM), ' Std ',num2str(bestCmpAucGM(indCmpGM,2)), ' Alpha ',num2str(xAxis(indCmpGM))])

[maxAbsGMEB,indAbsGMEB] = max(bestAbsAucGMEB(:,1),[],1);
[maxCmpGMEB,indCmpGMEB] = max(bestCmpAucGMEB(:,1),[],1);
disp('GMEB')
disp(['Absolute ',num2str(maxAbsGMEB), ' Std ',num2str(bestAbsAucGM(indAbsGMEB,2)), ' Alpha ',num2str(xAxis(indAbsGMEB))])
disp(['Comparison ',num2str(maxCmpGMEB), ' Std ',num2str(bestCmpAucGM(indCmpGMEB,2)), ' Alpha ',num2str(xAxis(indCmpGMEB))])

[maxAbsEM,indAbsEM] = max(bestAbsAucEM(:,1),[],1);
[maxCmpEM,indCmpEM] = max(bestCmpAucEM(:,1),[],1);
disp('EM')
disp(['Absolute ',num2str(maxAbsEM), ' Std ',num2str(bestAbsAucEM(indAbsEM,2)), ' Alpha ',num2str(xAxis(indAbsEM))])
disp(['Comparison ',num2str(maxCmpEM), ' Std ',num2str(bestCmpAucEM(indCmpEM,2)), ' Alpha ',num2str(xAxis(indCmpEM))])
clear all;
close all;
%% Figure Settings
axisRange = ([0,100,0,100]);
xStickSize = 30;
legendSize = 25;
xTickNumSize =20;

%% Read the csv File
fileContent = readtable('../../data/ropData/set100allusersElo.csv');
labelImgElo = table2array(fileContent(:,4));
indLabelPlus = find(strcmp(labelImgElo,'Plus'));
indLabelPreP =  find(strcmp(labelImgElo,'Pre'));
indLabelNormal =  find(strcmp(labelImgElo,'normal'));
imgAllElo = table2array(fileContent(:,6));
eloExp1 = table2array(fileContent(:,7));
eloExp2 = table2array(fileContent(:,8));
eloExp3 = table2array(fileContent(:,9));
eloExp4 = table2array(fileContent(:,10));
eloExp5 = table2array(fileContent(:,11));
rankExp1 = tiedrank(eloExp1);
rankExp2 = tiedrank(eloExp2);
rankExp3 = tiedrank(eloExp3);
rankExp4 = tiedrank(eloExp4);
rankExp5 = tiedrank(eloExp5);
rankAll = [rankExp1,rankExp2,rankExp3,rankExp4,rankExp5];
ccAll = [];
for i = 1:5
    for j = 1:5
        if i==j
            continue
        end
        ccAll = [ccAll, correlation(rankAll(:,i),rankAll(:,j))]; 
    end
end
ccAve = mean(ccAll);
ccStd = std(ccAll);
%% Draw and save the figure;
fig31 = figure();
hold on;
plot(rankExp3(indLabelPlus),rankExp1(indLabelPlus),'r.','Marker','o','MarkerFaceColor','r');
plot(rankExp3(indLabelPreP),rankExp1(indLabelPreP),'g.','Marker','s','MarkerFaceColor','g');
plot(rankExp3(indLabelNormal),rankExp1(indLabelNormal),'b.','Marker','^','MarkerFaceColor','b');
ax = gca;
ax.FontSize = xTickNumSize; 
xlabel('Expert 3 Rank','FontSize',xStickSize,'FontWeight','bold')
ylabel('Expert 1 Rank','FontSize',xStickSize,'FontWeight','bold')
% legend({'Plus','Pre-Plus','Normal'},'Location','Northwest','FontSize',legendSize);
set(fig31,'Units','Inches');
pos = get(fig31,'Position');
set(fig31, 'paperPositionMode', 'Auto', 'PaperUnits','Inches','PaperSize',[pos(3),pos(4)]);
print(fig31,'../../pic/Exp31Rank.pdf','-dpdf','-r0');

fig25 = figure();
hold on;
plot(rankExp2(indLabelPlus),rankExp5(indLabelPlus),'r.','Marker','o','MarkerFaceColor','r');
plot(rankExp2(indLabelPreP),rankExp5(indLabelPreP),'g.','Marker','s','MarkerFaceColor','g');
plot(rankExp2(indLabelNormal),rankExp5(indLabelNormal),'b.','Marker','^','MarkerFaceColor','b');
ax = gca;
ax.FontSize = xTickNumSize; 
xlabel('Expert 2 Rank','FontSize',xStickSize,'FontWeight','bold')
ylabel('Expert 5 Rank','FontSize',xStickSize,'FontWeight','bold')
legend({'Plus','Pre-Plus','Normal'},'Location','Northwest','FontSize',legendSize);
set(fig25,'Units','Inches');
pos = get(fig25,'Position');
set(fig25, 'paperPositionMode', 'Auto', 'PaperUnits','Inches','PaperSize',[pos(3),pos(4)]);
print(fig25,'../../pic/Exp25Rank.pdf','-dpdf','-r0');
% 
% fig24 = figure();
% hold on;
% plot(rankExp2(indLabelPlus),rankExp4(indLabelPlus),'r.','Marker','o','MarkerFaceColor','r');
% plot(rankExp2(indLabelPreP),rankExp4(indLabelPreP),'g.','Marker','s','MarkerFaceColor','g');
% plot(rankExp2(indLabelNormal),rankExp4(indLabelNormal),'b.','Marker','^','MarkerFaceColor','b');
% xlabel('Expert 2 Rank','FontSize',xStickSize,'FontWeight','bold')
% ylabel('Expert 4 Rank','FontSize',xStickSize,'FontWeight','bold')
% % legend({'Plus','Pre-Plus','Normal'},'Location','Northwest','FontSize',legendSize);
% set(fig24,'Units','Inches');
% pos = get(fig24,'Position');
% set(fig24, 'paperPositionMode', 'Auto', 'PaperUnits','Inches','PaperSize',[pos(3),pos(4)]);
% print(fig24,'Exp24Rank.pdf','-dpdf','-r0');

% fig21 = figure();
% hold on;
% plot(rankExp2(indLabelPlus),rankExp1(indLabelPlus),'r.','Marker','o','MarkerFaceColor','r');
% plot(rankExp2(indLabelPreP),rankExp1(indLabelPreP),'g.','Marker','s','MarkerFaceColor','g');
% plot(rankExp2(indLabelNormal),rankExp1(indLabelNormal),'b.','Marker','^','MarkerFaceColor','b');
% xlabel('Expert 2 Rank','FontSize',xStickSize,'FontWeight','bold')
% ylabel('Expert 1 Rank','FontSize',xStickSize,'FontWeight','bold')
% legend({'Plus','Pre-Plus','Normal'},'Location','Northwest','FontSize',legendSize);
% set(fig21,'Units','Inches');
% pos = get(fig21,'Position');
% set(fig21, 'paperPositionMode', 'Auto', 'PaperUnits','Inches','PaperSize',[pos(3),pos(4)]);
% print(fig21,'Exp21Rank.pdf','-dpdf','-r0');
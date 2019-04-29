function fig = plotAUCCI_numComparison(AUC, AUCCI, varargin)
% This function would plot mulitple AUC lines and return the An AUC figure with confidence
% interval. This function is able to draw 4 lines at most.
% Input:
%       - AUC: n by m matrix. m columns represents n AUC lines. n rows represent n
%       points on that line.
%       - AUCSTD: n by m matrix. size must be exact same with AUC matrix.
%       The value in AUCCI is the confidence value for the corresponding
%       AUC value in AUC matrix. The auc and its confidence iterval length
%       must be put at the same location in AUC and AUCCI matrix.
%       - isSaveFig: binary value. if saveFig=1, this code would save a pdf
%       figure with the figName specified below.
%       - figName: string. specified the pdf figName and do not need put '.pdf' in the string. default, 'fig'       
%       - legendStr: 1 by n cell matrix. i-th element represent the legend
%       for the i-th row in AUC matrix.
%       - titleStr: String for the figure title.
%       - legendLocation: String. Please following the matlab plot location
%       option.
%       - xLabel: String: x-axis label for figure;
%       - yLabel: String: y-axis label for figure;
%       - figLowerBound: number. the minimum y-axis value for figure.
%       - figUpperBound: number. the maximum y-axis value for figure.
%       - lineWidth: number. The width for the line.
%       - markerSize: number. The size for the marker.
%       - fontSize: number. The size for xlabel, ylabel and title fonts.
%       - stickFont: number. The size for x-axis and y-axis fonts.
%       - transparentFacotr: float number in [0,1], higher value, less
%       transparent.
%
% Example Usage: 
%       PlotAUCCI(AUC,AUCCI,'saveFig',1,'figName','figExample') would have a figure without
%       xlabel,ylabel and title. It would saved as 'figExample.pdf'.
% Author : Peng Tian
% Date: Feb 2017

args = inputParser;
addOptional(args, 'xAxis',1:size(AUC,1));
addOptional(args,'legendStr','None');
addParameter(args, 'legendLocation','SouthWest');
addOptional(args,'titleStr','None');
addOptional(args,'xLabel','None');
addOptional(args,'yLabel','None');
addOptional(args,'isSaveFig',0);
addOptional(args,'figName','fig')
addParameter(args,'figLowerBound',0.5);
addParameter(args,'figUpperBound',1.0);
addParameter(args,'lineWidth',2);
addParameter(args,'markerSize',8);
addParameter(args,'fontSize',20);
addParameter(args,'stickFontSize',15);
addParameter(args,'transparentFactor',0.2);
parse(args,varargin{:});
numOfLines = size(AUC,2);
if numOfLines>4
    error('This function is only able to draw 4 lines at most');
end
figAxisRange = [args.Results.xAxis(1),args.Results.xAxis(end),args.Results.figLowerBound,args.Results.figUpperBound,];
colorSpace = {[0.12572087695201239, 0.47323337360924367, 0.707327968232772],...
              [0.21171857311445125, 0.63326415104024547, 0.1812226118410335],...
              [0.89059593116535862, 0.10449827132271793, 0.11108035462744099],...
              [0.99990772780250103, 0.50099192647372981, 0.0051211073118098693]};   
LineStyleSet = {'-', '--', ':', '-.'};
MarkerSet = {'o', 's', '^', 'd'};
fig = figure();        
hold on;
args.Results.xAxis
for lineCount=1:numOfLines
plot(args.Results.xAxis, AUC(:,lineCount),'color',colorSpace{lineCount},...
    'LineStyle',LineStyleSet{lineCount},'LineWidth',args.Results.lineWidth,...
    'Marker',MarkerSet{lineCount},'MarkerFaceColor',colorSpace{lineCount},...
    'MarkerEdgeColor',colorSpace{lineCount},'MarkerSize',args.Results.markerSize);
end
for fillCount=1:numOfLines
fill([args.Results.xAxis, wrev(args.Results.xAxis)],...
    [AUC(:,fillCount)'+AUCCI(:,fillCount)' wrev(AUC(:,fillCount)'-AUCCI(:,fillCount)')],...
    colorSpace{fillCount},'FaceAlpha',args.Results.transparentFactor,'EdgeColor','none');
end
if ~strcmp(args.Results.titleStr,'None')
    title(args.Results.titleStr,'FontSize',args.Results.fontSize,'FontWeight','Bold')
end
if ~strcmp(args.Results.xLabel,'None')
    xlabel(args.Results.xLabel,'FontSize',args.Results.fontSize,'FontWeight','Bold')
end
if ~strcmp(args.Results.yLabel,'None')
    ylabel(args.Results.yLabel,'FontSize',args.Results.fontSize,'FontWeight','Bold')
end
if ~strcmp(args.Results.legendStr,'None')
    legend(args.Results.legendStr,'FontSize',args.Results.fontSize,'Location',args.Results.legendLocation)
end
set(gca, 'FontSize', args.Results.stickFontSize)
set(gca,'XTick',[20,100,200,1000,3000]);
xlabel('Number of Trained Comparisons','FontSize',20,'FontWeight','Bold');
ylabel('AUC','FontSize',20,'FontWeight','Bold');
set(gca, 'FontSize', 15);
axis([0,3000,0.5,1]);
set(fig,'Units','Inches');
set(gca,'XScale','log');
% axis(figAxisRange);
set(fig,'Units','Inches');
pos = get(fig,'Position');
set(fig, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
if args.Results.isSaveFig==1
    print(fig, [args.Results.figName '.pdf'],'-dpdf','-r0');

end
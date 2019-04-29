function rank = rankScore(RankValue)
[~,indx]=sort(RankValue,'ascend');
N = length(RankValue);
rank=[];
for j=1:N
rank_i=find(indx==j);
rank=[rank;rank_i];
end
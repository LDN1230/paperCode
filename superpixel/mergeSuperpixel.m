function mergedSuperpixel = mergeSuperpixel(originalSuperpixel,feature)
% 超像素融合
% originalSuperpixel: 原来的超像素map
% feature: 超像素块融合需要的特征，用于G-static计算
    beforeMergedSuperpixel = originalSuperpixel;
    gamma = 0.3; 
    nMerge = 50;
    for n = 1:nMerge
        % 判断合并前超像素map各个超像素的邻超像素
        nSuperpixel = length(unique(beforeMergedSuperpixel(:))); % 超像素的个数
        isNeighbor = zeros(nSuperpixel, nSuperpixel); % 用来标记各超像素点是否相邻，1为相邻，0为不相邻
        superpixelCell = cell(1,nSuperpixel); % superpixelCell.index:超像素内各点的索引；superpixelCell:neighbors:各点的邻域
        for i = 1:nSuperpixel
            index = find(beforeMergedSuperpixel==i);
            superpixelCell{i}.index = index;
            neighbors = findNeighbor(beforeMergedSuperpixel, index);
            superpixelCell{i}.neighbors = neighbors;
        end
     % 找出相邻的超像素  

        superpixelInfo = []; % col1:超像素1的标签；col2:超像素2的标签；col3:共享边界的长度；col4:它们的距离；codl5:合并的cost
        for i = 1:nSuperpixel
            neighbors = superpixelCell{i}.neighbors; 
            neighbors = neighbors(:,2:end); % 当前超像素的邻域点
            for j = (i+1):nSuperpixel
                index2 = superpixelCell{j}.index; % 其他超像素的点
                intersection = intersect(neighbors, index2);
                sharedBoundary = length(intersection);
                if isempty(intersection) == 0  % 有交集则是相邻的
                    isNeighbor(i,j) = 1;
                    tempSuperpixelInfo = zeros(1,5);
                    tempSuperpixelInfo(1,1:3) = [i j sharedBoundary];
                    % 计算两超像素的距离
                    index1 = superpixelCell{i}.index;
                    f1 = feature(index1);
                    f2 = feature(index2);
                    h1 = hist(f1,8);
                    p1 = h1 / sum(h1);
                    h2 = hist(f2,8);
                    p2 = h2 / sum(h2);
                    p3 = p1 + p2;
                    G12 = sum(p1.*log(p1))+sum(p2.*log(p2)) + 2*log(2) - sum(p3.*log(p3));
                    distance = G12/(gamma*sharedBoundary);
                    tempSuperpixelInfo(1,4) = distance;
                    % 计算合并的cost
                    s1 = length(index1);
                    s2 = length(index2);
                    cost = (s1*s2)*distance/(s1+s2);
                    tempSuperpixelInfo(1,5) = cost;
                    superpixelInfo = [superpixelInfo; tempSuperpixelInfo];
                end
            end
        end
        % 将合并cost最小的两个区域合并
        [minCost, index3] = min(superpixelInfo(:,5));
        label1 = superpixelInfo(index3,1);
        label2 = superpixelInfo(index3,2);
        beforeMergedSuperpixel(beforeMergedSuperpixel==label2) = label1;
    end
    mergedSuperpixel = beforeMergedSuperpixel;
    

    
    
    
end
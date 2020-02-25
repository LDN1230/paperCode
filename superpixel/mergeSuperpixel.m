function mergedSuperpixel = mergeSuperpixel(originalSuperpixel,feature)
% 超像素融合
% originalSuperpixel: 原来的超像素map
% feature: 超像素块融合需要的特征，用于G-static计算
    beforeMergedSuperpixel = originalSuperpixel;
    gamma = 0.3; 
    nMerge = 200;
    idd = [];
    for n = 1:nMerge
        fprintf('超像素融合中:%d/%d......', n, nMerge);
        t1 = cputime;
        if n == 1 % 第一次合并，计算各超像素的索引和邻域
            % 判断合并前超像素map各个超像素的邻超像素
            nSuperpixel = length(unique(beforeMergedSuperpixel(:))); % 超像素的个数
            isNeighbor = zeros(nSuperpixel, nSuperpixel); % 用来标记各超像素点是否相邻，1为相邻，0为不相邻
            superpixelCell = cell(1,nSuperpixel); % superpixelCell.index:超像素内各点的索引；superpixelCell:neighbors:各点的邻域
            superpixel_label = [1:nSuperpixel];
            for i = 1:length(superpixel_label)
                ii = superpixel_label(i);
                index = find(beforeMergedSuperpixel==ii);
                superpixelCell{ii}.index = index;
                neighbors = findNeighbor(beforeMergedSuperpixel, index);
                superpixelCell{ii}.neighbors = neighbors;
            end
       

         % 找出相邻的超像素  
            superpixelInfo = []; % col1:超像素1的标签；col2:超像素2的标签；col3:共享边界的长度；col4:它们的距离；codl5:合并的cost
            for i = 1:length(superpixel_label)
                 ii = superpixel_label(i);
                neighbors = superpixelCell{ii}.neighbors; 
                neighbors = neighbors(:,2:end); % 当前超像素的邻域点
                neighbors = neighbors(:);
                neighbors(neighbors==0) = []; %把用0填充的点清点
                for j = (i+1):length(superpixel_label)
                    jj = superpixel_label(j);
                    index2 = superpixelCell{jj}.index; % 其他超像素的点
                    intersection = intersect(neighbors, index2);
                    sharedBoundary = length(intersection);
                    if isempty(intersection) == 0  % 有交集则是相邻的
                        isNeighbor(i,j) = 1;
                        tempSuperpixelInfo = zeros(1,5);
                        tempSuperpixelInfo(1,1:3) = [ii jj sharedBoundary];
                        % 计算两超像素的距离
                        index1 = superpixelCell{ii}.index;
                        f1 = feature(index1);
                        f2 = feature(index2);
                        h1 = hist(f1,5);
                        p1 = h1 / sum(h1);
                        h2 = hist(f2,5);
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
        end
        
        if n >1
             index4 = [find(superpixelInfo(:,1)==label1); find(superpixelInfo(:,2)==label1);
                               find(superpixelInfo(:,1)==label2); find(superpixelInfo(:,2)==label2) ]; % 找出superpixelInfo存在上一次合并的超像素的行
             label3 = label1;
             index5 = superpixelCell{label3}.index;
             for gg = 1:length(index4)
                 gg1 = index4(gg);
                 if superpixelInfo(gg1,1) == label2
                        superpixelInfo(gg1,1) = label1;
                 end
                  if superpixelInfo(gg1,2) == label2
                        superpixelInfo(gg1,2) = label1;
                 end
                 if superpixelInfo(gg1, 1) == label1
                    label4 = superpixelInfo(gg1,2);
                 else 
                     label4 = superpixelInfo(gg1,1);
                 end
                neighbors = superpixelCell{label4}.neighbors;
                neighbors = neighbors(:,2:end); % 当前超像素的邻域点
                neighbors = neighbors(:);
                neighbors(neighbors==0) = []; %把用0填充的点清点
                intersection = intersect(neighbors, index5);
                sharedBoundary = length(intersection);
                if isempty(intersection) == 0  % 有交集则是相邻的
                       isNeighbor(label3,label4) = 1;
                       tempSuperpixelInfo = zeros(1,5);
                       if label3 < label4
                            tempSuperpixelInfo(1,1:3) = [label3 label4 sharedBoundary];
                       else
                           tempSuperpixelInfo(1,1:3) = [label4 label3 sharedBoundary];
                       end
                end
                 % 计算两超像素的距离
                 index6 = superpixelCell{label4}.index;
                 f1 = feature(index5);
                 f2 = feature(index6);
                 h1 = hist(f1,5);
                 p1 = h1 / sum(h1);
                 h2 = hist(f2,5);
                 p2 = h2 / sum(h2);
                 p3 = p1 + p2;
                 G12 = sum(p1.*log(p1))+sum(p2.*log(p2)) + 2*log(2) - sum(p3.*log(p3));
                 distance = G12/(gamma*sharedBoundary);
                 tempSuperpixelInfo(1,4) = distance;
                 % 计算合并的cost
                 s1 = length(index5);
                 s2 = length(index6);
                 cost = (s1*s2)*distance/(s1+s2);
                 tempSuperpixelInfo(1,5) = cost;
                 superpixelInfo(index4(gg),:)  =  tempSuperpixelInfo;
             end
        end
        
        
        % 将合并cost最小的两个区域合并
        [minCost, index3] = min(superpixelInfo(:,5));
        label1 = superpixelInfo(index3,1);
        label2 = superpixelInfo(index3,2);
        superpixelInfo(index3,:) = []; % 清掉合并的行
        beforeMergedSuperpixel(beforeMergedSuperpixel==label2) = label1; % 替换被合并超像素的标签
        id = find(superpixel_label==label2);
        if isempty(id)
            fprintf('error!!!!!!!!!!!!!!!!!!!!');
        end
        superpixel_label(id) = []; % 将合并的超像素对应的标签删除
        superpixelCell{label1}.index = [superpixelCell{label1}.index; superpixelCell{label2}.index]; % 合并的超像素索引也合并
        neighbors = findNeighbor(beforeMergedSuperpixel, superpixelCell{label1}.index);
        superpixelCell{label1}.neighbors = neighbors ; % 重新计算合并后超像素的邻域
        %superpixelInfo()
        t2 = cputime-t1;
        fprintf('用时%.2fs\n', t2);
        
       

                
                
    end
    mergedSuperpixel = beforeMergedSuperpixel;
    

    
    
    
end
function mergedSuperpixel = mergeSuperpixel(originalSuperpixel,feature)
% �������ں�
% originalSuperpixel: ԭ���ĳ�����map
% feature: �����ؿ��ں���Ҫ������������G-static����
    beforeMergedSuperpixel = originalSuperpixel;
    gamma = 0.3; 
    nMerge = 50;
    for n = 1:nMerge
        % �жϺϲ�ǰ������map���������ص��ڳ�����
        nSuperpixel = length(unique(beforeMergedSuperpixel(:))); % �����صĸ���
        isNeighbor = zeros(nSuperpixel, nSuperpixel); % ������Ǹ������ص��Ƿ����ڣ�1Ϊ���ڣ�0Ϊ������
        superpixelCell = cell(1,nSuperpixel); % superpixelCell.index:�������ڸ����������superpixelCell:neighbors:���������
        for i = 1:nSuperpixel
            index = find(beforeMergedSuperpixel==i);
            superpixelCell{i}.index = index;
            neighbors = findNeighbor(beforeMergedSuperpixel, index);
            superpixelCell{i}.neighbors = neighbors;
        end
     % �ҳ����ڵĳ�����  

        superpixelInfo = []; % col1:������1�ı�ǩ��col2:������2�ı�ǩ��col3:�����߽�ĳ��ȣ�col4:���ǵľ��룻codl5:�ϲ���cost
        for i = 1:nSuperpixel
            neighbors = superpixelCell{i}.neighbors; 
            neighbors = neighbors(:,2:end); % ��ǰ�����ص������
            for j = (i+1):nSuperpixel
                index2 = superpixelCell{j}.index; % ���������صĵ�
                intersection = intersect(neighbors, index2);
                sharedBoundary = length(intersection);
                if isempty(intersection) == 0  % �н����������ڵ�
                    isNeighbor(i,j) = 1;
                    tempSuperpixelInfo = zeros(1,5);
                    tempSuperpixelInfo(1,1:3) = [i j sharedBoundary];
                    % �����������صľ���
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
                    % ����ϲ���cost
                    s1 = length(index1);
                    s2 = length(index2);
                    cost = (s1*s2)*distance/(s1+s2);
                    tempSuperpixelInfo(1,5) = cost;
                    superpixelInfo = [superpixelInfo; tempSuperpixelInfo];
                end
            end
        end
        % ���ϲ�cost��С����������ϲ�
        [minCost, index3] = min(superpixelInfo(:,5));
        label1 = superpixelInfo(index3,1);
        label2 = superpixelInfo(index3,2);
        beforeMergedSuperpixel(beforeMergedSuperpixel==label2) = label1;
    end
    mergedSuperpixel = beforeMergedSuperpixel;
    

    
    
    
end
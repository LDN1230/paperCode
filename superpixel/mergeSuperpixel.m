function mergedSuperpixel = mergeSuperpixel(originalSuperpixel,feature)
% �������ں�
% originalSuperpixel: ԭ���ĳ�����map
% feature: �����ؿ��ں���Ҫ������������G-static����
    beforeMergedSuperpixel = originalSuperpixel;
    gamma = 0.3; 
    nMerge = 200;
    idd = [];
    for n = 1:nMerge
        fprintf('�������ں���:%d/%d......', n, nMerge);
        t1 = cputime;
        if n == 1 % ��һ�κϲ�������������ص�����������
            % �жϺϲ�ǰ������map���������ص��ڳ�����
            nSuperpixel = length(unique(beforeMergedSuperpixel(:))); % �����صĸ���
            isNeighbor = zeros(nSuperpixel, nSuperpixel); % ������Ǹ������ص��Ƿ����ڣ�1Ϊ���ڣ�0Ϊ������
            superpixelCell = cell(1,nSuperpixel); % superpixelCell.index:�������ڸ����������superpixelCell:neighbors:���������
            superpixel_label = [1:nSuperpixel];
            for i = 1:length(superpixel_label)
                ii = superpixel_label(i);
                index = find(beforeMergedSuperpixel==ii);
                superpixelCell{ii}.index = index;
                neighbors = findNeighbor(beforeMergedSuperpixel, index);
                superpixelCell{ii}.neighbors = neighbors;
            end
       

         % �ҳ����ڵĳ�����  
            superpixelInfo = []; % col1:������1�ı�ǩ��col2:������2�ı�ǩ��col3:����߽�ĳ��ȣ�col4:���ǵľ��룻codl5:�ϲ���cost
            for i = 1:length(superpixel_label)
                 ii = superpixel_label(i);
                neighbors = superpixelCell{ii}.neighbors; 
                neighbors = neighbors(:,2:end); % ��ǰ�����ص������
                neighbors = neighbors(:);
                neighbors(neighbors==0) = []; %����0���ĵ����
                for j = (i+1):length(superpixel_label)
                    jj = superpixel_label(j);
                    index2 = superpixelCell{jj}.index; % ���������صĵ�
                    intersection = intersect(neighbors, index2);
                    sharedBoundary = length(intersection);
                    if isempty(intersection) == 0  % �н����������ڵ�
                        isNeighbor(i,j) = 1;
                        tempSuperpixelInfo = zeros(1,5);
                        tempSuperpixelInfo(1,1:3) = [ii jj sharedBoundary];
                        % �����������صľ���
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
                        % ����ϲ���cost
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
                               find(superpixelInfo(:,1)==label2); find(superpixelInfo(:,2)==label2) ]; % �ҳ�superpixelInfo������һ�κϲ��ĳ����ص���
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
                neighbors = neighbors(:,2:end); % ��ǰ�����ص������
                neighbors = neighbors(:);
                neighbors(neighbors==0) = []; %����0���ĵ����
                intersection = intersect(neighbors, index5);
                sharedBoundary = length(intersection);
                if isempty(intersection) == 0  % �н����������ڵ�
                       isNeighbor(label3,label4) = 1;
                       tempSuperpixelInfo = zeros(1,5);
                       if label3 < label4
                            tempSuperpixelInfo(1,1:3) = [label3 label4 sharedBoundary];
                       else
                           tempSuperpixelInfo(1,1:3) = [label4 label3 sharedBoundary];
                       end
                end
                 % �����������صľ���
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
                 % ����ϲ���cost
                 s1 = length(index5);
                 s2 = length(index6);
                 cost = (s1*s2)*distance/(s1+s2);
                 tempSuperpixelInfo(1,5) = cost;
                 superpixelInfo(index4(gg),:)  =  tempSuperpixelInfo;
             end
        end
        
        
        % ���ϲ�cost��С����������ϲ�
        [minCost, index3] = min(superpixelInfo(:,5));
        label1 = superpixelInfo(index3,1);
        label2 = superpixelInfo(index3,2);
        superpixelInfo(index3,:) = []; % ����ϲ�����
        beforeMergedSuperpixel(beforeMergedSuperpixel==label2) = label1; % �滻���ϲ������صı�ǩ
        id = find(superpixel_label==label2);
        if isempty(id)
            fprintf('error!!!!!!!!!!!!!!!!!!!!');
        end
        superpixel_label(id) = []; % ���ϲ��ĳ����ض�Ӧ�ı�ǩɾ��
        superpixelCell{label1}.index = [superpixelCell{label1}.index; superpixelCell{label2}.index]; % �ϲ��ĳ���������Ҳ�ϲ�
        neighbors = findNeighbor(beforeMergedSuperpixel, superpixelCell{label1}.index);
        superpixelCell{label1}.neighbors = neighbors ; % ���¼���ϲ������ص�����
        %superpixelInfo()
        t2 = cputime-t1;
        fprintf('��ʱ%.2fs\n', t2);
        
       

                
                
    end
    mergedSuperpixel = beforeMergedSuperpixel;
    

    
    
    
end
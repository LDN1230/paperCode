function [result] = myJSR(img, im_2d, train, testX, testY, testIndex, scale, K, superpixel_label, nList, LBP_feature, k)
% img: 高光谱图像的的3D数据
% im_2d: 高光谱图像的的2D数据
% train: 训练样本
% testX, testY, testIndex: 分别是测试样本的数据，标签，索引
% scale: 领域的范围
% K: 联合稀疏表示的稀疏水平
% superpixel_label: 超像素图像
% nList：4邻域点
    addpath('./classifier/JSR');
    [i_row, i_col, nb] = size(img);
    trainX = train.data;
    trainY = train.label;
    trainIndex = train.index;
    C = max(trainY(:));
    
    lbpOfClass = zeros(size(LBP_feature,1), C);
    for cc = 1: C
        ind = trainIndex(find(trainY==cc));
        averageLbp = sum(LBP_feature(:, ind), 2)/length(ind);
        lbpOfClass(:, cc) = averageLbp;
    end
    
    for i = 1:1:size(testIndex,1)
        if  mod(i, uint16(size(testX,2)/10)) == 0
            fprintf('******');
        end
       testIndex(i) = sub2ind([512 217], 220, 137);
        row = mod(testIndex(i),i_row);
        if row == 0
            row = i_row;
        end
        col = ceil(testIndex(i)/i_row);
        row_range = ceil(row-(scale-1)/2 : row+(scale-1)/2);
        row_range(row_range<=0)= 1;row_range(row_range>=i_row)= i_row;
        col_range = ceil(col-(scale-1)/2 : col+(scale-1)/2);
        col_range(col_range<=0)= 1;col_range(col_range>=i_col)= i_col;
        rc = [];
        for r = 1: length(row_range)
            for c = 1: length(col_range)
                    rc = [rc; [row_range(r) col_range(c)]];
            end
        end
        index1 = sub2ind([i_row i_col], rc(:,1), rc(:,2));%测试样本的邻域点索引
        [tr, tc] = ind2sub([i_row i_col], testIndex(i));
        index2 = find(superpixel_label==superpixel_label(tr, tc));%测试样本的同超像素点索引
        index3 = intersect(index1, index2);
        index6 = setdiff(index1, index3);
        nn = length(index1)-length(index3);
        index4 = setdiff(index2, index1);
        index5 = [];
        if length(index4)>0 || length(index6)>0
            index7 = [index4; index6];
            X1 = LBP_feature(:, index7);
            xc = LBP_feature(:, testIndex(i));
            correlation = sqrt(sum((repmat(xc, [1 size(X1,2)])-X1).^2));
            [sortedCorr, sortedIndex] = sort(correlation, 'ascend');
            if k < length(index7)
                %Indian Pines:0.05 Salinas:1
%                 index5 = index7(sortedIndex(1: (nn+round(0.05*(length(index7)-nn)))));
                    index5 = index7(sortedIndex(1: k));
            else
                 index5 = index7;
            end
        end
        index8 = [index3; index5];

        X = im_2d(:, index8);
        S = SOMP(trainX,X,K);
        
%         X1 = im_2d(:, index1);
%         S1 = SOMP(trainX,X1,K);
        
        for j = 1:1:C
            temp = find(trainY == j);
            D_c = trainX(:,temp);
            S_c = S(temp,:);
            re_temp = X - D_c*S_c; 
            residual(i,j) = norm(re_temp,'fro');    
            
%             S_c1 = S1(temp,:);
%             re_temp1 = X1 - D_c*S_c1;         
%             residual1(i,j) = norm(re_temp1,'fro');    
            

        end  
    end
    fprintf('\n');
    residual_1 = residual./repmat(sqrt(sum(residual.*residual)),[size(residual,1) 1]);
    residual_1 = residual_1';
    
%     residual_11 = residual1./repmat(sqrt(sum(residual1.*residual1)),[size(residual1,1) 1]);
%     residual_11 = residual_11';
    

    
    for i = 1:length(testY)
        result(i) = find(residual_1(:, i) == min(residual_1(:, i)), 1);
%         result1(i) = find(residual_11(:, i) == min(residual_11(:, i)));

    end
    
    
    
end
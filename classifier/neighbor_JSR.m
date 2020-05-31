function [result, residual_1] = neighbor_JSR(img, train, testX, testY, testIndex, scale, K)
    addpath('./classifier/JSR');
    [i_row, i_col, nb] = size(img);
    trainX = train.data;
    trainY = train.label;
    trainIndex = train.index;
    C = max(trainY(:));
    for i = 1:1:size(testX,2)
        if  mod(i, uint16(size(testX,2)/10)) == 0
            fprintf('******');
        end
        row = mod(testIndex(i),i_row);
        if row == 0
            row = i_row;
        end
        col = ceil(testIndex(i)/i_row);
        row_range = ceil(row-(scale-1)/2 : row+(scale-1)/2);
        row_range(row_range<=0)= 1;row_range(row_range>=i_row)= i_row;
        col_range = ceil(col-(scale-1)/2 : col+(scale-1)/2);
        col_range(col_range<=0)= 1;col_range(col_range>=i_col)= i_col;
        temp = img(row_range,col_range,:);
        X = ToVector(temp)';       
        X = X./repmat(sqrt(sum(X.*X)),[size(X,1) 1]);
        S = SOMP(trainX,X,K);
        for j = 1:1:C
            temp = find(trainY == j);
            D_c = trainX(:,temp);
            S_c = S(temp,:);
            re_temp = X - D_c*S_c;%计算邻域的JSR稀疏系数         
            residual(i,j) = norm(re_temp,'fro');       
        end  
    end
    fprintf('\n');
    residual_1 = residual./repmat(sqrt(sum(residual.*residual)),[size(residual,1) 1]);
    residual_1 = residual_1';
    for i = 1:length(testY)
        result(i) = find(residual_1(:, i) == min(residual_1(:, i)));
    end
    
end
function [OA,AA,kappa,CA] = superpixel_JSR(img, train, test, superpixel_label,index_map, K)
    addpath('./classifier/JSR');
    trainX = train.data;
    trainY = train.label;
    trainIndex = train.index;
    testX = test.data;
    testY = test.label;
    testIndex = test.index;
    C = max(trainY(:));
    for i = 1:1:size(testX,2)
        index = [trainIndex; testIndex];
        y_index = testIndex(i);
        y_jonit_index = index_map((superpixel_label==superpixel_label(y_index)));
%         [~,tt_index_location,~] =...
%             intersect(index , y_jonit_index);
        X = img( :,y_jonit_index);
        S = SOMP(trainX,X,K);

        for j = 1:1:C
            temp = find(trainY == j);
            D_c = trainX(:,temp);
            S_c = S(temp,:);
            re_temp = X - D_c*S_c;%计算邻域的JSR稀疏系数         
            residual(i,j) = norm(re_temp,'fro');       
        end  
    end
    residual_1 = residual./repmat(sqrt(sum(residual.*residual)),[size(residual,1) 1]);
    residual_1 = residual_1';
    for i = 1:length(testY)
        result(i) = find((residual_1(:, i) == min(residual_1(:, i))),1);
    end
    [OA,AA,kappa,CA] = confusion(testY, result);
end
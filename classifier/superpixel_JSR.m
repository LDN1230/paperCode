function [result, residual_1]  = superpixel_JSR(img, trainX, trainY, trainIndex, testX, testY, testIndex, superpixel_label, K, nList)
    addpath('./classifier/JSR');
%     trainX = train.data;
%     trainY = train.label;
%     trainIndex = train.index;
%     testX = test.data;
%     testY = test.label;
%     testIndex = test.index;
   

    cells = cell(1, size(testX,2));
    C = max(trainY(:));
    flag = zeros(1, size(testX,2));
    residual = zeros(size(testX,2), C);
    for i = 1:1:size(testX,2)
        if  mod(i, uint16(size(testX,2)/10)) == 0
            fprintf('******');
       end
%         fprintf('superpixel_JSR: %d/%d\n', i, size(testX,2));
        if flag(1,i) == 0
            y_index = testIndex(i);
%             if isempty(cells{i})
%                 y_joint_index  = pixelInSameSuperpixel([y_index], superpixel_label, y_index,  nList);
%                 for w = 1: length(y_joint_index)
%                         d = find(testIndex==y_joint_index(w));
%                         if ~isempty(d)
%                             cells{d}.YY = y_joint_index;
%                         end
%                 end
%             else
%                 y_joint_index = cells{i}.YY;
%             end
            y_joint_index = findPixelInSameSuperpixel(superpixel_label, y_index, nList);
%             y_joint_index  = pixelInSameSuperpixel([y_index], superpixel_label, y_index,  nList);
            X = img( :,y_joint_index);
            S = SOMP(trainX,X,K);

            for j = 1:1:C
                temp = find(trainY == j);
                D_c = trainX(:,temp);
                S_c = S(temp,:);
                re_temp = X - D_c*S_c;%计算邻域的JSR稀疏系数         
                residual(i,j) = norm(re_temp,'fro');       
            end  
            flag(1,i) = 1; 
            re = residual(i,:);
            for w = 2: length(y_joint_index)
                    d = find(testIndex==y_joint_index(w));
                    flag(1,d) = 1;
                    
                    residual(d,:) = residual(d,:) + re;
            end
        end
    end
    fprintf('\n');
    residual_1 = residual./repmat(sqrt(sum(residual.*residual)),[size(residual,1) 1]);
    residual_1 = residual_1';
    for i = 1:length(testY)
        result(i) = find((residual_1(:, i) == min(residual_1(:, i))),1);
    end
%     [OA,AA,kappa,CA] = confusion(testY, result);
end
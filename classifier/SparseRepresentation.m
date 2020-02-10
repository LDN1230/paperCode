function [classification_result] = SparseRepresentation(train, test, nClass, K)
    addpath('./common');
    addpath('./classifier/JSR');
    trainX = train.data;
    trainY = train.label;
    testX = test.data;
    testY = test.label;
    % ��SOMP����ϡ���ʾϵ���Ͳв�
    for i = 1:1:size(testX,2)
        tic;
        S = SOMP(trainX, testX(:,i), K);
        
        for j = 1:1:nClass
            temp = find(trainY == j);
            D_c = trainX(:,temp);
            S_c = S(temp,:);
            re_temp = testX(:,i) - D_c*S_c;
            residual(i,j) = norm(re_temp,'fro');
        end
        toc;
    end
    residual_1 = residual./repmat(sqrt(sum(residual.*residual)),[size(residual,1) 1]);
    residual_1 = residual_1';
    % �в���С����ΪԤ��ķ�����
    for i = 1:length(testY)
        result(i) = find(residual_1(:, i) == min(residual_1(:, i)), 1);
    end

    [OA,AA,kappa,CA] = confusion(testY, result);
    
    classification_result.OA = OA;
    classification_result.AA = AA;
    classification_result.kappa = kappa;
    classification_result.CA = CA;
end
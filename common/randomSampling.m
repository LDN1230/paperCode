function [train, test, nClass] = randomSampling(data, labels, method, n)
    nClass = max(labels(:));
    trainIndex = [];%ѵ��������������
    testIndex = [];%���Լ�����������
    nEveryClass =zeros(1, nClass);%�������������
    nEveryClassTrain =zeros(1, nClass);%���������ѵ������
    nEveryClassTest =zeros(1, nClass);%������������Ը���
    labelTrain = [];
    labelTest = [];
    
    if strcmp(method, 'byPercent')%���ٷֱ������ȡ
        proportion = n;%ѵ����ռ�ı���
        for i = 1:nClass
            index = [];
            index = find(labels == i);%��ÿ������������
            nEveryClass(i) = length(index);
            num = round(length(index) * proportion);
            nEveryClassTrain(i) = num;
            labelTrain = [labelTrain; ones(num,1)*i];
            index1 = randperm(length(index));
            trainIndex = [trainIndex; index(index1(1:num))]; 
            testIndex = [testIndex; index(index1((num+1):end))];
            nEveryClassTest(i) = nEveryClass(i)-nEveryClassTrain(i);
            labelTest = [labelTest; ones(nEveryClassTest(i),1)*i];
        end
    end
    
    if strcmp(method, 'byNumber')%���̶���Ŀ�����ȡ
        for i = 1:nClass
            index = [];
            index = find(labels == i);%��ÿ������������
            nEveryClass(i) = length(index);
            index1 = randperm(length(index));
            trainIndex = [trainIndex; index(index1(1:n))];
            labelTrain = [labelTrain; ones(n,1)*i];
            testIndex = [testIndex; index(index1((n+1):end))];
            nEveryClassTrain(i) = n;
            nEveryClassTest(i) = nEveryClass(i)-nEveryClassTrain(i);
            labelTest = [labelTest; ones(nEveryClassTest(i),1)*i];
        end
    end
   
    train.index = trainIndex;
    train.nEveryClass = nEveryClassTrain;
    train.label = labelTrain;
    train.data = data(:, trainIndex);
    test.index = testIndex;
    test.nEveryClass = nEveryClassTest;
    test.label = labelTest;
    test.data = data(:, testIndex);
end
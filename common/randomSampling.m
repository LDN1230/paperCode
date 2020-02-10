function [train, test, nClass] = randomSampling(data, labels, method, n)
    nClass = max(labels(:));
    trainIndex = [];%训练集样本的索引
    testIndex = [];%测试集样本的索引
    nEveryClass =zeros(1, nClass);%各类的样本个数
    nEveryClassTrain =zeros(1, nClass);%各类的样本训练个数
    nEveryClassTest =zeros(1, nClass);%各类的样本测试个数
    labelTrain = [];
    labelTest = [];
    
    if strcmp(method, 'byPercent')%按百分比随机抽取
        proportion = n;%训练集占的比例
        for i = 1:nClass
            index = [];
            index = find(labels == i);%找每类样本的索引
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
    
    if strcmp(method, 'byNumber')%按固定数目随机抽取
        for i = 1:nClass
            index = [];
            index = find(labels == i);%找每类样本的索引
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
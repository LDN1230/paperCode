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
            num = ceil(length(index) * proportion);
            nEveryClassTrain(i) = num;
            temp_label = double(i)*ones(num,1);
            labelTrain = [labelTrain; temp_label];
            index1 = randperm(length(index));
            trainIndex = [trainIndex; index(index1(1:num))]; 
            testIndex = [testIndex; index(index1((num+1):end))];
            nEveryClassTest(i) = nEveryClass(i)-nEveryClassTrain(i);
            labelTest = [labelTest; ones(nEveryClassTest(i),1)*double(i)];
        end
    end
    
    if strcmp(method, 'byNumber')%按固定数目随机抽取
        
         if length(n) ~= 1
                    for i = 1:nClass
                       
                        index = [];
                        index = find(labels == i);%找每类样本的索引
                        nEveryClass(i) = length(index);
                        index1 = randperm(length(index));
                        trainIndex = [trainIndex; index(index1(1:n(i)))]; 
                        nEveryClassTrain(i) = n(i);
                        temp_label = double(i)*ones(n(i),1);
                        labelTrain = [labelTrain; temp_label];
                        testIndex = [testIndex; index(index1((n(i)+1):end))];
                        nEveryClassTest(i) = nEveryClass(i)-nEveryClassTrain(i);
                        labelTest = [labelTest; ones(nEveryClassTest(i),1)*double(i)];
                    end
         else
                    for i = 1:nClass
                        index = [];
                        index = find(labels == i);%找每类样本的索引
                        nEveryClass(i) = length(index);
                        index1 = randperm(length(index));
                        if length(index) < n
                            n = uint32(length(index)/10)*8;
                        end
                        trainIndex = [trainIndex; index(index1(1:n))]; 
                        nEveryClassTrain(i) = n;
                        temp_label = double(i)*ones(n,1);
                        labelTrain = [labelTrain; temp_label];
                        testIndex = [testIndex; index(index1((n+1):end))];
                        nEveryClassTest(i) = nEveryClass(i)-nEveryClassTrain(i);
                        labelTest = [labelTest; ones(nEveryClassTest(i),1)*double(i)];
                    end
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
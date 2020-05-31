function [trainIndex, testIndex] = randomSampling(labels, nClass, method, n)
    trainIndex = [];%训练集样本的索引
    testIndex = [];%测试集样本的索引
    labels_count =zeros(1, nClass);%各类的样本个数
    
    if strcmp(method, 'byPercent')%按百分比随机抽取
       proportion = n;%训练集占的比例
        for i = 1:nClass
            index = [];
            index = find(labels == i);%找每类样本的索引
            labels_count(i) = length(index);
            num = ceil(length(index) * proportion);
            index1 = randperm(length(index));
            trainIndex = [trainIndex; index(index1(1:num))]; 
            testIndex = [testIndex; index(index1((num+1):end))];
        end
    end
    
    if strcmp(method, 'byNumber')%按固定数目随机抽取
         if length(n) ~= 1
                    for i = 1:nClass
                        index = [];
                        index = find(labels == i);%找每类样本的索引
                        labels_count(i) = length(index);
                        index1 = randperm(length(index));
                        trainIndex = [trainIndex; index(index1(1:n(i)))]; 
                        testIndex = [testIndex; index(index1((n(i)+1):end))];
                    end
        
         else
                    for i = 1:nClass
                        index = [];
                        index = find(labels == i);%找每类样本的索引
                        labels_count(i) = length(index);
                        index1 = randperm(length(index));
                        if length(index) < n
                            n = uint32(length(index)/10)*8;
                        end
                        trainIndex = [trainIndex; index(index1(1:n))]; 
                        testIndex = [testIndex; index(index1((n+1):end))];
                    end
        end
    end
end
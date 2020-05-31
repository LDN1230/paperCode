function [trainIndex, testIndex] = randomSampling(labels, nClass, method, n)
    trainIndex = [];%ѵ��������������
    testIndex = [];%���Լ�����������
    labels_count =zeros(1, nClass);%�������������
    
    if strcmp(method, 'byPercent')%���ٷֱ������ȡ
       proportion = n;%ѵ����ռ�ı���
        for i = 1:nClass
            index = [];
            index = find(labels == i);%��ÿ������������
            labels_count(i) = length(index);
            num = ceil(length(index) * proportion);
            index1 = randperm(length(index));
            trainIndex = [trainIndex; index(index1(1:num))]; 
            testIndex = [testIndex; index(index1((num+1):end))];
        end
    end
    
    if strcmp(method, 'byNumber')%���̶���Ŀ�����ȡ
         if length(n) ~= 1
                    for i = 1:nClass
                        index = [];
                        index = find(labels == i);%��ÿ������������
                        labels_count(i) = length(index);
                        index1 = randperm(length(index));
                        trainIndex = [trainIndex; index(index1(1:n(i)))]; 
                        testIndex = [testIndex; index(index1((n(i)+1):end))];
                    end
        
         else
                    for i = 1:nClass
                        index = [];
                        index = find(labels == i);%��ÿ������������
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
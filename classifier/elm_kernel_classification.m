function [TTrain,TTest,TrainAC,accur_ELM,TY,label] = elm_kernel_classification(train,test,n,c,kernel_type, kerneloption)
    addpath('./classifier/ELM');
    
    trainX = train.data;
    trainX = trainX';
    testX = test.data;
    testX = testX';
    xapp = [train.label trainX];
    xtest = [test.label testX];
    
    [TTrain,TTest,TrainAC,accur_ELM,TY,label] = elm_kernel(xapp,xtest,n ,c,kernel_type, kerneloption);
end
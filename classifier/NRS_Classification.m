function [OA,AA,kappa,CA] = NRS_Classification(train, test, lambda)
%
% Using MH weights to produce class label
%

addpath('./classifier/NRS');
    CTrain = train.nEveryClass;
    DataTest = test.data;
    DataTest = DataTest';
    DataTrain = train.data;
    DataTrain = DataTrain';

    numClass = length(CTrain);
    [m Nt]= size(DataTest);
    for j = 1: m
        Y = DataTest(j, :);
        a = 0;
        for i = 1: numClass 
            % Obtain Multihypothesis from training data
            HX = DataTrain((a+1): (CTrain(i)+a), :);
            a = CTrain(i) + a;

            % Multihypothesis to produce prediction Y
            Y_hat = NRS_tik(Y, HX, lambda);

    %         % sparse coding: solve a reduced linear system
    %         Yrecover = sparse_coding_methods(Yinit, D{i}', Yspar, sc_algo);
            Y_snr(i) = SNR(Y, Y_hat);
        end
       [value class(j)] = max(Y_snr);
    end
    [OA,AA,kappa,CA] = confusion(test.label, class);
end

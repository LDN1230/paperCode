function class = SRC_Classification_post(DataTrain, CTrain, DataTest, lambda)
%
% Using MH weights to produce class label
%
addpath('./classifier/JSR');
addpath('./common');
for ii = 1: size(DataTest, 1)
   DataTest(ii, :) = DataTest(ii, :)./norm(DataTest(ii, :));
end
for ii = 1: size(DataTrain, 1)
   DataTrain(ii, :) = DataTrain(ii, :)./norm(DataTrain(ii, :));
end

numClass = length(CTrain);
[m Nt]= size(DataTest);
for j = 1: m
    Y = DataTest(j, :); % 1 x dim
    % weighted vector
    % W = ones(size(DataTrain, 1), 1);
    W = (sum((DataTrain' - repmat(Y', [1 size(DataTrain,1)])).^2))'; % num x 1
    % SPAMS Prediction
    % parameter of the optimization procedure are chosen
    param.L=min(size(DataTrain,2), size(DataTrain,1)); % not more than 20 non-zeros coefficients (default: min(size(D,1),size(D,2)))
    param.lambda=lambda; % not more than 20 non-zeros coefficients
    param.numThreads=8; % number of processors/cores to use; the default choice is -1
                        % and uses all the cores of the machine
    param.mode=2;       % penalized formulation
    A = mexLassoWeighted(Y', DataTrain', W, param) ;
    weights = full(A);
    a = 0;
    for i = 1: numClass 
        % Obtain Multihypothesis from training data
        HX = DataTrain((a+1): (CTrain(i)+a), :); % sam x dim
        HW = weights((a+1): (CTrain(i)+a));
        a = CTrain(i) + a;
        Y_hat = HW'*HX;
        % Y_hat = abs(Y_hat)./norm(Y_hat);
        Dist_Y(i) = norm(Y - Y_hat); 
    end
   [value class(j)] = min(Dist_Y);
    
end

function class = CRT_Classification_post(DataTrain, CTrain, DataTest, lambda)
%
% Using MH weights to produce class label
%

numClass = length(CTrain);
[m Nt]= size(DataTest);
for j = 1: m
    Y = DataTest(j, :); % 1 x dim
    norms = sum((DataTrain' - repmat(Y', [1 size(DataTrain,1)])).^2);
    % norms = ones(size(DataTrain,1), 1);
    G = diag(lambda.*norms);
    weights = (DataTrain*DataTrain' + G)\(DataTrain*Y');
    a = 0;
    for i = 1: numClass 
        % Obtain Multihypothesis from training data
        HX = DataTrain((a+1): (CTrain(i)+a), :); % sam x dim
        HW = weights((a+1): (CTrain(i)+a));
        a = CTrain(i) + a;
        Y_hat = HW'*HX;
        
        Dist_Y(i) = norm(Y - Y_hat); 
    end
   [value class(j)] = min(Dist_Y);
end

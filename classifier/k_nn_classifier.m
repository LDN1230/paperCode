function [OA,AA,kappa,CA] = k_nn_classifier(Z,v,k,X, testY)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
%   [z]=k_nn_classifier(Z,v,k,X)
% k-nearest neighbor classifier for c classes (also used in Chapter 4). The
% classification is based on a reference data set, Z, for which the class
% labels of its vectors are known.
%
% INPUT ARGUMENTS:
%   Z:  lxN1  matrix, whose i-th column corresponds to the
%       i-th reference vector.
%   v:  N1-dimensional vector whose i-th  component contains the
%       label of the class where the i-th reference vector belongs.
%   k:  the number of nearest neighbors of the reference set that are
%       taken into account for the classification of a given vector.
%   X:  lxN matrix whose columns are the data vectors to be classfied.
%
% OUTPUT ARGUMENTS:
%   z:  N-dimensional vector whose i-th component contains the label
%       of the class where the i-th vector of X is assigned.
%
% (c) 2010 S. Theodoridis, A. Pikrakis, K. Koutroumbas, D. Cavouras
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[l,N1]=size(Z);
[l,N]=size(X);
c=max(v);  %Number of classes
% Computation of the (squared) Euclidean distance of a point from each
% reference vector
for i=1:N
    Y = repmat(X(:,i),1, size(Z,2));
    dist=sqrt(sum((Y-Z).^2));
    %Sorting the above distances in ascending order
    [sorted,nearest]=sort(dist, 2, 'ascend');
    %Counting the class occurences among the k closest reference vectors
    % Z(:,i)
    refe=zeros(1,c); %Counting the reference vectors per class
    for q=1:k
        class=v(nearest(q));
        refe(class)=refe(class)+1;
    end
    [val,z(i)]=max(refe);
end
[OA,AA,kappa,CA] = confusion(testY, z);
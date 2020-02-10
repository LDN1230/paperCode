function [z]=Knn(k,train_array,train_labels,array)

train_array = train_array';
array = array';
train_labels = train_labels';
[l,N1]=size(train_array);
[l,N]=size(array);
num_of_classes=max(train_labels);
% Calculate the squared eucleidian distance of a point from each reference vector
for i=1:N
    Y = repmat(array(:,i),1, size(train_array,2));
    distance=sqrt(sum((Y-train_array).^2));
    
    [sorted,nearest]=sort(distance, 2, 'ascend');
    % Count occurence of each class for the top k reference vectors
    ref_vector=zeros(1,num_of_classes);
    for j=1:k
        class=train_labels(nearest(j));
        ref_vector(class)=ref_vector(class)+1;
    end
    [val,z(i)]=max(ref_vector);
end
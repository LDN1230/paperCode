function [labels_t, numlabels, seedx, seedy] = SuperpixelSegmentation(data,numSuperpixels)

[nl, ns, nb] = size(data);
x = data;
x = reshape(x, nl*ns, nb);
x = x';

input_img = zeros(1, nl * ns * nb);
startpos = 1;
for i = 1 : nl
    for j = 1 : ns
        input_img(startpos : startpos + nb - 1) = data(i, j, :);
        startpos = startpos + nb;
    end
end


%% perform Regional Clustering

%numSuperpixels = 200;  % number of segments
compactness = 0.1; % compactness2 = 1-compactness, compactness*dxy+compactness2*dspectral
dist_type = 2; % 1:ED£»2£ºSAD; 3:SID; 4:SAD-SID
seg_all = 1; % 1: All pixels are clustered£¬ 2£ºexist un-clustered pixels
% labels:segment no of each pixel
% numlabels: actual number of segments
[labels, numlabels, seedx, seedy] = RCSPP(input_img, nl, ns, nb, numSuperpixels, compactness, dist_type, seg_all);
clear input_img;

labels_t = zeros(nl, ns, 'int32');
for i=1:nl
    for j=1:ns
        labels_t(i,j) = labels((i-1)*ns+j);
    end
end
end
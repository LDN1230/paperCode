clc;
clear all;

addpath('./dataset');
load Salinas_corrected.mat;
img = salinas_corrected;






%SLIC
numSuperpixels = 100;  % the desired number of superpixels
compactness = 0.1; % compactness2 = 1-compactness, the clustering is according to: compactness*dxy+compactness2*dspectral
dist_type = 2; % distance type - 1:Euclidean£»2£ºSAD; 3:SID; 4:SAD-SID
seg_all = 1; % 1: all the pixles participate in the clustering£¬ 2: some pixels would not
[superpixel_label, numlabels, seedx, seedy] = SLIC( img, numSuperpixels, compactness, dist_type, seg_all);
superpixel_label = double(reshape(superpixel_label, nRow, nCol));

figure;
subplot(121); imshow(img1);


clc;
clear all;

%% 包含文件目录
addpath('./dataset');
addpath('./common');
addpath('./feature/LBP');
addpath('./feature/RF');
addpath('./feature/Gabor');
addpath('./classifier');
addpath('./superpixel');
addpath('./classifier/KNN');
addpath '.\classifier\libsvm-3.17';
addpath '.\classifier\libsvm-3.17\matlab';


%% 导入数据集
load IndiaP.mat; 

load Indian_pines_gt.mat

[nRow, nCol, nBand] = size(img);
gt = indian_pines_gt;
% figure('Name', 'indian_pines的label');imshow(label2rgb(gt));

%% 高光谱图像-》伪彩色图像
RgbImg = hyperspectral2rgb(img, [3 30 87]);

%% 向量化、归一化
im_2d = ToVector(img)';
im_2d = im_2d./repmat(sqrt(sum(im_2d.*im_2d)),[size(im_2d,1) 1]);


%% PCA 降维
[coeff score latent] = pca(im_2d');
pca_result = score(:,1:3);
superpixel_input =reshape(pca_result, nRow, nCol, 3);  %超像素分割的输入


%% 产生超像素
% SLIC
numSuperpixels = 200;  % the desired number of superpixels  indian_pines:1500
compactness = 0.1; % compactness2 = 1-compactness, the clustering is according to: compactness*dxy+compactness2*dspectral
dist_type = 2; % distance type - 1:Euclidean；2：SAD; 3:SID; 4:SAD-SID
seg_all = 1; % 1: all the pixles participate in the clustering， 2: some pixels would not
[superpixel_label, numlabels, seedx, seedy] = SLIC( img, numSuperpixels, compactness, dist_type, seg_all);
superpixel_label = double(reshape(superpixel_label, nRow, nCol));

% Entropy rate
% number_superpixels = 326;lambda_prime = 0.8;sigma = 10; conn8 = 1;
% superpixel_label = EntropyRate(superpixel_input, number_superpixels, lambda_prime, sigma,conn8);




%% 特征提取

% LBP特征
lbp_input = reshape(pca_result, nRow, nCol, 3);
r = 2;
nr = 8;
mapping = getmapping(nr,'u2');
[LBP_feature, lbp_img] = LBP_feature_global(lbp_input, 2, nr, mapping, 11, indian_pines_gt);
d = size(LBP_feature, 3);
LBP_feature = reshape(LBP_feature, nRow*nCol, d);
LBP_feature = LBP_feature';
maxV = max(LBP_feature(:));
LBP_feature = LBP_feature./maxV;

% 利用lbp特征，超像素融合
mergedSuperpixel = mergeSuperpixel(superpixel_label,lbp_img);

% 画超像素图像
paintSuperpixelAdge(RgbImg, superpixel_label);
paintSuperpixelAdge(RgbImg, mergedSuperpixel);

% Recursive filter特征
% RF_feature = spatial_feature(img, 204, 0.5);
% RF_feature = ToVector(RF_feature)';
% RF_feature = RF_feature./repmat(sqrt(sum(RF_feature.*RF_feature)),[size(RF_feature,1) 1]);


% Gabor 特征
% 第一种
% lambda=[0,1/2,1/3];
% theta=[0,45,90,135]/180*pi;
% var_xy=[3,3;7,7;11,11;15,15];
% Gabor_feature = GetGaborFeat(pca_result,lambda,theta,var_xy);
% 第二种
% pca_result1 = score(:,1:10);
% gabor_input =reshape(pca_result1, nRow, nCol, 10); 
% BW = 5;
% DataGabor = Gabor_feature_extraction_PC(gabor_input, BW);
% d = size(DataGabor, 3);
% Data_tmp = reshape(DataGabor, nRow*nCol, d);
% Data_tmp = Data_tmp';
%% 训练集、测试集的划分
[train, test, nClass] = randomSampling(im_2d, indian_pines_gt, 'byPercent', 0.1);

%% 分类

trainX = train.data;
trainY = train.label;
trainIndex = train.index;
testX = test.data;
testY = test.label;
testIndex = test.index;

% 稀疏表示
% K = 1;
% [classification_result] = SparseRepresentation(train, test, nClass, K);


% ELM-kernel
% kerneloption = [5];
% c = 1024;
% [TTrain,TTest,TrainAC,accur_ELM,TY,label] = elm_kernel_classification(train,test,1,c,'RBF_kernel',kerneloption);


% 基于邻域的JSR
% scale = 3;
% K = 1;
% [OA,AA,kappa,CA] = neighbor_JSR(img, train, test, scale, K);

% 基于超像素的JSR  
K = 1;
index_map = reshape(1:size(im_2d,2),[nRow,nCol]);
[OA,AA,kappa,CA] = superpixel_JSR(im_2d, train, test, superpixel_label,index_map, K);
[OA1,AA1,kappa1,CA1] = superpixel_JSR(im_2d, train, test, mergedSuperpixel,index_map, K);
% NRS
% lambda = 0.6;
% [OA,AA,kappa,CA] = NRS_Classification(train, test, lambda);

% KNN
% [best_k, knn_test_response]=calculate_best_k(trainY, trainX);
% [OA,AA,kappa,CA] = k_nn_classifier(trainX, trainY, best_k, testX, testY);

% SVM
% [C, P] = SVM_MG(testX', trainX', trainY, testY, 128, 0.0156);
% [OA,AA,kappa,CA] = confusion(testY, C);
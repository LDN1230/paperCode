clc;
clear all;

%% �����ļ�Ŀ¼
addpath('./dataset');
addpath('./common');
addpath('./feature/LBP');
addpath('./feature/RF');
addpath('./feature/Gabor');
addpath('./classifier');
addpath('./superpixel');
addpath('./GCmex');
addpath('./classifier/KNN');
addpath '.\classifier\libsvm-3.17';
addpath '.\classifier\libsvm-3.17\matlab';
addpath('./fast FCM');

%% �������ݼ�
nDataset =3 ;
if nDataset == 1
    load IndiaP.mat; 
    load Indian_pines_gt.mat;
    gt = indian_pines_gt;
    gtImg = label2color(gt, 'india');
    cluster = 16;
    numSuperpixels = 4500;
    num =  [ 1  15  9  3  5  8  1  5  1  10  25  6  3  13  4  1];
    fprintf('Indian Pines\n');
end
if nDataset == 2
    load Salinas_corrected.mat;
    load Salinas_gt.mat;
    img = salinas_corrected;
    gt = salinas_gt;
    cluster = 16;
    numSuperpixels = 3000;
    num = [10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 ];
    fprintf('Salinas\n');
end
if nDataset == 3
    load PaviaU.mat;
    load PaviaU_gt.mat;
    img = paviaU;
    gt = paviaU_gt;
    cluster = 9;
     numSuperpixels = 10000;
     fprintf('PaviaU\n');
end
if nDataset == 4
    load PaviaC.mat;
    load PaviaC_gt.mat;
    cluster = 9;
     numSuperpixels = 13000;
     fprintf('PaviaC\n');
end
[nRow, nCol, nBand] = size(img);
% ��������
[numN, nList] = getNeighFromGrid(nRow, nCol);
% figure('Name', 'indian_pines��label');imshow(label2rgb(gt));

%% �߹���ͼ��-��α��ɫͼ��
RgbImg = hyperspectral2rgb(img, [3 30 87]);
% figure('Name', 'α��ɫͼ��'); imshow(RgbImg);
%% ����������һ��
im_2d = ToVector(img)';
im_2d = im_2d./repmat(sqrt(sum(im_2d.*im_2d)),[size(im_2d,1) 1]);


%% PCA ��ά
[coeff score latent] = pca(im_2d');
pca_result = score(:,1:3);
superpixel_input =reshape(pca_result, nRow, nCol, 3);  %�����طָ������
% pca3_rgbImg = uint8(255*superpixel_input);
% figure('Name', 'PCAǰ�������ɷֵ�α��ɫͼ��'); imshow(pca3_rgbImg);


%% ������ȡ
% LBP����
% t1 = cputime;
% fprintf('LBP������ȡ��......\n');
% lbp_input = reshape(pca_result(:,1), nRow, nCol, 1);
% r = 2;
% nr = 8;
% mapping = getmapping(nr,'u2');
% [LBP_feature, lbp_img] = LBP_feature_global(lbp_input, 2, nr, mapping, 11, gt);
% d = size(LBP_feature, 3);
% LBP_feature = reshape(LBP_feature, nRow*nCol, d);
% LBP_feature = LBP_feature';
% maxV = max(LBP_feature(:));
% LBP_feature = LBP_feature./maxV;
% t2 = cputime-t1;
% fprintf('LBP������ȡ����ʱ��:%.fs\n', t2);

% ����lbp�������������ں�
% t1 = cputime;
% mergedSuperpixel = mergeSuperpixel_test(superpixel_label,lbp_img);
% t2 = cputime-t1;
% fprintf('�������ں�����ʱ��:%.4fs\n', t2);


% ��������ͼ��
% paintSuperpixelAdge(RgbImg, segment_result);
% paintSuperpixelAdge(RgbImg, superpixel_label);

% Recursive filter����
% RF_feature = spatial_feature(img, 204, 0.5);
% RF_feature = ToVector(RF_feature)';
% RF_feature = RF_feature./repmat(sqrt(sum(RF_feature.*RF_feature)),[size(RF_feature,1) 1]);


% Gabor ����
% ��һ��
% lambda=[0,1/2,1/3];
% theta=[0,45,90,135]/180*pi;
% var_xy=[3,3;7,7;11,11;15,15];
% Gabor_feature = GetGaborFeat(pca_result,lambda,theta,var_xy);
% �ڶ���
% BW = 1;
% DataGabor = Gabor_feature_extraction_PC(img, BW);
% d = size(DataGabor, 3);
% Data_tmp = reshape(DataGabor, nRow*nCol, d);
% Data_tmp = Data_tmp';
%% ѵ���������Լ��Ļ���
[train, test, nClass] = randomSampling(im_2d, gt, 'byPercent', 0.1);
% [train, test, nClass] = randomSampling(im_2d, gt, 'byNumber', 30);

%% ����

trainX = train.data;
trainY = train.label;
trainIndex = train.index;
testX = test.data;
testY = test.label;
testIndex = test.index;











% ϡ���ʾ
% K = 1;
% t1 = cputime;
% [classification_result] = SparseRepresentation(train, test, nClass, K);
% t2 = cputime-t1;
% fprintf('SRC1->    ����ʱ�䣺%.2fs;     OA=%.2f\n', t2, classification_result.OA);
% para = [1e-2];  % regularized parameter
% t1 = cputime;
% class = SRC_Classification_post(trainX', train.nEveryClass, testX',para);
% [SRC_OA,SRC_AA,SRC_kappa,SRC_CA] = confusion(testY, class');
% t2 = cputime-t1;
% fprintf('SRC2->    ����ʱ�䣺%.2fs;     OA=%.2f\n', t2, SRC_OA);





% ELM-kernel
% kerneloption = [5];
% c = 1024;
% [TTrain,TTest,TrainAC,accur_ELM,TY,label] = elm_kernel_classification(train,test,1,c,'RBF_kernel',kerneloption);
%  [OA,AA,kappa,CA] = confusion(testY, label);

% JSaCR
%  c  = 4;
% lambda = 0.01;
% gamma  = 1;
% class = JSaCR_Classification(trainX', train.nEveryClass, testX', lambda, c, gamma);
%  [OA,AA,kappa,CA] = confusion(testY, class);
 
% ���������JSR
% scale =5;  %Indian Pines:5 Salinas:14 PaviaU:19 PaviaC:3
% K = 2;
% t1 = cputime;
% [result, r1] = neighbor_JSR(img, train, testX, testY, testIndex, scale, K);
% % [result, pp] = neighbor_JSR(img, train, im_2d, gt(:)', [1: (nRow*nCol)], scale, K);
% [neighborJSR_OA, neighborJSR_AA, neighborJSR_kappa, neighborJSR_CA] = confusion(testY, result);
% t2 = cputime-t1;
% fprintf('neighborJSR->    ����ʱ�䣺%.2fs;     OA=%.4f\n', t2, neighborJSR_OA);


% ��������ͳ����ص�JSR
% scale = 14;
% K = 1;
% t1 = cputime;
% % [result, pp] = neighbor_JSR(img, train, testX, testY, testIndex, scale, K);
% [result, pp] = intersect_JSR(img, train, im_2d, gt(:)', [1: (nRow*nCol)], segment_result, scale, K, nList);
% [intersectJSR_OA, intersectJSR_AA, intersectJSR_kappa, intersectJSR_CA] = confusion(gt(:)', result);
% t2 = cputime-t1;
% fprintf('intersectJSR->    ����ʱ�䣺%.2fs;     OA=%.2f\n', t2, intersectJSR_OA);
% % +MRF
% P = reshape(pp', nRow, nCol, []);  
% beta = 2;
% [Nx Ny Nc] = size(P);
% Dc = -log(P+eps);
% Sc = ones(Nc) - eye(Nc);   
% gch = GraphCut('open', Dc, beta*Sc);
% [gch seg] = GraphCut('expand',gch);
% gch = GraphCut('close', gch);
% cc = seg+1;
% [MRF_OA, MRF_AA, MRF_kappa, MRF_CA] = confusion(gt(:), cc(:));
% fprintf('����MRF������->   OA=%.2f\n', MRF_OA);


%% ����������
% SLIC
% numSuperpixels = 300;
% fprintf('������������...\n');
% compactness = 0.3; % compactness2 = 1-compactness, the clustering is according to: compactness*dxy+compactness2*dspectral
% dist_type = 2; % distance type - 1:Euclidean��2��SAD; 3:SID; 4:SAD-SID
% seg_all = 1; % 1: all the pixles participate in the clustering�� 2: some pixels would not
% [superpixel_label, numlabels, seedx, seedy] = SuperpixelSegmentation(RgbImg, numSuperpixels);


% Entropy rate
% number_superpixels = 100;lambda_prime = 0.4;sigma = 10; conn8 = 1;
% superpixel_label = EntropyRate(superpixel_input, number_superpixels, lambda_prime, sigma,conn8);


% fast FCM
% fprintf('�������ں�...\n');
% [~,~,Num,centerLab]=Label_image(RgbImg, superpixel_label);
% segment_result = w_super_fcm(superpixel_label,centerLab,Num,cluster);
% segment_result = double(segment_result);

% ���ڳ����ص�JSR  
% K = 1;%
% index_map = reshape(1:size(im_2d,2),[nRow,nCol]);
% t1 = cputime;
% [superpixelJSR_OA,superpixelJSR_AA,superpixelJSR_kappa,superpixelJSR_CA] = superpixel_JSR(im_2d,  trainX, trainY, trainIndex, testX, testY, testIndex, superpixel_label,index_map, K, nList);
% t2 = cputime-t1;
% fprintf('superpixelJSR->    ����ʱ�䣺%.2fs;     OA=%.4f\n', t2, superpixelJSR_OA);

% t1 = cputime;
% [result, r2]= superpixel_JSR(im_2d,  trainX, trainY, trainIndex, testX, testY, testIndex, segment_result,index_map, K, nList);
% t2 = cputime-t1;
% [superpixelJSR_OA1,superpixelJSR_AA1,superpixelJSR_kappa1,superpixelJSR_CA1] = confusion(testY, result);
% fprintf('superpixelJSR->    ����ʱ�䣺%.2fs;     OA=%.4f\n', t2, superpixelJSR_OA1);
% save d1sp1150 segment_result;
% t1 = cputime;
% [result, r3]= SPJSR(im_2d,  trainX, trainY, trainIndex, testX, testY, testIndex, segment_result,index_map, K, nList);
% t2 = cputime-t1;
% [superpixelJSR_OA2,superpixelJSR_AA2,superpixelJSR_kappa2,superpixelJSR_CA2] = confusion(testY, result);
% fprintf('superpixelJSR->    ����ʱ�䣺%.2fs;     OA=%.4f\n', t2, superpixelJSR_OA2);

% jSR+MRF
% [result, pp] = superpixel_JSR_backgroud(im_2d, trainX, trainY, trainIndex, im_2d, gt(:),[1:size(im_2d,2)]' , superpixel_label,index_map, K);
% [OA,AA,kappa,CA] = confusion_backgroud(gt(:), result);


% CRC
% para = [1e-2];  % regularized parameter
% t1 = cputime;
% class = CRC_Classification_post(trainX', train.nEveryClass, testX',para);
% [CRC_OA,CRC_AA,CRC_kappa,CRC_CA] = confusion(testY, class');
% t2 = cputime-t1;
% fprintf('CRC->    ����ʱ�䣺%.2fs;     OA=%.2f\n', t2, CRC_OA);

% CRT
% para = [1e-2];  % regularized parameter
% t1 = cputime;
% class = CRT_Classification_post(trainX', train.nEveryClass, testX',para);
% [CRT_OA,CRT_AA,CRT_kappa,CRT_CA] = confusion(testY, class');
% t2 = cputime-t1;
% fprintf('CRT->    ����ʱ�䣺%.2fs;     OA=%.2f\n', t2, CRT_OA);



% KNN
% [best_k, knn_test_response]=calculate_best_k(trainY, trainX);
% result = k_nn_classifier(trainX, trainY, best_k, testX, testY);
% [OA,AA,kappa,CA] = confusion(testY, result);
% SVM
[C, P] = SVM_MG(testX', trainX', trainY, testY, 128, 10);
[SVM_OA, SVM_AA, SVM_kappa, SVM_CA] = confusion(testY, C);

%% ������ͼ

finalMap_1d = zeros(1, nRow*nCol);
for i = 1: length(trainIndex)
    finalMap_1d(trainIndex(i)) = trainY(i);
end
for i = 1: length(testIndex)
    finalMap_1d(testIndex(i)) = C(i);
end
finalMap = reshape(finalMap_1d, nRow, nCol);
map = label2color(finalMap, 'india');
figure;imshow(map);

map = label2color(gt, 'uni');
figure;imshow(map);
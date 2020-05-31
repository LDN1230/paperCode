clear all;
%% 包含文件目录
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

%% 导入数据集
nDataset =2 ;
if nDataset == 1
    load IndiaP.mat; 
    load Indian_pines_gt.mat;
    gt = indian_pines_gt;
    gtImg = label2color(gt, 'india');
    cluster = 16;
    numSuperpixels = 2000;
    scale =5;
    num = [12 140 83 20 50 73 5 50 5 100 250 60 20 130 40 10];
    fprintf('当前数据集：Indian Pines\n');
end
if nDataset == 2
    load Salinas_corrected.mat;
    load Salinas_gt.mat;
    img = salinas_corrected;
    gt = salinas_gt;
    cluster = 16;
    numSuperpixels = 1000;
    scale =14;
    num = [10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 ];
    fprintf('当前数据集：Salinas\n');
end
if nDataset == 3
    load PaviaU.mat;
    load PaviaU_gt.mat;
    img = paviaU;
    gt = paviaU_gt;
    cluster = 9;
     numSuperpixels = 3000;
     scale =19;
     fprintf('当前数据集：PaviaU\n');
end
if nDataset == 4
    load PaviaC.mat;
    load PaviaC_gt.mat;
    cluster = 9;
     numSuperpixels = 6000;
     scale =3;
     fprintf('当前数据集：PaviaC\n');
end
[nRow, nCol, nBand] = size(img);

% 各点邻域
[numN, nList] = getNeighFromGrid(nRow, nCol);

%% 高光谱图像-》伪彩色图像
RgbImg = hyperspectral2rgb(img, [3 30 87]);
% figure('Name', '伪彩色图像'); imshow(RgbImg);
%% 向量化、归一化
im_2d = ToVector(img)';
im_2d = im_2d./repmat(sqrt(sum(im_2d.*im_2d)),[size(im_2d,1) 1]);

%% 特征提取
% LBP特征
t1 = cputime;
fprintf('LBP特征提取：');
lbp_input = img(:,:,30);
r = 2;
nr = 8;
w0 = 11;
mapping = getmapping(nr,'u2');
[LBP_feature, lbp_img] = LBP_feature_global(lbp_input, r, nr, mapping, w0, gt);
t2 = cputime-t1;fprintf('所用时间%.2fs\n', t2);
LBP_feature = reshape(LBP_feature, nRow*nCol, size(LBP_feature,3));
LBP_feature = LBP_feature';
%% PCA 降维
[coeff score latent] = pca(im_2d');
pca_result = score(:,1:3);
superpixel_input =reshape(pca_result, nRow, nCol, 3);  %超像素分割的输入


   

    %% 产生超像素
    % SLIC

     
     fprintf("numSuperpixels=%d\n",numSuperpixels);
    fprintf('产生超像素中：'); t1 = cputime;
    compactness = 0.3; % compactness2 = 1-compactness, the clustering is according to: compactness*dxy+compactness2*dspectral
    dist_type = 2; % distance type - 1:Euclidean；2：SAD; 3:SID; 4:SAD-SID
    seg_all = 1; % 1: all the pixles participate in the clustering， 2: some pixels would not
    [superpixel_label, numlabels, seedx, seedy] = SuperpixelSegmentation(RgbImg, numSuperpixels);
    t2 = cputime-t1; fprintf('\t超像素个数：%d\t用时%.2fs\n', numlabels, t2); 
    
    % fast FCM
    fprintf('超像素融合:');t1 = cputime;
    [~,~,Num,centerLab]=Label_image(RgbImg, superpixel_label);
    segment_result = w_super_fcm(superpixel_label,centerLab,Num,cluster);
    segment_result = double(segment_result);
    [mergedSuperpixel, nFinalSuperpixel] = normalizedSuperpixel(segment_result, nList);
    t2 = cputime-t1; fprintf('融合后超像素个数：%d\t用时%.2fs\n',nFinalSuperpixel,  t2); 



% oa1 = zeros(1, 21);
% oa2 = zeros(1, 21);
% oa3 = zeros(1, 21);
% oa4 = zeros(1, 21);
% result =[];
% myJSR_OAs = [];
% myJSR_AAs = [];
% myJSR_CAs = [];
% myJSR_Kappas = [];
% testIndexs = [];
% trainIndexs = [];
% trainYs = [];

     
      %% 训练集、测试集的划分
%     p = 0.1;
%     [train, test, nClass] = randomSampling(im_2d, gt, 'byPercent', p);
%     fprintf('训练样本每类占%.1f%%\n', p*100);
    p = 60;
    [train, test, nClass] = randomSampling(im_2d, gt, 'byNumber', p);
    fprintf('训练样本每类%d个\n', p);
    %% 分类

    trainX = train.data;
    trainY = train.label;
    trainIndex = train.index;
    testX = test.data;
    testY = test.label;
    testIndex = test.index;
    
    % 基于邻域和超像素的JSR
%     scale =5;  %Indian Pines:5 Salinas:14 PaviaU:19 PaviaC:3
        K = 1;
%     t1 = cputime;
%     [result1, r1] = neighbor_JSR(img, train, testX, testY, testIndex, scale, K);
%     [NJSR_OA, NJSR_AA, NJSR_kappa, NJSR_CA] = confusion(testY, result1);
%     t2 = cputime-t1;
%     fprintf('NJSR->    运行时间：%.2fs;     OA=%.4f\n', t2, NJSR_OA);
    
%     t1 = cputime;
%     result  = my_superpixel_JSR(im_2d, trainX, trainY, trainIndex, testX, testY, testIndex, superpixel_label, K);
%     [SPJSR_OA, SPJSR_AA, SPJSR_kappa, SPJSR_CA] = confusion(testY, result);
%     t2 = cputime-t1;
%     fprintf('超像素融合前SPJSR->    运行时间：%.2fs;     OA=%.4f\n', t2, SPJSR_OA);
%     oa1(n) = SPJSR_OA*100;
%     
%      t1 = cputime;
%     result  = my_superpixel_JSR(im_2d, trainX, trainY, trainIndex, testX, testY, testIndex, mergedSuperpixel, K);
%     [SPJSR_OA, SPJSR_AA, SPJSR_kappa, SPJSR_CA] = confusion(testY, result);
%     t2 = cputime-t1;
%     fprintf('超像素融合后SPJSR->    运行时间：%.2fs;     OA=%.4f\n', t2, SPJSR_OA);
%     oa2(n) = SPJSR_OA*100;

% kk = [310:10:500];
 for n = 1: 1
    t1 = cputime;
    [result1] = myJSR(img, im_2d, train, testX, testY, testIndex, scale, K, mergedSuperpixel, nList, LBP_feature, 350);
    [myJSR_OA, myJSR_AA, myJSR_kappa, myJSR_CA] = confusion(testY, result1);
    t2 = cputime-t1;
    fprintf('myJSR->    运行时间：%.2fs;     OA=%.4f\n', t2, myJSR_OA);
%     [NJSR_OA, NJSR_AA, NJSR_kappa, NJSR_CA] = confusion(testY, result2);
%     fprintf('NJSR->    运行时间：%.2fs;     OA=%.4f\n', t2, NJSR_OA);
    oa3(n) = myJSR_OA*100;
%     oa4(n) = NJSR_OA*100;
%     result = [result result1(:)];
%     myJSR_OAs = [myJSR_OAs myJSR_OA];
%     myJSR_AAs = [myJSR_AAs myJSR_AA];
%     myJSR_CAs = [myJSR_CAs myJSR_CA];
%     myJSR_Kappas = [myJSR_Kappas myJSR_kappa];
%     testIndexs =[testIndexs  testIndex];
%     trainIndexs = [trainIndexs trainIndex];
%     trainYs = [trainYs trainY];
 end
 
% fprintf('\n');
% for i =1:length(oa1)
%     fprintf('%.2f, ', oa1(i));
% end
% fprintf('\n');
% for i =1:length(oa2)
%     fprintf('%.2f, ', oa2(i));
% end
fprintf('\n');
for i =1:length(oa3)
    fprintf('%.2f, ', oa3(i));
end
fprintf('\n');
% for i =1:length(oa4)
%     fprintf('%.2f, ', oa4(i));
% end
% fprintf('\n');



%% 画分类图
% mann = find(oa3==max(oa3));
% resultMap = result(:, mann);
% testIndex = testIndexs(:, mann);
% trainIndex = trainIndexs(:, mann);
% trainY = trainYs(:, mann);
% [xi,yi] = find(gt == 0);
% xisize = size(xi);
finalMap_1d = zeros(1, nRow*nCol);
for i = 1: length(trainIndex)
    finalMap_1d(trainIndex(i)) = trainY(i);
end
for i = 1: length(testIndex)
    finalMap_1d(testIndex(i)) = result1(i);
end
finalMap = reshape(finalMap_1d, nRow, nCol);
map = label2color(finalMap, 'india');
figure;imshow(map);

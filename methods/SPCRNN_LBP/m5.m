
%% 方法对比
close all;
clear all;clc;
addpath('../../dataset');
addpath('../../common');
addpath('../../superpixel');
addpath('../../feature/RF');
addpath('./Loopy Belief Propagation');


nDataset = 3;
if nDataset == 1
    load IndiaP;
    load Indian_pines_gt.mat;
    im_3d = img;     
    im_gt = indian_pines_gt;          
    C = 16;
    numSuperpixels = 10000;
    K_1 = 1; K_2 = 2;
    lambda = 0.01;
%     num = [5 143 83 24 48 73 3 48 2 97 246 59 21 127 39 9];
end
if nDataset == 2
    load Salinas_corrected.mat;
    load Salinas_gt.mat;
    im_3d = salinas_corrected;
    im_gt = salinas_gt;
    C = 16;
    numSuperpixels = 3000;
     K_1 = 1; K_2 = 16;
     lambda = 0.01;
     num = [30 30 30 30 30 30 30 30 30 30 30 30 30 30 30 30];
end
if nDataset == 3
    load PaviaU.mat;
    load PaviaU_gt.mat;
    im_3d = paviaU;
    im_gt = paviaU_gt;
    C = 9;
    numSuperpixels = 7000;
     K_1 = 1; K_2 = 19;
     lambda = 0.001;
     num = [100 100 100 100 100 100 100 100 100];
end
if nDataset == 4
    load PaviaC.mat;
    load PaviaC_gt.mat;
    im_3d = img;
    im_gt = gt;
    C = 9;
    numSuperpixels = 13000;
     K_1 = 1; K_2 = 1;
     lambda = 0.01;
     
end

[i_row, i_col] = size(im_gt);
im_gt_1d = reshape(im_gt,1,i_row*i_col);    
index_map = reshape(1:length(im_gt_1d),[i_row,i_col]); 
[r,c,b]=size(im_3d);
x=reshape(im_3d,[r*c b]);      
[x] = scale_new(x); %归一化
x1=reshape(x,[r c b]);  
% 提取特征
fimage=spatial_feature(x1,204,0.5);  
im_2d = ToVector(fimage)';          



%特征提取后，进行PCA降维
[coeff score latent] = pca(im_2d');
pca_result = score(:,1:3);
superpixel_data =reshape(pca_result,r,c,3);

%产生超像素
fprintf('超像素数目：%d\n', numSuperpixels);
compactness = 0.3; % compactness2 = 1-compactness, the clustering is according to: compactness*dxy+compactness2*dspectral
dist_type = 2; % distance type - 1:Euclidean；2：SAD; 3:SID; 4:SAD-SID
seg_all = 1; % 1: all the pixles participate in the clustering， 2: some pixels would not
[superpixel_label, numlabels, seedx, seedy] = SuperpixelSegmentation(superpixel_data, numSuperpixels);

                    
SPCRNN_result.OA = zeros(1,10);
SPCRNN_result.AA = zeros(1,10);
SPCRNN_result.kappa = zeros(1,10);
SPCRNN_result.CA = zeros(C, 10);
SPCRNNLBP_results.OA = zeros(1,10);
SPCRNNLBP_results.AA = zeros(1,10);
SPCRNNLBP_results.kappa = zeros(1,10);
SPCRNNLBP_results.CA = zeros(10, C);
trainIndexes = [];
testIndexes = [];
SPCRNN_test_results = [];
SPCRNNLBP_test_results = [];
D_labels = [];
for it = 1:10
    %训练集和测试集的选取
    fprintf('采集训练样本和测试样本\n');
    [trainIndex, testIndex] = randomSampling(im_gt, C, 'byPercent', 0.01);
%     [trainIndex, testIndex] = randomSampling(im_gt, C, 'byNumber', 30);
    tt_index = testIndex';%测试集样本的索引
    D_index = trainIndex';%训练集样本的索引
    D = im_2d(:,D_index);%训练集
    D_label = im_gt_1d(D_index);%训练集标签
    tt_data = im_2d(:,tt_index);%测试集
    tt_label = im_gt_1d(tt_index);%测试集标签

    data_all = [D,tt_data ];  %(200*10249)
    labels = [D_label,tt_label];
    %D = D./repmat(sqrt(sum(D.*D)),[size(D,1) 1]); %归一化
    label_result = zeros(size(tt_label)); %测试集的预测标签

    n_train = size(D_index,2);
    train_data_ori = data_all(:, 1:n_train);  %(200*1022)
    test_data_ori = data_all(:, (n_train+1):end);    %(200*9227)
    index = [1 : 1 : size(im_gt_1d,2)];
    
    trainIndexes = [trainIndexes trainIndex];
    testIndexes = [testIndexes testIndex];
    D_labels = [D_labels; D_label];
    
    [distance] = SPCRNN(im_2d, test_data_ori , tt_index, train_data_ori, D_label,superpixel_label, K_1,K_2,index,index_map, lambda);


    for i = 1:1:size(tt_index,2)
        residual = distance(:,tt_index(i))';
        temp = find(residual == min(residual));
        label_result(i) = temp(1); %与哪个类距离小的，该类就是样本点的标签
    end
    SPCRNN_test_results = [SPCRNN_test_results; label_result];
    [SPCRNN_result.OA(it), SPCRNN_result.AA(it), SPCRNN_result.kappa(it), SPCRNN_result.CA(:, it)]=confusion(tt_label,label_result);%已知真实标签和预测标签，计算准确度

    distance = distance +eps;
    pp = distance./repmat(sum(distance,1), [size(distance,1),1]);
    pp = 1-pp;

    nclass = max(D_label(:));

    p = pp;
    mu = 2;
    v0 = exp(mu);
    v1 = exp(0);

    psi = v1*ones(nclass, nclass);
    for i = 1:nclass
        psi(i,i) = v0;   
    end

    psi_temp = sum(psi);
    psi_temp = repmat(psi_temp, nclass, 1);
    psi = psi./psi_temp;
    p =p';
    [numN, nList] = getNeighFromGrid(i_row,i_col);
    trainY = [D_index; D_label];
    % belief propagation
    [belief] = BP_message(p,psi,nList,trainY);
    [maxb,classb] = max(belief);
    indexb = double(classb);
    mpm_results.map(1,:) = indexb;
    SPCRNNLBP_test_results = [SPCRNNLBP_test_results; indexb];
    [SPCRNNLBP_results.OA(it), SPCRNNLBP_results.kappa(it), SPCRNNLBP_results.AA(it),...
        SPCRNNLBP_results.CA(it,:)]= calcError(tt_label(1,:)-1, mpm_results.map(1,tt_index(1,:))-1,1:nclass);
%                     
    SPCRNN_OA = SPCRNN_result.OA(it);
    SPCRNN_AA = SPCRNN_result.AA(it);
    SPCRNN_Kappa = SPCRNN_result.kappa(it);
    SPCRNN_CA = SPCRNN_result.CA(:, it);
% 
    SPCRNNLBP_OA = SPCRNNLBP_results.OA(it);
    SPCRNNLBP_AA = SPCRNNLBP_results.AA(it);
    SPCRNNLBP_Kappa = SPCRNNLBP_results.kappa(it);
    SPCRNNLBP_CA = SPCRNNLBP_results.CA(it, :);
% 
%     fprintf('num = %d\n', num);
    fprintf("SPCRNN-OA: %.2f\n", 100*SPCRNN_OA);
    fprintf("SPCRNN-AA: %.2f\n", 100*SPCRNN_AA);
    fprintf("SPCRNN-Kappa: %.2f\n", 100*SPCRNN_Kappa);

    fprintf("SPCRNNLBP-OA: %.2f\n", 100*SPCRNNLBP_OA);
    fprintf("SPCRNNLBP-AA: %.2f\n", 100*SPCRNNLBP_AA);
    fprintf("SPCRNNLBP-Kappa: %.2f\n", 100*SPCRNNLBP_Kappa);

end
                    

                    


fprintf('SPCRNN_OA: \n [');
for i = 1:10
    fprintf('%.2f ', SPCRNN_result.OA(i)*100);
end
fprintf(']\nSPCRNN_AA:\n[');
for i = 1:10
    fprintf('%.2f ', SPCRNN_result.AA(i)*100);
end
fprintf(']\nSPCRNN_Kappa:\n[');
for i = 1:10
    fprintf('%.2f ', SPCRNN_result.kappa(i)*100);
end
fprintf(']\n');

fprintf('SPCRNNLBP_OA: \n [');
for i = 1:10
    fprintf('%.2f ', SPCRNNLBP_results.OA(i)*100);
end
fprintf(']\nSPCRNNLBP_AA:\n[');
for i = 1:10
    fprintf('%.2f ', SPCRNNLBP_results.AA(i)*100);
end
fprintf(']\nSPCRNNLBP_Kappa:\n[');
for i = 1:10
    fprintf('%.2f ', SPCRNNLBP_results.kappa(i)*100);
end
fprintf(']\n');



%% 画分类图
best = find(SPCRNN_result.OA==max(SPCRNN_result.OA));
best = 1;
best_trainIndex = trainIndexes(:, best);
best_testIndex = testIndexes(:, best);
best_D_label = D_labels(best, :);
SPCRNN_best_test_result = SPCRNN_test_results(best, :);
finalMap_1d = zeros(1, i_row*i_col);
for i = 1: length(best_trainIndex)
    finalMap_1d(best_trainIndex(i)) = best_D_label(i);
end
for i = 1: length(best_testIndex)
    finalMap_1d(best_testIndex(i)) = SPCRNN_best_test_result(i);
end
finalMap = reshape(finalMap_1d, i_row, i_col);
map = label2color(finalMap, 'uni');
figure;imshow(map);

best_SPCRNNLBP_test_result = SPCRNNLBP_test_results(best, :); 
hgfh = best_SPCRNNLBP_test_result(1, best_testIndex);
finalMap_1d = zeros(1, i_row*i_col);
for i = 1: length(best_trainIndex)
    finalMap_1d(best_trainIndex(i)) = best_D_label(i);
end
for i = 1: length(best_testIndex)
    finalMap_1d(best_testIndex(i)) = hgfh(i);
end
finalMap = reshape(finalMap_1d, i_row, i_col);
map = label2color(finalMap, 'uni');
figure;imshow(map);




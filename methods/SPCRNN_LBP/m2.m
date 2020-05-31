
%% K_1和K_2对实验的影响
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
    im_3d = img;     %(145*145*200)
    im_gt = indian_pines_gt;            %(145*145)
    num_tr = [10,143,83,34,48,23,2,28,2,150,246,60,21,127,35,10];
    C = 16;
    numSuperpixels = 10000;
end
if nDataset == 2
    load Salinas_corrected.mat;
    load Salinas_gt.mat;
    im_3d = salinas_corrected;
    im_gt = salinas_gt;
    num_tr = [40,76,38,26,52,79,70,225,124,21,21,38,18,20,140,36];
    C = 16;
     numSuperpixels = 3000;
end
if nDataset == 3
    load PaviaU.mat;
    load PaviaU_gt.mat;
    im_3d = paviaU;
    im_gt = paviaU_gt;
    C = 9;
    numSuperpixels = 7000;
end
if nDataset == 4
    load PaviaC.mat;
    load PaviaC_gt.mat;
    im_3d = img;
    im_gt = gt;
    C = 9;
    numSuperpixels = 13000;
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
% [coeff score latent] = pca(im_2d');
% pca_result = score(:,1:3);
% superpixel_data =reshape(pca_result,r,c,3);

%产生超像素
% fprintf('超像素分割\n');
% compactness = 0.3; % compactness2 = 1-compactness, the clustering is according to: compactness*dxy+compactness2*dspectral
% dist_type = 2; % distance type - 1:Euclidean；2：SAD; 3:SID; 4:SAD-SID
% seg_all = 1; % 1: all the pixles participate in the clustering， 2: some pixels would not
% [superpixel_label, numlabels, seedx, seedy] = SuperpixelSegmentation(superpixel_data, numSuperpixels);


%训练集和测试集的选取
% fprintf('采集训练样本和测试样本\n');
% [trainIndex, testIndex] = randomSampling(im_gt, C, 'byPercent', 0.01);
% [trainIndex, testIndex] = randomSampling(im_gt, C, 'byNumber', 20);
% save paviaC_trainIndex trainIndex;
% save paviaC_testIndex testIndex;
% save paviaC_superpixel_label;
load paviaU_trainIndex;
load paviaU_testIndex;
load paviaU_superpixel_label;
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

K12OA = zeros(1, 5);
% K_1 = [1: 1: 10];
% K_2 = [1: 1: 10];
for kk1 = 1: 1%length(K_1)
    for kk2 = 21: 25
            t1 = cputime;   
            [distance] = SPCRNN(im_2d, test_data_ori , tt_index, train_data_ori, D_label,superpixel_label, kk1,kk2,index,index_map, 0.5);

            for i = 1:1:size(tt_index,2)
                residual = distance(:,tt_index(i))';
                temp = find(residual == min(residual));
                label_result(i) = temp(1); %与哪个类距离小的，该类就是样本点的标签
            end
            [SPCRNN_result.OA, SPCRNN_result.AA, SPCRNN_result.kappa, SPCRNN_result.CA]=confusion(tt_label,label_result);%已知真实标签和预测标签，计算准确度
            K12OA(kk1, kk2-20) = roundn(SPCRNN_result.OA*100, -2);
            
%             distance = distance +eps;
%             pp = distance./repmat(sum(distance,1), [size(distance,1),1]);
%             pp = 1-pp;
% 
%             nclass = max(D_label(:));
% 
%             p = pp;
%             mu = 2;
%             v0 = exp(mu);
%             v1 = exp(0);
% 
%             psi = v1*ones(nclass, nclass);
%             for i = 1:nclass
%                 psi(i,i) = v0;   
%             end
% 
%             psi_temp = sum(psi);
%             psi_temp = repmat(psi_temp, nclass, 1);
%             psi = psi./psi_temp;
%             p =p';
%             [numN, nList] = getNeighFromGrid(i_row,i_col);
%             trainY = [D_index; D_label];
%             % belief propagation
%             [belief] = BP_message(p,psi,nList,trainY);
%             [maxb,classb] = max(belief);
%             indexb = double(classb);
%             mpm_results.map(1,:) = indexb;
% 
%             [SPCRNNLBP_results.OA, SPCRNNLBP_results.kappa, SPCRNNLBP_results.AA,...
%                 SPCRNNLBP_results.CA]= calcError(tt_label(1,:)-1, mpm_results.map(1,tt_index(1,:))-1,1:nclass);
            t2 = cputime -  t1;
            
            fprintf("K_1 = %d   K_2 = %d    SPCRNN-OA: %.2f     时间：%.2f秒 \n", kk1, kk2, 100*SPCRNN_result.OA, t2);
                
    end
end
                  

                    



%  save indian10n KNNSCR_result;
%  save indian_lbp10n LBP_KNNSCR_results;
% 
% 
% im_gt_1d(tt_index) = label_result; % mpm_results.map(1,tt_index(1,:));
% im_gt_1d_reshape = reshape(im_gt_1d,i_row, i_col);
% indian10n = label2color(im_gt_1d_reshape,'india');
% save indian10n_pic indian10n;
% 
% im_gt_1d(tt_index) = mpm_results.map(1,tt_index(1,:));
% im_gt_1d_reshape = reshape(im_gt_1d,i_row, i_col);
% indian_lbp10n = label2color(im_gt_1d_reshape,'india');
% save indian10n_pic indian_lbp10n;

% figure()
% imshow(indian_lbp10p);
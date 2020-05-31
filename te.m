
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
addpath('./methods/SPCRNN_LBP');

load IndiaP.mat; 
load Indian_pines_gt.mat;
gt = indian_pines_gt;
gtImg = label2color(gt, 'india');
cluster = 16;
numSuperpixels = 2000;
fprintf('��ǰ���ݼ���Salinas\n');
RgbImg = hyperspectral2rgb(img, [3 30 87]);
figure('Name', 'α��ɫͼ��'); imshow(RgbImg);
grayImage = RgbImg(:,:,1);
figure();imshow(grayImage);

[r,c,b]=size(img);
x=reshape(img,[r*c b]);      
[x] = scale_new(x); %��һ��
x1=reshape(x,[r c b]);  
% ��ȡ����
fimage=spatial_feature(x1,204,0.5);  
RFImage = fimage(:,:,1);
figure();imshow(gtImg);  

numSuperpixels = 300;
fprintf('������������...\n');
compactness = 0.3; % compactness2 = 1-compactness, the clustering is according to: compactness*dxy+compactness2*dspectral
dist_type = 2; % distance type - 1:Euclidean��2��SAD; 3:SID; 4:SAD-SID
seg_all = 1; % 1: all the pixles participate in the clustering�� 2: some pixels would not
[superpixel_label, numlabels, seedx, seedy] = SuperpixelSegmentation(RgbImg, numSuperpixels);

paintSuperpixelAdge(RgbImg, superpixel_label);

% LBP����
t1 = cputime;
fprintf('LBP������ȡ��......\n');
lbp_input = reshape(pca_result(:,1), 145, 145, 1);
r = 2;
nr = 8;
mapping = getmapping(nr,'u2');
[LBP_feature, lbp_img] = LBP_feature_global(lbp_input, 2, nr, mapping, 11, gt);
figure();imshow(uint8(lbp_img));  

d = size(LBP_feature, 3);
LBP_feature = reshape(LBP_feature, nRow*nCol, d);
LBP_feature = LBP_feature';
maxV = max(LBP_feature(:));
LBP_feature = LBP_feature./maxV;
t2 = cputime-t1;
fprintf('LBP������ȡ����ʱ��:%.fs\n', t2);


RgbImg = hyperspectral2rgb(fimage, [3 30 87]);



indd = find(mergedSuperpixel == mergedSuperpixel(220, 137));
indd1 = setdiff([1: size(im_2d,2)], indd);
RgbImg1 = RgbImg;
for i = 1:length(index8)
    index = index8(i);
    [r, c] = ind2sub([512 217], index);
    RgbImg1(r,c,1) = 0;
    RgbImg1(r,c,2) = 255;
    RgbImg1(r,c,3) = 0;
end
figure();imshow(RgbImg1);

indd2 = sub2ind([512 217], 197, 138);
lbpOfPixel = LBP_feature(:, indd2);
figure(); b = bar(lbpOfPixel);

251 138
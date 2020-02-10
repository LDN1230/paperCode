function DataGabor = Gabor_feature_extraction_PC(Data, BW)
% Mean feature extraction
% Data: mxnxD, W: Window size (even), Feature_P: mxnxD1
% P: number of PCs
%

N_PC = 10;
% % BW  = 1;
% [m n d] = size(Data);
% DataTest = reshape(Data, m*n, d);
% Psi = PCA_Train(DataTest', N_PC);
% DataTestN = DataTest*Psi;
% DataN = reshape(DataTestN, m, n, N_PC);
DataN = Data;
[m n d] = size(DataN);


lambda  = 16;
psi     = [0 pi/2];
gamma   = 0.5;
N       = 8;
DataGabor = zeros(m, n, N*N_PC);
for i = 1: N_PC
    img_in = DataN(:, :, i);
    bw = BW;
    theta   = 0;
    for n=1:N
        gb = gabor_fn(bw,gamma,psi(1),lambda,theta)...
            + 1i * gabor_fn(bw,gamma,psi(2),lambda,theta);
        % gb is the n-th gabor filter
        DataGabor(:,:,(i-1)*N+n)=abs(imfilter(img_in, gb, 'symmetric'));
        % filter output to the n-th channel
        theta = theta + 2*pi/N;
        % next orientation
    end
end

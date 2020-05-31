
function [distance] = SPCRNN(im_2d, test_data_ori, tt_index, train_data_ori, D_label,labels_1, K_1,K_2,index,index_map, lambda)

[~, n] = size(im_2d);
distance = zeros(max(D_label), n);
i_row = size(index_map,1);i_col = size(index_map,2);
distance_old = [];sum_distance = [];
sum_distance_end = [];
XtX = train_data_ori'*train_data_ori ;

% tic;
for p = 1: length(tt_index)
    if  mod(p, uint16(length(tt_index)/10)) == 0
        fprintf('******');
    end
   y_index = tt_index(p);
    if distance(1, y_index) == 0
%         fprintf("%d / %d\n", p,  length(tt_index));
        y_jonit_index = index_map((labels_1==labels_1(y_index)));
        y_jonit_index = intersect(y_jonit_index, tt_index);
        y_jonit_data = im_2d( :, y_jonit_index);
        for no = 1:size(y_jonit_data,2)
            y = y_jonit_data(:,no);
            norms = sum((train_data_ori - repmat(y, [1 size(train_data_ori,2)])).^2);
%             lambda = 0.5;
            G = diag(lambda.*norms);
            weight = (XtX+ G)\(train_data_ori'*y);%论文中的（4）
            tempd(no, :) = weight;

            for i = 1:max(D_label)
                X1 = train_data_ori(:, (D_label == i));
                tempdd = tempd(no, (D_label == i));
                [dd, ind] = sort(tempdd, 2, 'descend');
                if size(X1,2) > K_1
                    XX = X1(:, ind(1,1:K_1));
                    d = sqrt(sum((repmat(y, [1, K_1]) - XX).^2));
                    sum_distance(i,no) = sum(d) / K_1;
                else
                    d = sqrt(sum((repmat(y, [1, size(X1,2)]) - X1).^2));
                    sum_distance(i,no) = sum(d) / size(X1,2);
                end
                
            end
        end

        sort_d_1 = sort(sum_distance,2, 'ascend');
        for i = 1:size(sum_distance,1)
            if size(sum_distance,2)<K_2
                K2 = size(sum_distance,2);
            else
                K2 = K_2;
            end
            sort_d = sort_d_1(i,:);
            distance(i,y_jonit_index) = sum(sort_d(1:K2)) / K2;
        end
    else
%          fprintf("%d / %d\n", p, length(tt_index));
        continue;
    end
end
% toc;
fprintf('\n');
end

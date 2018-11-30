% G=E*-0.35+b 产生E and F and G
clear;clc;
data = csvread('train.csv');
data_EF = data(:,[5,6]);
num_to_oversample = size(data, 1) * 2; %*2最好，因为可能有的G生成的超过范围会被删除，所以多生成一些最后再删除
synthetic_samples_EFG = ones(size(data,1),size(data,2)); %保存合成样本的
min_G = min(data(:,7)); %G列的最大值和最小值，最后校验防止超过正常范围用
max_G = max(data(:,7));

for i=1:num_to_oversample
    
    % select the base instance randomly
    base_index = round(rand(1) * size(data, 1));
    while (base_index == 0)%防止生成0导致计算错误
        base_index = round(rand(1) * size(data, 1));
    end
    
    % find the nearest instance
    to_be_cal = [ones(size(data_EF,1),1)*data_EF(base_index,1),ones(size(data_EF,1),1)*data_EF(base_index,2)];%为了能用矩阵运算计算距离，所以生成一个跟数据集同样大小但是每个都是base样本的数组
    dis = floor(sqrt(sum((to_be_cal-data_EF).*(to_be_cal-data_EF),2)));%计算欧式距离
    dis_s = sort(dis);
    nearest_index = find(dis==dis_s(2)); nearest_index = nearest_index(1);%第一个是自己，要找第二近的
    synthetic_samples_EFG(i,1:4) = data(base_index,1:4);
    % oversampling
    rand_num = rand(1);
    synthetic_one = data_EF(base_index,:)+rand_num*(data_EF(nearest_index,:)-data_EF(base_index,:));%找最近邻的过采样xi1=xi+ζ1?(xi(nn)?xi)
    
    synthetic_samples_EFG(i,5:6) = synthetic_one;%保存
    % cal the b and G
    current_b = 0.35*data(nearest_index,5) + data(nearest_index,7);
    synthetic_samples_EFG(i, 7) = -0.35 * synthetic_one(1) + current_b;%G值计算
    
end

% delete the samples with sknew G value
index_to_delete = [];
for i = 1:size(synthetic_samples_EFG,1)
    if synthetic_samples_EFG(i,7)<min_G || synthetic_samples_EFG(i,7)>max_G%删除超过G值正常范围的那行样本
        index_to_delete = [index_to_delete,i];
    end
end
synthetic_samples_EFG(index_to_delete,:) = [];
% store the top n samples
synthetic_samples_EFG = floor(synthetic_samples_EFG(1:size(data,1),:));%结果在这里

while(size(data,1) ~= 0)
    i = round(rand(1)*size(data,1));
    while (i == 0)%防止生成0导致计算错误
        i = round(rand(1) * size(data, 1));
    end
    A = data(i,:);
    synthetic_samples_EFG = [synthetic_samples_EFG; data(i,:)];
    data(i,:) = [];
end


csvwrite('result.csv',synthetic_samples_EFG)
fprintf 'work done'

data = csvread('train.csv');
data_EF = data(:,[5,6]);
num_to_oversample = size(data, 1) * 2; %*2最好，因为可能有的G生成的超过范围会被删除，所以多生成一些最后再删除
synthetic_samples_EFG = ones(size(data,1),3); %保存合成样本的
min_G = min(data(:,7)); %G列的最大值和最小值，最后校验防止超过正常范围用
max_G = max(data(:,7));
base_index = 2;
to_be_cal = [ones(size(data_EF,1),1)*data_EF(base_index,1),ones(size(data_EF,1),1)*data_EF(base_index,2)];
dis = floor(sqrt(sum((to_be_cal-data_EF).*(to_be_cal-data_EF),2)));%计算欧式距离
dis_s = sort(dis);
nearest_index = find(dis==dis_s(2)); 
nearest_index = nearest_index(1);
rand_num = rand(1);
synthetic_one = data_EF(base_index,:)+rand_num*(data_EF(nearest_index,:)-data_EF(base_index,:));
synthetic_samples_EFG(1,1:2) = synthetic_one;%保存
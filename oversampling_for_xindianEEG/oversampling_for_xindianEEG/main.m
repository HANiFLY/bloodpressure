% G=E*-0.35+b ����E and F and G
clear;clc;
data = csvread('train.csv');
data_EF = data(:,[5,6]);
num_to_oversample = size(data, 1) * 2; %*2��ã���Ϊ�����е�G���ɵĳ�����Χ�ᱻɾ�������Զ�����һЩ�����ɾ��
synthetic_samples_EFG = ones(size(data,1),size(data,2)); %����ϳ�������
min_G = min(data(:,7)); %G�е����ֵ����Сֵ�����У���ֹ����������Χ��
max_G = max(data(:,7));

for i=1:num_to_oversample
    
    % select the base instance randomly
    base_index = round(rand(1) * size(data, 1));
    while (base_index == 0)%��ֹ����0���¼������
        base_index = round(rand(1) * size(data, 1));
    end
    
    % find the nearest instance
    to_be_cal = [ones(size(data_EF,1),1)*data_EF(base_index,1),ones(size(data_EF,1),1)*data_EF(base_index,2)];%Ϊ�����þ������������룬��������һ�������ݼ�ͬ����С����ÿ������base����������
    dis = floor(sqrt(sum((to_be_cal-data_EF).*(to_be_cal-data_EF),2)));%����ŷʽ����
    dis_s = sort(dis);
    nearest_index = find(dis==dis_s(2)); nearest_index = nearest_index(1);%��һ�����Լ���Ҫ�ҵڶ�����
    synthetic_samples_EFG(i,1:4) = data(base_index,1:4);
    % oversampling
    rand_num = rand(1);
    synthetic_one = data_EF(base_index,:)+rand_num*(data_EF(nearest_index,:)-data_EF(base_index,:));%������ڵĹ�����xi1=xi+��1?(xi(nn)?xi)
    
    synthetic_samples_EFG(i,5:6) = synthetic_one;%����
    % cal the b and G
    current_b = 0.35*data(nearest_index,5) + data(nearest_index,7);
    synthetic_samples_EFG(i, 7) = -0.35 * synthetic_one(1) + current_b;%Gֵ����
    
end

% delete the samples with sknew G value
index_to_delete = [];
for i = 1:size(synthetic_samples_EFG,1)
    if synthetic_samples_EFG(i,7)<min_G || synthetic_samples_EFG(i,7)>max_G%ɾ������Gֵ������Χ����������
        index_to_delete = [index_to_delete,i];
    end
end
synthetic_samples_EFG(index_to_delete,:) = [];
% store the top n samples
synthetic_samples_EFG = floor(synthetic_samples_EFG(1:size(data,1),:));%���������

while(size(data,1) ~= 0)
    i = round(rand(1)*size(data,1));
    while (i == 0)%��ֹ����0���¼������
        i = round(rand(1) * size(data, 1));
    end
    A = data(i,:);
    synthetic_samples_EFG = [synthetic_samples_EFG; data(i,:)];
    data(i,:) = [];
end


csvwrite('result.csv',synthetic_samples_EFG)
fprintf 'work done'

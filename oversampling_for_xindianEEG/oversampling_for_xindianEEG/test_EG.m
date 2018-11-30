clear;clc;
data = csvread('train.csv');
length = size(data,1);
for i = 1: length
    to_be_append = data(i,:);
    to_be_append(5) = to_be_append(5)+200;
    to_be_append(7) = to_be_append(7)-70;
    data = [data;to_be_append];
end
csvwrite('generated_data.csv',data)
    
    

